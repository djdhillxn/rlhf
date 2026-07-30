import copy
import gc
import hashlib
import inspect
import json
import math
import random
import shutil
from pathlib import Path

import torch

from .config import save_config
from .experiment import finalize_experiment, initialize_experiment
from .trl_common import (
    build_callbacks,
    build_lora_config,
    latest_trainer_checkpoint,
    load_tokenizer,
    maybe_sync_tree,
    resolve_resume_checkpoint,
    trainer_report_to,
    write_json,
)
from .trl_data import load_stage_dataset
from .trl_models import (
    apply_reward_center,
    configure_ppo_sampling_distribution,
    load_causal_model,
    load_reward_center,
    load_sequence_classification_model,
    merge_peft_model,
    remove_reward_center,
    score_tokenized_sequences,
)


DEFAULT_DOMAINS = ("code", "general", "stem", "multilingual")
EXACT_RESUME_MARKER = "exact_resume_complete.json"
RESUME_METADATA_NAME = "ppo_resume_metadata.json"
VALUE_STATE_NAME = "value_model.safetensors"
POLICY_ADAPTER_DIR = "policy_adapter"
DIAGNOSTICS_NAME = "ppo_diagnostics.jsonl"


def _quantile(values, probability):
    if not values:
        raise ValueError("Cannot compute a quantile from an empty sequence.")
    ordered = sorted(float(value) for value in values)
    position = min(1.0, max(0.0, float(probability))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def repeated_ngram_fraction(token_ids, ngram_size=4):
    tokens = [int(token) for token in token_ids]
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive.")
    count = len(tokens) - int(ngram_size) + 1
    if count <= 0:
        return 0.0
    ngrams = {tuple(tokens[index : index + ngram_size]) for index in range(count)}
    return 1.0 - len(ngrams) / count


def shape_terminal_reward(
    raw_reward,
    *,
    has_eos,
    repetition_fraction,
    lower_bound,
    upper_bound,
    repetition_threshold,
    repetition_penalty_scale,
):
    lower = float(lower_bound)
    upper = float(upper_bound)
    if not lower < upper:
        raise ValueError("Reward lower_bound must be smaller than upper_bound.")
    if not has_eos:
        return lower, 0.0, raw_reward < lower, raw_reward > upper

    clipped = min(upper, max(lower, float(raw_reward)))
    threshold = min(0.999999, max(0.0, float(repetition_threshold)))
    excess = max(0.0, float(repetition_fraction) - threshold)
    normalized_excess = excess / max(1.0 - threshold, 1e-8)
    penalty = (
        float(repetition_penalty_scale)
        * (upper - lower)
        * normalized_excess
        * normalized_excess
    )
    shaped = max(lower, clipped - penalty)
    return shaped, penalty, raw_reward < lower, raw_reward > upper


def build_balanced_rollout_plan(
    domain_values,
    *,
    total_updates,
    batch_size,
    seed,
    domains=DEFAULT_DOMAINS,
):
    domains = tuple(str(domain).lower() for domain in domains)
    if len(set(domains)) != len(domains):
        raise ValueError("rollout_balance.domains contains duplicate names.")
    if batch_size % len(domains) != 0:
        raise ValueError(
            f"Rollout batch size {batch_size} must be divisible by "
            f"{len(domains)} balanced domains."
        )

    pools = {domain: [] for domain in domains}
    for index, raw_domain in enumerate(domain_values):
        domain = str(raw_domain).lower()
        if domain in pools:
            pools[domain].append(index)
    missing = [domain for domain, indices in pools.items() if not indices]
    if missing:
        raise ValueError(
            "Prepared PPO data is missing required rollout domains: "
            + ", ".join(missing)
        )

    per_domain = batch_size // len(domains)
    streams = {}
    positions = {}
    cycles = {}

    def refill(domain):
        cycle = cycles.get(domain, 0)
        rows = list(pools[domain])
        random.Random(f"{int(seed)}:{domain}:{cycle}").shuffle(rows)
        streams[domain] = rows
        positions[domain] = 0
        cycles[domain] = cycle + 1

    for domain in domains:
        refill(domain)

    plan = []
    coverage = {domain: set() for domain in domains}
    for update in range(int(total_updates)):
        batch = []
        for domain in domains:
            for _ in range(per_domain):
                if positions[domain] >= len(streams[domain]):
                    refill(domain)
                index = streams[domain][positions[domain]]
                positions[domain] += 1
                batch.append(index)
                coverage[domain].add(index)
        random.Random(f"{int(seed)}:batch:{update}").shuffle(batch)
        plan.extend(batch)

    payload = json.dumps(plan, separators=(",", ":")).encode("utf-8")
    report = {
        "schema_version": 1,
        "seed": int(seed),
        "domains": list(domains),
        "total_updates": int(total_updates),
        "batch_size": int(batch_size),
        "examples_per_domain_per_update": int(per_domain),
        "plan_sha256": hashlib.sha256(payload).hexdigest(),
        "available_by_domain": {domain: len(pools[domain]) for domain in domains},
        "unique_examples_by_domain": {
            domain: len(coverage[domain]) for domain in domains
        },
    }
    return plan, report


def _stratified_indices(dataset, max_samples, domains, seed):
    domains = tuple(str(domain).lower() for domain in domains)
    groups = {domain: [] for domain in domains}
    for index, value in enumerate(dataset["domain"]):
        domain = str(value).lower()
        if domain in groups:
            groups[domain].append(index)
    missing = [domain for domain, values in groups.items() if not values]
    if missing:
        raise ValueError(
            "Reward calibration data is missing domains: " + ", ".join(missing)
        )

    quota = max(1, int(max_samples) // len(domains))
    selected = []
    for domain in domains:
        values = list(groups[domain])
        random.Random(f"{int(seed)}:guardrail:{domain}").shuffle(values)
        selected.extend(values[: min(quota, len(values))])
    random.Random(f"{int(seed)}:guardrail:combined").shuffle(selected)
    return selected[: int(max_samples)]


def _distribution_summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "min": min(values),
        "p01": _quantile(values, 0.01),
        "p05": _quantile(values, 0.05),
        "median": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values),
    }


def _guardrail_calibration_fingerprint(cfg):
    payload = {
        "reward_model_path": cfg["model"].get("reward_model_path"),
        "reward_center_path": cfg["model"].get("reward_center_path"),
        "cache_dir": cfg["data"].get("cache_dir"),
        "guardrails": dict(cfg.get("reward_guardrails", {})),
    }
    return _json_sha256(payload)


def _derive_reward_guardrails(
    cfg,
    *,
    reward_model,
    tokenizer,
    output_dir,
):
    guardrail_path = output_dir / "ppo_reward_guardrails.json"
    expected_fingerprint = _guardrail_calibration_fingerprint(cfg)
    if guardrail_path.is_file():
        saved = json.loads(guardrail_path.read_text(encoding="utf-8"))
        if saved.get("calibration_fingerprint") == expected_fingerprint:
            return saved
    checkpoint = latest_exact_ppo_checkpoint(output_dir)
    if checkpoint is not None:
        metadata_path = checkpoint / RESUME_METADATA_NAME
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            saved = metadata.get("reward_guardrails")
            if (
                isinstance(saved, dict)
                and saved.get("calibration_fingerprint") == expected_fingerprint
            ):
                write_json(saved, guardrail_path)
                return saved

    guard_cfg = cfg.get("reward_guardrails", {})
    reward_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"],
        "reward",
        str(guard_cfg.get("calibration_split", "train")),
    )
    domains = tuple(
        guard_cfg.get(
            "domains", cfg.get("rollout_balance", {}).get("domains", DEFAULT_DOMAINS)
        )
    )
    indices = _stratified_indices(
        reward_dataset,
        int(guard_cfg.get("calibration_max_pairs", 4096)),
        domains,
        int(cfg["train"].get("data_seed", cfg["train"].get("seed", 839))),
    )
    sample = reward_dataset.select(indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
    batch_size = int(guard_cfg.get("calibration_batch_size", 16))
    chosen = score_tokenized_sequences(
        reward_model,
        [list(row) for row in sample["chosen_ids"]],
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        batch_size=batch_size,
    )
    rejected = score_tokenized_sequences(
        reward_model,
        [list(row) for row in sample["rejected_ids"]],
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        batch_size=batch_size,
    )
    reward_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pooled_rewards = chosen.tolist() + rejected.tolist()
    lower_quantile = float(guard_cfg.get("lower_quantile", 0.005))
    upper_quantile = float(guard_cfg.get("upper_quantile", 0.995))
    lower_bound = _quantile(pooled_rewards, lower_quantile)
    upper_bound = _quantile(pooled_rewards, upper_quantile)
    minimum_span = float(guard_cfg.get("minimum_reward_span", 1.0))
    if upper_bound - lower_bound < minimum_span:
        midpoint = (upper_bound + lower_bound) / 2.0
        lower_bound = midpoint - minimum_span / 2.0
        upper_bound = midpoint + minimum_span / 2.0

    ngram_size = int(guard_cfg.get("repetition_ngram_size", 4))
    repetition = []
    for row in sample:
        prompt_length = int(row["prompt_length"])
        response_ids = list(row["chosen_ids"])[prompt_length:]
        if response_ids and response_ids[-1] == tokenizer.eos_token_id:
            response_ids = response_ids[:-1]
        repetition.append(repeated_ngram_fraction(response_ids, ngram_size))
    repetition_quantile = float(guard_cfg.get("repetition_threshold_quantile", 0.95))
    repetition_threshold = _quantile(repetition, repetition_quantile)

    report = {
        "schema_version": 1,
        "calibration_fingerprint": expected_fingerprint,
        "source": "stratified sample of HelpSteer3 reward-training pairs",
        "calibration_split": str(guard_cfg.get("calibration_split", "train")),
        "calibration_pairs": len(sample),
        "domains": list(domains),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "missing_eos_reward": lower_bound,
        "reward_distribution": _distribution_summary(pooled_rewards),
        "chosen_reward_distribution": _distribution_summary(chosen.tolist()),
        "rejected_reward_distribution": _distribution_summary(rejected.tolist()),
        "repetition_ngram_size": ngram_size,
        "repetition_threshold_quantile": repetition_quantile,
        "repetition_threshold": repetition_threshold,
        "repetition_distribution": _distribution_summary(repetition),
        "repetition_penalty_scale": float(
            guard_cfg.get("repetition_penalty_scale", 1.0)
        ),
    }
    write_json(report, guardrail_path)
    return report


class _RewardDiagnostics:
    def __init__(self, domains):
        self.domains = tuple(domains)
        self.rows = []

    def add(self, row):
        self.rows.append(dict(row))

    def consume(self):
        if not self.rows:
            return {}
        rows = self.rows
        self.rows = []

        def values(name, selected=rows):
            return [float(row[name]) for row in selected]

        raw = values("raw_reward")
        shaped = values("shaped_reward")
        lengths = values("response_length")
        repetition = values("repetition_fraction")
        penalties = values("repetition_penalty")
        count = len(rows)
        metrics = {
            "guardrail/rm_score_p05": _quantile(raw, 0.05),
            "guardrail/rm_score_p50": _quantile(raw, 0.50),
            "guardrail/rm_score_p95": _quantile(raw, 0.95),
            "guardrail/shaped_score_p05": _quantile(shaped, 0.05),
            "guardrail/shaped_score_p50": _quantile(shaped, 0.50),
            "guardrail/shaped_score_p95": _quantile(shaped, 0.95),
            "guardrail/reward_clipped_low_fraction": sum(
                bool(row["clipped_low"]) for row in rows
            )
            / count,
            "guardrail/reward_clipped_high_fraction": sum(
                bool(row["clipped_high"]) for row in rows
            )
            / count,
            "guardrail/reward_clipped_fraction": sum(
                bool(row["clipped_low"] or row["clipped_high"]) for row in rows
            )
            / count,
            "guardrail/repetition_penalty_mean": sum(penalties) / count,
            "guardrail/repetition_penalty_max": max(penalties),
            "rollout/response_length_mean": sum(lengths) / count,
            "rollout/response_length_p50": _quantile(lengths, 0.50),
            "rollout/response_length_p95": _quantile(lengths, 0.95),
            "rollout/eos_rate": sum(bool(row["has_eos"]) for row in rows) / count,
            "rollout/cap_rate": sum(not bool(row["has_eos"]) for row in rows) / count,
            "rollout/repeated_token_4gram_fraction_mean": sum(repetition) / count,
            "rollout/repeated_token_4gram_fraction_p50": _quantile(repetition, 0.50),
            "rollout/repeated_token_4gram_fraction_p95": _quantile(repetition, 0.95),
            "rollout/repetition_guardrail_rate": sum(
                bool(row["repetition_triggered"]) for row in rows
            )
            / count,
        }
        for domain in self.domains:
            selected = [row for row in rows if row["domain"] == domain]
            if not selected:
                continue
            domain_count = len(selected)
            domain_raw = values("raw_reward", selected)
            domain_shaped = values("shaped_reward", selected)
            domain_lengths = values("response_length", selected)
            domain_repetition = values("repetition_fraction", selected)
            metrics[f"domain/{domain}/count"] = domain_count
            metrics[f"domain/{domain}/rm_score_mean"] = sum(domain_raw) / domain_count
            metrics[f"domain/{domain}/rm_score_p05"] = _quantile(domain_raw, 0.05)
            metrics[f"domain/{domain}/rm_score_p50"] = _quantile(domain_raw, 0.50)
            metrics[f"domain/{domain}/rm_score_p95"] = _quantile(domain_raw, 0.95)
            metrics[f"domain/{domain}/shaped_score_mean"] = (
                sum(domain_shaped) / domain_count
            )
            metrics[f"domain/{domain}/shaped_score_p05"] = _quantile(
                domain_shaped, 0.05
            )
            metrics[f"domain/{domain}/shaped_score_p50"] = _quantile(
                domain_shaped, 0.50
            )
            metrics[f"domain/{domain}/shaped_score_p95"] = _quantile(
                domain_shaped, 0.95
            )
            metrics[f"domain/{domain}/reward_clipped_fraction"] = (
                sum(bool(row["clipped_low"] or row["clipped_high"]) for row in selected)
                / domain_count
            )
            metrics[f"domain/{domain}/repetition_penalty_mean"] = (
                sum(values("repetition_penalty", selected)) / domain_count
            )
            metrics[f"domain/{domain}/eos_rate"] = (
                sum(bool(row["has_eos"]) for row in selected) / domain_count
            )
            metrics[f"domain/{domain}/cap_rate"] = (
                sum(not bool(row["has_eos"]) for row in selected) / domain_count
            )
            metrics[f"domain/{domain}/response_length_mean"] = (
                sum(domain_lengths) / domain_count
            )
            metrics[f"domain/{domain}/response_length_p95"] = _quantile(
                domain_lengths, 0.95
            )
            metrics[f"domain/{domain}/repeated_token_4gram_fraction_mean"] = (
                sum(domain_repetition) / domain_count
            )
            metrics[f"domain/{domain}/repeated_token_4gram_fraction_p95"] = _quantile(
                domain_repetition, 0.95
            )
            metrics[f"domain/{domain}/repetition_guardrail_rate"] = (
                sum(bool(row["repetition_triggered"]) for row in selected)
                / domain_count
            )
            metrics[f"domain/{domain}/reward_clipped_low_fraction"] = (
                sum(bool(row["clipped_low"]) for row in selected) / domain_count
            )
            metrics[f"domain/{domain}/reward_clipped_high_fraction"] = (
                sum(bool(row["clipped_high"]) for row in selected) / domain_count
            )
        return metrics


def _patch_trl_generate_for_fixed_length(ppo_module):
    """Sample the configured rollout length before truncating at the first EOS."""
    if getattr(ppo_module, "_rlhf_fixed_length_generate_patch", False):
        return
    if not hasattr(ppo_module, "generate"):
        raise RuntimeError(
            "TRL PPO module does not expose generate; cannot apply fixed-length EOS patch."
        )
    original_generate = ppo_module.generate

    def generate_without_eos_stop(
        lm_backbone, queries, pad_token_id, generation_config
    ):
        generation_config = copy.deepcopy(generation_config)
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        return original_generate(lm_backbone, queries, pad_token_id, generation_config)

    ppo_module.generate = generate_without_eos_stop
    ppo_module._rlhf_fixed_length_generate_patch = True


def _query_key(token_ids, pad_token_id):
    return tuple(int(token) for token in token_ids if int(token) != int(pad_token_id))


def _patch_trl_reward_guardrails(ppo_module):
    """Bound generated rewards and replace invalid EOS/repetition outcomes."""
    if getattr(ppo_module, "_rlhf_reward_guardrail_patch", False):
        return
    if not hasattr(ppo_module, "get_reward"):
        raise RuntimeError(
            "TRL PPO module does not expose get_reward; cannot apply reward guardrails."
        )
    original_get_reward = ppo_module.get_reward

    def guarded_get_reward(model, query_responses, pad_token_id, context_length):
        reward_logits, final_rewards, sequence_lengths = original_get_reward(
            model,
            query_responses,
            pad_token_id,
            context_length,
        )
        settings = getattr(model, "rlhf_reward_guardrails", None)
        collector = getattr(model, "rlhf_reward_diagnostics", None)
        if not settings or collector is None:
            return reward_logits, final_rewards, sequence_lengths

        responses = query_responses[:, context_length:]
        shaped = final_rewards.clone()
        eos_token_id = int(settings["eos_token_id"])
        ngram_size = int(settings["repetition_ngram_size"])
        query_domains = settings["query_domains"]
        for index in range(responses.shape[0]):
            response = [
                int(token)
                for token in responses[index].detach().cpu().tolist()
                if int(token) != int(pad_token_id)
            ]
            try:
                eos_index = response.index(eos_token_id)
                has_eos = True
                response_for_metrics = response[:eos_index]
                response_length = eos_index + 1
            except ValueError:
                has_eos = False
                response_for_metrics = response
                response_length = len(response)

            repetition = repeated_ngram_fraction(
                response_for_metrics, ngram_size=ngram_size
            )
            raw_reward = float(final_rewards[index].detach().float().cpu().item())
            shaped_reward, penalty, clipped_low, clipped_high = shape_terminal_reward(
                raw_reward,
                has_eos=has_eos,
                repetition_fraction=repetition,
                lower_bound=settings["lower_bound"],
                upper_bound=settings["upper_bound"],
                repetition_threshold=settings["repetition_threshold"],
                repetition_penalty_scale=settings["repetition_penalty_scale"],
            )
            shaped[index] = shaped_reward
            query = query_responses[index, :context_length].detach().cpu().tolist()
            domain = query_domains.get(_query_key(query, pad_token_id), "unknown")
            collector.add(
                {
                    "domain": domain,
                    "raw_reward": raw_reward,
                    "shaped_reward": shaped_reward,
                    "clipped_low": clipped_low,
                    "clipped_high": clipped_high,
                    "has_eos": has_eos,
                    "response_length": response_length,
                    "repetition_fraction": repetition,
                    "repetition_penalty": penalty,
                    "repetition_triggered": (
                        has_eos and repetition > float(settings["repetition_threshold"])
                    ),
                }
            )
        return reward_logits, shaped, sequence_lengths

    ppo_module.get_reward = guarded_get_reward
    ppo_module._rlhf_reward_guardrail_patch = True


def _resume_fingerprint(cfg):
    train_keys = (
        "seed",
        "data_seed",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "adam_epsilon",
        "weight_decay",
        "max_grad_norm",
        "warmup_ratio",
        "lr_scheduler_type",
        "optim",
        "bf16",
        "fp16",
        "tf32",
        "gradient_checkpointing",
    )
    payload = {
        "model": dict(cfg.get("model", {})),
        "data": {
            "train_split": cfg.get("data", {}).get("train_split", "train"),
            "max_train_samples": cfg.get("data", {}).get("max_train_samples"),
        },
        "lora": dict(cfg.get("lora", {})),
        "ppo": dict(cfg.get("ppo", {})),
        "reward_guardrails": dict(cfg.get("reward_guardrails", {})),
        "rollout_balance": dict(cfg.get("rollout_balance", {})),
        "train": {
            key: cfg.get("train", {}).get(key)
            for key in train_keys
            if key in cfg.get("train", {})
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_trainer_state(checkpoint):
    path = Path(checkpoint) / "trainer_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing trainer state in PPO checkpoint: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def latest_exact_ppo_checkpoint(output_dir):
    candidates = []
    for path in Path(output_dir).glob("checkpoint-*"):
        if not path.is_dir() or not (path / EXACT_RESUME_MARKER).is_file():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if (path / "trainer_state.json").is_file():
            candidates.append((step, path))
    return max(candidates, default=(None, None))[1]


def _resolve_ppo_resume_checkpoint(output_dir, configured):
    if configured in {None, "", False, "none", "null"}:
        return None
    if str(configured).strip().lower() == "auto":
        exact = latest_exact_ppo_checkpoint(output_dir)
        generic = latest_trainer_checkpoint(output_dir)
        if generic is not None and generic != exact:
            if exact is None:
                print(
                    "Ignoring an incomplete PPO checkpoint; no verified exact "
                    "resume checkpoint exists yet."
                )
            else:
                print(
                    "Ignoring a newer incomplete PPO checkpoint and resuming from "
                    f"the latest verified checkpoint: {exact}"
                )
        return exact
    return resolve_resume_checkpoint(output_dir, configured)


def _restore_checkpoint_run_artifacts(checkpoint, output_dir):
    if checkpoint is None:
        return
    source_dir = Path(checkpoint) / "run_artifacts"
    if not source_dir.is_dir():
        return
    output_dir = Path(output_dir)
    for source in source_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)


def _validate_resume_checkpoint(
    checkpoint,
    *,
    cfg,
    plan_report,
    guardrails,
):
    checkpoint = Path(checkpoint)
    marker_path = checkpoint / EXACT_RESUME_MARKER
    metadata_path = checkpoint / RESUME_METADATA_NAME
    required = (
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        checkpoint / "rng_state.pth",
        checkpoint / "trainer_state.json",
        checkpoint / POLICY_ADAPTER_DIR,
        checkpoint / VALUE_STATE_NAME,
        marker_path,
        metadata_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "PPO checkpoint is not exact-resume capable. Missing:\n- "
            + "\n- ".join(missing)
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "resume_fingerprint": _resume_fingerprint(cfg),
        "rollout_plan_sha256": plan_report["plan_sha256"],
        "reward_guardrails_sha256": _json_sha256(guardrails),
    }
    mismatches = {
        key: {"checkpoint": metadata.get(key), "current": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Refusing an inexact PPO resume because immutable run settings changed: "
            + json.dumps(mismatches, indent=2)
        )
    return metadata


def _json_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _restore_rng_state(path):
    import numpy as np

    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and "cuda" in state:
        cuda_state = state["cuda"]
        if isinstance(cuda_state, list):
            torch.cuda.random.set_rng_state_all(cuda_state)
        else:
            torch.cuda.random.set_rng_state(cuda_state)


def _restore_exact_trainer_state(trainer, checkpoint):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file, load_model

    checkpoint = Path(checkpoint)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    adapter_file = checkpoint / POLICY_ADAPTER_DIR / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"Missing policy adapter state: {adapter_file}")
    adapter_state = load_file(str(adapter_file), device="cpu")
    incompatible = set_peft_model_state_dict(
        unwrapped.policy, adapter_state, adapter_name="default"
    )
    if getattr(incompatible, "unexpected_keys", None):
        raise RuntimeError(
            "Unexpected policy adapter keys during PPO resume: "
            + ", ".join(incompatible.unexpected_keys[:20])
        )
    load_model(
        unwrapped.value_model,
        str(checkpoint / VALUE_STATE_NAME),
        strict=True,
        device="cpu",
    )

    optimizer_state = torch.load(
        checkpoint / "optimizer.pt", map_location="cpu", weights_only=True
    )
    scheduler_state = torch.load(
        checkpoint / "scheduler.pt", map_location="cpu", weights_only=True
    )
    trainer.optimizer.load_state_dict(optimizer_state)
    trainer.lr_scheduler.load_state_dict(scheduler_state)
    _restore_rng_state(checkpoint / "rng_state.pth")
    return _read_trainer_state(checkpoint)


def _make_segment_callbacks(
    *,
    start_state,
    planned_updates,
    target_update,
    train_dataset_len,
    resume_metadata,
    guardrails,
):
    from transformers import TrainerCallback

    class SegmentStateCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            local_zero = state.is_local_process_zero
            world_zero = state.is_world_process_zero
            if start_state:
                for key, value in start_state.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
            state.is_local_process_zero = local_zero
            state.is_world_process_zero = world_zero
            state.max_steps = int(planned_updates)
            state.num_train_epochs = float(args.total_episodes) / train_dataset_len
            state.logging_steps = int(args.logging_steps)
            state.save_steps = int(args.save_steps)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if int(state.global_step) == int(target_update):
                control.should_save = True
            return control

    class ExactCheckpointCallback(TrainerCallback):
        trainer = None

        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            if self.trainer is None:
                raise RuntimeError("Exact PPO checkpoint callback has no trainer.")
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            unwrapped = self.trainer.accelerator.unwrap_model(self.trainer.model)
            adapter_dir = checkpoint / POLICY_ADAPTER_DIR
            unwrapped.policy.save_pretrained(adapter_dir, safe_serialization=True)
            from safetensors.torch import save_model

            save_model(
                unwrapped.value_model,
                str(checkpoint / VALUE_STATE_NAME),
            )
            metadata = {
                **resume_metadata,
                "completed_update": int(state.global_step),
                "completed_episodes": int(state.episode),
                "target_update_for_process": int(target_update),
                "reward_guardrails": guardrails,
            }
            write_json(metadata, checkpoint / RESUME_METADATA_NAME)
            artifact_dir = checkpoint / "run_artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                DIAGNOSTICS_NAME,
                "ppo_reward_guardrails.json",
                "ppo_rollout_plan.json",
                "ppo_segment_plan.json",
                "ppo_eos_trick.json",
                "ppo_sampling_distribution.json",
            ):
                source = Path(args.output_dir) / name
                if source.is_file():
                    shutil.copy2(source, artifact_dir / name)
            required = (
                checkpoint / "optimizer.pt",
                checkpoint / "scheduler.pt",
                checkpoint / "rng_state.pth",
                checkpoint / "trainer_state.json",
                checkpoint / POLICY_ADAPTER_DIR / "adapter_model.safetensors",
                checkpoint / VALUE_STATE_NAME,
                checkpoint / RESUME_METADATA_NAME,
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise RuntimeError(
                    "Refusing to mark an incomplete PPO checkpoint as resumable:\n- "
                    + "\n- ".join(missing)
                )
            write_json(
                {
                    "schema_version": 1,
                    "global_step": int(state.global_step),
                    "episode": int(state.episode),
                    "required_files_verified": True,
                },
                checkpoint / EXACT_RESUME_MARKER,
            )
            health = _checkpoint_health(checkpoint)
            write_json(health, checkpoint / "checkpoint_health.json")
            write_json(
                health,
                Path(args.output_dir) / f"checkpoint-{state.global_step}-health.json",
            )
            return control

    segment_callback = SegmentStateCallback()
    exact_callback = ExactCheckpointCallback()
    return segment_callback, exact_callback


def _install_diagnostics_logging(trainer, collector, output_dir):
    diagnostics_path = Path(output_dir) / DIAGNOSTICS_NAME
    original_log = trainer.log

    def log_with_diagnostics(logs, *args, **kwargs):
        logs = dict(logs)
        diagnostics = collector.consume()
        if diagnostics:
            logs.update(diagnostics)
            mean_length = diagnostics.get("rollout/response_length_mean", 0.0)
            if mean_length > 0 and "objective/kl" in logs:
                logs["objective/kl_per_response_token"] = (
                    float(logs["objective/kl"]) / mean_length
                )
            record = {
                "update": int(trainer.state.global_step),
                "episode": int(trainer.state.episode),
                **logs,
            }
            with diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return original_log(logs, *args, **kwargs)

    trainer.log = log_with_diagnostics


def _replace_with_segment_dataloader(trainer, segment_dataset):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        segment_dataset,
        batch_size=trainer.local_dataloader_batch_size,
        shuffle=False,
        collate_fn=trainer.data_collator,
        drop_last=True,
    )
    trainer.dataloader = trainer.accelerator.prepare(dataloader)
    trainer.train_dataset = segment_dataset


def _checkpoint_health(checkpoint):
    checkpoint = Path(checkpoint)
    files = {}
    for relative in (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        f"{POLICY_ADAPTER_DIR}/adapter_model.safetensors",
        VALUE_STATE_NAME,
        RESUME_METADATA_NAME,
        EXACT_RESUME_MARKER,
    ):
        path = checkpoint / relative
        files[relative] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "checkpoint": str(checkpoint),
        "healthy": all(value["exists"] for value in files.values()),
        "files": files,
    }


def export_ppo_policy(cfg, checkpoint, output_dir):
    from peft import PeftModel

    checkpoint = Path(checkpoint)
    marker = checkpoint / EXACT_RESUME_MARKER
    adapter_dir = checkpoint / POLICY_ADAPTER_DIR
    if not marker.is_file() or not adapter_dir.is_dir():
        raise RuntimeError(
            f"Cannot export an incomplete guarded PPO checkpoint: {checkpoint}"
        )
    output_dir = Path(output_dir)
    tokenizer = load_tokenizer(
        str(cfg["model"]["policy_model_path"]),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        padding_side="left",
    )
    base_policy = load_causal_model(
        str(cfg["model"]["policy_model_path"]), tokenizer, cfg["model"]
    )
    policy = PeftModel.from_pretrained(
        base_policy,
        str(adapter_dir),
        is_trainable=False,
    )
    merge_peft_model(policy, output_dir, tokenizer)
    metadata = json.loads(
        (checkpoint / RESUME_METADATA_NAME).read_text(encoding="utf-8")
    )
    write_json(
        {
            "schema_version": 1,
            "selected_checkpoint": str(checkpoint),
            "selected_update": int(metadata["completed_update"]),
            "selected_episodes": int(metadata["completed_episodes"]),
            "selection_is_external": True,
        },
        output_dir / "ppo_checkpoint_selection.json",
    )
    return output_dir


def prepare_ppo_guardrails(cfg):
    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(
        str(cfg["model"]["policy_model_path"]),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        padding_side="left",
    )
    reward_model = load_sequence_classification_model(
        str(cfg["model"]["reward_model_path"]), tokenizer, cfg["model"]
    )
    reward_offset = load_reward_center(cfg["model"].get("reward_center_path"))
    apply_reward_center(reward_model, reward_offset)
    report = _derive_reward_guardrails(
        cfg,
        reward_model=reward_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )
    write_json(report, output_dir / "ppo_reward_guardrails.json")
    maybe_sync_tree(output_dir, cfg["train"].get("final_sync_dir"))
    return output_dir / "ppo_reward_guardrails.json"


def run_trl_ppo(cfg, *, config_path=None):
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    ppo_trainer_module = inspect.getmodule(PPOTrainer)
    if ppo_trainer_module is None:
        raise RuntimeError("Could not locate the TRL PPO trainer module.")

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="trl_ppo",
        config_path=config_path,
        extra={"trl_backend": True, "model_name": cfg["model"]["policy_model_path"]},
    )

    tokenizer = load_tokenizer(
        str(cfg["model"]["policy_model_path"]),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        padding_side="left",
    )
    policy = load_causal_model(
        str(cfg["model"]["policy_model_path"]), tokenizer, cfg["model"]
    )
    sampling_distribution = configure_ppo_sampling_distribution(
        policy,
        temperature=float(cfg["ppo"].get("temperature", 0.7)),
    )
    write_json(sampling_distribution, output_dir / "ppo_sampling_distribution.json")

    reference_path = str(
        cfg["model"].get("reference_model_path", cfg["model"]["policy_model_path"])
    )
    if (
        Path(reference_path).resolve()
        == Path(str(cfg["model"]["policy_model_path"])).resolve()
    ):
        reference = None
    else:
        reference = load_causal_model(reference_path, tokenizer, cfg["model"])
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    reward_path = str(cfg["model"]["reward_model_path"])
    value_path = str(cfg["model"].get("value_model_path", reward_path))
    reward_model = load_sequence_classification_model(
        reward_path, tokenizer, cfg["model"]
    )
    value_model = load_sequence_classification_model(
        value_path, tokenizer, cfg["model"]
    )
    reward_offset = load_reward_center(cfg["model"].get("reward_center_path"))
    apply_reward_center(reward_model, reward_offset)
    apply_reward_center(value_model, reward_offset)

    full_train_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"], "ppo", cfg["data"].get("train_split", "train")
    )
    eval_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"], "ppo", cfg["data"].get("eval_split", "validation")
    )
    if cfg["data"].get("max_train_samples"):
        full_train_dataset = full_train_dataset.select(
            range(
                min(
                    len(full_train_dataset),
                    int(cfg["data"]["max_train_samples"]),
                )
            )
        )
    if cfg["data"].get("max_eval_samples"):
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), int(cfg["data"]["max_eval_samples"])))
        )

    lora_cfg = dict(cfg.get("lora", {}))
    if float(lora_cfg.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("The TRL PPO policy requires lora_dropout: 0.0.")

    train_cfg = cfg["train"]
    ppo_cfg = cfg["ppo"]
    guard_cfg = cfg.get("reward_guardrails", {})
    balance_cfg = cfg.get("rollout_balance", {})
    if not bool(guard_cfg.get("enabled", True)):
        raise ValueError("The guarded PPO run requires reward_guardrails.enabled=true.")
    if bool(ppo_cfg.get("whiten_rewards", False)):
        raise ValueError(
            "This guarded PPO run requires ppo.whiten_rewards=false so terminal "
            "reward bounds and EOS replacement retain their calibrated scale."
        )
    if not bool(ppo_cfg.get("fixed_length_generation", False)):
        raise ValueError("Guarded PPO requires ppo.fixed_length_generation=true.")
    if not bool(ppo_cfg.get("require_eos_for_reward", False)):
        raise ValueError("Guarded PPO requires ppo.require_eos_for_reward=true.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Guarded PPO requires tokenizer.eos_token_id.")

    _patch_trl_generate_for_fixed_length(ppo_trainer_module)
    _patch_trl_reward_guardrails(ppo_trainer_module)
    guardrails = _derive_reward_guardrails(
        cfg,
        reward_model=reward_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )

    batch_size = int(train_cfg.get("per_device_train_batch_size", 2)) * int(
        train_cfg.get("gradient_accumulation_steps", 32)
    )
    total_episodes = int(ppo_cfg.get("total_episodes", 12000))
    planned_updates = math.ceil(total_episodes / batch_size)
    domains = tuple(balance_cfg.get("domains", DEFAULT_DOMAINS))
    if not bool(balance_cfg.get("enabled", True)):
        raise ValueError("The guarded PPO run requires rollout_balance.enabled=true.")
    plan, plan_report = build_balanced_rollout_plan(
        full_train_dataset["domain"],
        total_updates=planned_updates,
        batch_size=batch_size,
        seed=int(train_cfg.get("data_seed", train_cfg.get("seed", 839))),
        domains=domains,
    )
    write_json(plan_report, output_dir / "ppo_rollout_plan.json")

    resume_checkpoint = _resolve_ppo_resume_checkpoint(
        output_dir, train_cfg.get("resume_from_checkpoint")
    )
    _restore_checkpoint_run_artifacts(resume_checkpoint, output_dir)
    start_state = _read_trainer_state(resume_checkpoint) if resume_checkpoint else None
    start_update = int(start_state.get("global_step", 0)) if start_state else 0
    target_update = int(train_cfg.get("target_update", planned_updates))
    if target_update <= start_update:
        raise ValueError(
            f"target_update={target_update} must exceed completed update {start_update}."
        )
    if target_update > planned_updates:
        raise ValueError(
            f"target_update={target_update} exceeds the planned {planned_updates} updates."
        )
    segment_updates = target_update - start_update
    segment_indices = plan[start_update * batch_size : target_update * batch_size]
    segment_dataset = full_train_dataset.select(segment_indices).select_columns(
        ["input_ids"]
    )
    eval_dataset = eval_dataset.select_columns(["input_ids"])

    query_domains = {}
    for row in full_train_dataset:
        key = _query_key(row["input_ids"], tokenizer.pad_token_id)
        domain = str(row.get("domain", "unknown")).lower()
        previous = query_domains.get(key)
        if previous is not None and previous != domain:
            raise RuntimeError("Identical PPO prompt IDs map to multiple domains.")
        query_domains[key] = domain

    collector = _RewardDiagnostics(domains)
    guardrail_settings = {
        "eos_token_id": int(tokenizer.eos_token_id),
        "lower_bound": float(guardrails["lower_bound"]),
        "upper_bound": float(guardrails["upper_bound"]),
        "repetition_ngram_size": int(guardrails["repetition_ngram_size"]),
        "repetition_threshold": float(guardrails["repetition_threshold"]),
        "repetition_penalty_scale": float(guardrails["repetition_penalty_scale"]),
        "query_domains": query_domains,
    }
    setattr(reward_model, "rlhf_reward_guardrails", guardrail_settings)
    setattr(reward_model, "rlhf_reward_diagnostics", collector)
    eos_trick = {
        "fixed_length_generation": True,
        "require_eos_for_reward": True,
        "missing_eos_reward": float(guardrails["missing_eos_reward"]),
        "source": "reward calibration lower quantile boundary",
        "eos_token_id": int(tokenizer.eos_token_id),
    }
    write_json(eos_trick, output_dir / "ppo_eos_trick.json")

    ppo_config_kwargs = dict(
        output_dir=str(output_dir),
        seed=int(train_cfg.get("seed", 839)),
        data_seed=int(train_cfg.get("data_seed", train_cfg.get("seed", 839))),
        per_device_train_batch_size=int(
            train_cfg.get("per_device_train_batch_size", 2)
        ),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 32)
        ),
        learning_rate=float(train_cfg.get("learning_rate", 3e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "linear")),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy=str(train_cfg.get("save_strategy", "steps")),
        save_steps=int(train_cfg.get("save_steps", 25)),
        save_total_limit=int(train_cfg.get("save_total_limit", 8)),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        tf32=bool(train_cfg.get("tf32", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        report_to=trainer_report_to(train_cfg.get("report_to")),
        run_name=train_cfg.get("run_name"),
        optim=str(train_cfg.get("optim", "adamw_torch_fused")),
        total_episodes=total_episodes,
        num_ppo_epochs=int(ppo_cfg.get("num_ppo_epochs", 4)),
        num_mini_batches=int(ppo_cfg.get("num_mini_batches", 1)),
        local_rollout_forward_batch_size=int(
            ppo_cfg.get("local_rollout_forward_batch_size", 2)
        ),
        response_length=int(ppo_cfg.get("response_length", 768)),
        stop_token=str(ppo_cfg.get("stop_token", "eos")),
        temperature=float(ppo_cfg.get("temperature", 0.7)),
        missing_eos_penalty=0.0,
        whiten_rewards=False,
        kl_coef=float(ppo_cfg.get("kl_coef", 0.10)),
        kl_estimator=str(ppo_cfg.get("kl_estimator", "k1")),
        cliprange=float(ppo_cfg.get("cliprange", 0.2)),
        cliprange_value=float(ppo_cfg.get("cliprange_value", 0.2)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.1)),
        gamma=float(ppo_cfg.get("gamma", 1.0)),
        lam=float(ppo_cfg.get("lam", 0.95)),
        num_sample_generations=int(ppo_cfg.get("num_sample_generations", 0)),
        sft_model_path=str(cfg["model"]["policy_model_path"]),
        reward_model_path=reward_path,
    )
    if "adam_epsilon" in inspect.signature(PPOConfig).parameters:
        ppo_config_kwargs["adam_epsilon"] = float(train_cfg.get("adam_epsilon", 1e-5))
    args = PPOConfig(**ppo_config_kwargs)
    if int(args.batch_size or batch_size) != batch_size:
        raise RuntimeError(
            "This exact balanced-resume implementation currently requires one "
            f"Colab process; expected batch {batch_size}, TRL resolved {args.batch_size}."
        )

    resume_metadata = {
        "schema_version": 1,
        "resume_fingerprint": _resume_fingerprint(cfg),
        "rollout_plan_sha256": plan_report["plan_sha256"],
        "reward_guardrails_sha256": _json_sha256(guardrails),
        "planned_updates": planned_updates,
        "planned_episodes": planned_updates * batch_size,
        "configured_total_episodes": total_episodes,
        "rollout_batch_size": batch_size,
    }
    if resume_checkpoint:
        _validate_resume_checkpoint(
            resume_checkpoint,
            cfg=cfg,
            plan_report=plan_report,
            guardrails=guardrails,
        )

    segment_callback, exact_callback = _make_segment_callbacks(
        start_state=start_state,
        planned_updates=planned_updates,
        target_update=target_update,
        train_dataset_len=len(full_train_dataset),
        resume_metadata=resume_metadata,
        guardrails=guardrails,
    )
    callbacks = [
        segment_callback,
        exact_callback,
        *build_callbacks(train_cfg),
    ]
    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=reference,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=segment_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(lora_cfg),
        callbacks=callbacks,
    )
    exact_callback.trainer = trainer
    if int(args.world_size) != 1:
        raise RuntimeError(
            "Exact domain-balanced segmented PPO currently supports one GPU process. "
            f"TRL detected world_size={args.world_size}."
        )

    _replace_with_segment_dataloader(trainer, segment_dataset)
    trainer.train_dataset_len = len(full_train_dataset)
    full_schedule_updates = int(args.num_total_batches)
    if full_schedule_updates != planned_updates:
        raise RuntimeError(
            f"TRL planned {full_schedule_updates} updates; expected {planned_updates}."
        )
    args.num_total_batches = segment_updates
    if resume_checkpoint:
        start_state = _restore_exact_trainer_state(trainer, resume_checkpoint)
    _install_diagnostics_logging(trainer, collector, output_dir)

    segment_report = {
        "schema_version": 1,
        "start_update": start_update,
        "target_update": target_update,
        "segment_updates": segment_updates,
        "planned_updates": planned_updates,
        "start_episode": start_update * batch_size,
        "target_episode": target_update * batch_size,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "rollout_plan_sha256": plan_report["plan_sha256"],
        "reward_guardrails_sha256": _json_sha256(guardrails),
    }
    write_json(segment_report, output_dir / "ppo_segment_plan.json")
    print(json.dumps(segment_report, indent=2))
    trainer.train()

    checkpoint = output_dir / f"checkpoint-{target_update}"
    health = _checkpoint_health(checkpoint)
    write_json(health, output_dir / f"checkpoint-{target_update}-health.json")
    if not health["healthy"]:
        raise RuntimeError(
            f"PPO segment reached update {target_update}, but its exact-resume "
            "checkpoint failed the health check."
        )

    adapter_dir = output_dir / "final_policy_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    trained_policy = unwrapped.policy
    trained_value = unwrapped.value_model
    value_offset = remove_reward_center(trained_value)
    value_dir = output_dir / "final_value_model"
    trained_value.save_pretrained(value_dir, safe_serialization=True)
    tokenizer.save_pretrained(value_dir)
    write_json(
        {"reward_offset": value_offset, "apply_on_load": True},
        value_dir / "reward_center.json",
    )

    merged_policy_dir = output_dir / "final_merged_policy"
    merge_peft_model(trained_policy, merged_policy_dir, tokenizer)
    write_json(
        {
            "schema_version": 1,
            "selected_checkpoint": str(checkpoint),
            "selected_update": target_update,
            "selected_episodes": target_update * batch_size,
            "selection_is_external": False,
            "purpose": "latest completed segment snapshot",
        },
        merged_policy_dir / "ppo_checkpoint_selection.json",
    )

    log_history = list(trainer.state.log_history)
    write_json(log_history, output_dir / "trainer_log_history.json")
    summary = {
        "backend": "trl",
        "stage": "ppo",
        "train_prompts": len(full_train_dataset),
        "eval_prompts": len(eval_dataset),
        "configured_total_episodes": total_episodes,
        "planned_episodes": planned_updates * batch_size,
        "planned_updates": planned_updates,
        "completed_update": target_update,
        "completed_episodes": target_update * batch_size,
        "rollout_batch_size": batch_size,
        "policy_adapter_dir": str(adapter_dir),
        "merged_policy_dir": str(merged_policy_dir),
        "value_model_dir": str(value_dir),
        "reference_model_path": reference_path,
        "reward_model_path": reward_path,
        "reward_offset": reward_offset,
        "reward_guardrails": guardrails,
        "eos_trick": eos_trick,
        "rollout_plan": plan_report,
        "sampling_distribution": sampling_distribution,
        "checkpoint_health": health,
        "last_metrics": log_history[-1] if log_history else {},
    }
    write_json(summary, output_dir / "run_summary.json")
    status = "completed" if target_update == planned_updates else "segment_completed"
    finalize_experiment(output_dir, status=status, summary=summary)
    maybe_sync_tree(output_dir, train_cfg.get("final_sync_dir"))

    del trainer, reward_model, value_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir
