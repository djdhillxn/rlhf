import random
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import load_config, save_config
from .data import build_prompt_records, load_helpsteer3_preference, save_jsonl
from .experiment import finalize_experiment, initialize_experiment
from .lm_policy import FrozenCausalLM, TokenPolicyWithValue
from .metrics import (
    append_jsonl,
    collect_run_metadata,
    jsonl_to_csv,
    read_jsonl,
    save_metric_plots,
    write_json,
)
from .ppo_lm import AdaptiveKLController, LMPPOTrainer
from .reward_model import RewardModel
from .rollout import GenerationConfig, collect_lm_rollouts


def _device_from_cfg(cfg):
    name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _batched_cycle(records, batch_size, seed):
    rng = random.Random(seed)
    while True:
        shuffled = list(records)
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            if batch:
                yield batch


def _cuda_memory():
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 4),
        "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 4),
        "cuda_max_memory_allocated_gb": round(
            torch.cuda.max_memory_allocated() / (1024**3), 4
        ),
    }


def run_ppo_training(config_path, *, output_dir=None):
    cfg = load_config(config_path)
    output_dir = Path(
        output_dir
        or cfg.train.get("output_dir", "outputs/rlhf/qwen25_05b_helpsteer3_ppo")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    checkpoint_subdir = str(cfg.train.get("checkpoint_subdir", "checkpoints"))
    checkpoint_root = output_dir / checkpoint_subdir
    checkpoint_root.mkdir(exist_ok=True)
    checkpoint_manifest = []
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="rlhf_ppo",
        config_path=config_path,
        extra={"model_name": str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))},
    )

    from transformers import AutoTokenizer

    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=bool(cfg.model.get("trust_remote_code", False))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    seed = int(cfg.train.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device_from_cfg(cfg.train)

    raw_prompts = load_helpsteer3_preference(str(cfg.data.get("prompt_split", "train")))
    prompt_records = build_prompt_records(
        raw_prompts,
        tokenizer,
        max_samples=cfg.data.get("max_prompt_samples"),
        seed=seed,
        shuffle=True,
    )
    if not prompt_records:
        raise RuntimeError("No prompts were loaded from HelpSteer3.")
    save_jsonl(
        prompt_records[: min(len(prompt_records), 1000)],
        output_dir / "prompt_preview.jsonl",
    )
    write_json(
        collect_run_metadata(
            run_type="rlhf_ppo",
            config_path=config_path,
            extra={"model_name": model_name, "num_prompt_records": len(prompt_records)},
        ),
        output_dir / "run_metadata.json",
    )

    policy_init_checkpoint = cfg.model.get("policy_init_checkpoint_dir")
    if policy_init_checkpoint:
        policy = TokenPolicyWithValue.load_rlhf_pretrained(
            str(policy_init_checkpoint),
            base_model_name=model_name,
            torch_dtype=str(cfg.model.get("torch_dtype", "auto")),
            device_map=cfg.model.get("policy_device_map"),
            load_in_4bit=bool(
                cfg.model.get(
                    "policy_load_in_4bit", cfg.model.get("load_in_4bit", False)
                )
            ),
            load_in_8bit=bool(cfg.model.get("policy_load_in_8bit", False)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    else:
        policy = TokenPolicyWithValue.from_model_name(
            model_name,
            torch_dtype=str(cfg.model.get("torch_dtype", "auto")),
            device_map=cfg.model.get("policy_device_map"),
            load_in_4bit=bool(
                cfg.model.get(
                    "policy_load_in_4bit", cfg.model.get("load_in_4bit", False)
                )
            ),
            load_in_8bit=bool(cfg.model.get("policy_load_in_8bit", False)),
            lora=dict(cfg.get("lora", {})),
            gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", True)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    if cfg.model.get("policy_device_map") is None:
        policy.to(device)

    ref_checkpoint = cfg.model.get("ref_checkpoint_dir") or policy_init_checkpoint
    if ref_checkpoint:
        reference = FrozenCausalLM.load_rlhf_pretrained(
            str(ref_checkpoint),
            base_model_name=model_name,
            torch_dtype=str(
                cfg.model.get("ref_torch_dtype", cfg.model.get("torch_dtype", "auto"))
            ),
            device_map=cfg.model.get("ref_device_map"),
            load_in_4bit=bool(cfg.model.get("ref_load_in_4bit", True)),
            load_in_8bit=bool(cfg.model.get("ref_load_in_8bit", False)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    else:
        reference = FrozenCausalLM.from_model_name(
            model_name,
            torch_dtype=str(
                cfg.model.get("ref_torch_dtype", cfg.model.get("torch_dtype", "auto"))
            ),
            device_map=cfg.model.get("ref_device_map"),
            load_in_4bit=bool(cfg.model.get("ref_load_in_4bit", True)),
            load_in_8bit=bool(cfg.model.get("ref_load_in_8bit", False)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    if cfg.model.get("ref_device_map") is None:
        reference.to(device)

    reward_checkpoint = Path(str(cfg.reward_model.get("checkpoint_dir")))
    if not reward_checkpoint.exists():
        raise FileNotFoundError(
            f"Reward checkpoint not found: {reward_checkpoint}. Run scripts/rlhf_train_reward_model.py first."
        )
    reward_model = RewardModel.load_rlhf_pretrained(
        reward_checkpoint,
        base_model_name=model_name,
        torch_dtype=str(
            cfg.reward_model.get("torch_dtype", cfg.model.get("torch_dtype", "auto"))
        ),
        device_map=cfg.reward_model.get("device_map"),
        load_in_4bit=bool(cfg.reward_model.get("load_in_4bit", True)),
        load_in_8bit=bool(cfg.reward_model.get("load_in_8bit", False)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)
    if cfg.reward_model.get("device_map") is None:
        reward_model.to(device)

    generation = GenerationConfig(**dict(cfg.get("generation", {})))
    ppo_trainer = LMPPOTrainer(policy, dict(cfg.get("ppo", {})))
    kl_ctl = AdaptiveKLController(
        init_kl_coef=float(cfg.kl.get("init_kl_coef", 0.05)),
        target_kl=float(cfg.kl.get("target_ref_kl", 0.05)),
        horizon=int(cfg.kl.get("horizon", 10000)),
        min_kl_coef=float(cfg.kl.get("min_kl_coef", 0.02)),
        max_kl_coef=float(cfg.kl.get("max_kl_coef", 1.0)),
        adaptive=bool(cfg.kl.get("adaptive", True)),
    )

    batch_size = int(cfg.train.get("rollout_batch_size", 8))
    total_updates = int(cfg.train.get("total_updates", 1000))
    # Backward compatible: save_every was the original name. checkpoint_every is clearer.
    save_every = int(
        cfg.train.get("checkpoint_every", cfg.train.get("save_every", 100))
    )
    sample_every = int(cfg.train.get("sample_every", 25))
    prompt_iter = _batched_cycle(prompt_records, batch_size, seed=seed)
    start_time = time.time()
    stopped_by_safety_guard = False

    progress = tqdm(range(1, total_updates + 1), desc="PPO updates")
    for update_idx in progress:
        records = next(prompt_iter)
        group_size = int(cfg.train.get("num_generations_per_prompt", 1))
        if group_size > 1:
            prompts = []
            expanded_records = []
            for r in records:
                for sample_idx in range(group_size):
                    prompts.append(r["prompt"])
                    rr = dict(r)
                    rr["group_sample_idx"] = sample_idx
                    rr["group_size"] = group_size
                    expanded_records.append(rr)
        else:
            prompts = [r["prompt"] for r in records]
            expanded_records = records
        shaping_cfg = dict(cfg.get("reward_shaping", {}))
        rollout = collect_lm_rollouts(
            policy,
            reference,
            reward_model,
            tokenizer,
            prompts,
            generation=generation,
            kl_coef=kl_ctl.value,
            device=device,
            metadata=expanded_records,
            reward_clip_min=shaping_cfg.get("reward_clip_min"),
            reward_clip_max=shaping_cfg.get("reward_clip_max"),
            length_penalty_coef=float(shaping_cfg.get("length_penalty_coef", 0.0)),
            missing_eos_penalty=float(shaping_cfg.get("missing_eos_penalty", 0.0)),
            min_response_tokens=int(shaping_cfg.get("min_response_tokens", 0)),
            short_response_penalty=float(
                shaping_cfg.get("short_response_penalty", 0.0)
            ),
            group_size=group_size,
            group_normalize=bool(shaping_cfg.get("group_normalize", False)),
            group_advantage_eps=float(shaping_cfg.get("group_advantage_eps", 1e-6)),
        )
        rollout = ppo_trainer.prepare_batch(rollout)
        stats = ppo_trainer.update(rollout, kl_coef=kl_ctl.value)
        kl_ctl.update(stats.objective_kl, stats.num_response_tokens)

        response_lengths = rollout.response_mask.sum(dim=1).float()
        responses = rollout.responses or []
        empty_response_rate = float(
            sum(1 for x in responses if not str(x).strip()) / max(len(responses), 1)
        )
        record = stats.__dict__.copy()
        record.update(
            {
                "update": update_idx,
                "elapsed_sec": time.time() - start_time,
                "mean_response_tokens": float(response_lengths.mean().item()),
                "min_response_tokens": float(response_lengths.min().item()),
                "empty_response_rate": empty_response_rate,
                "mean_response_chars": float(
                    sum(len(x) for x in responses) / max(len(responses), 1)
                ),
            }
        )
        record.update(_cuda_memory())
        append_jsonl(record, output_dir / "ppo_metrics.jsonl")
        if (
            int(cfg.train.get("artifact_every", 25)) > 0
            and update_idx % int(cfg.train.get("artifact_every", 25)) == 0
        ):
            jsonl_to_csv(
                output_dir / "ppo_metrics.jsonl", output_dir / "ppo_metrics.csv"
            )
        progress.set_postfix(
            reward=f"{record['reward_model_score']:.3f}",
            kl=f"{record['objective_kl']:.4f}",
            tok=f"{record['mean_response_tokens']:.1f}",
            loss=f"{record['loss']:.3f}",
        )

        safety_cfg = dict(cfg.get("safety", {}))
        safety_start = int(safety_cfg.get("start_after_updates", 10))
        stop_reason = None
        if update_idx >= safety_start:
            max_ref_kl = safety_cfg.get("max_abs_ref_logratio")
            min_mean_tokens = safety_cfg.get("min_mean_response_tokens")
            max_empty_rate = safety_cfg.get("max_empty_response_rate")
            if max_ref_kl is not None and float(record["abs_ref_logratio"]) > float(
                max_ref_kl
            ):
                stop_reason = f"abs_ref_logratio {record['abs_ref_logratio']:.4f} exceeded {float(max_ref_kl):.4f}"
            if min_mean_tokens is not None and float(
                record["mean_response_tokens"]
            ) < float(min_mean_tokens):
                stop_reason = f"mean_response_tokens {record['mean_response_tokens']:.4f} fell below {float(min_mean_tokens):.4f}"
            if max_empty_rate is not None and float(
                record["empty_response_rate"]
            ) > float(max_empty_rate):
                stop_reason = f"empty_response_rate {record['empty_response_rate']:.4f} exceeded {float(max_empty_rate):.4f}"
        if stop_reason is not None:
            stopped_by_safety_guard = True
            write_json(
                {
                    "status": "stopped_by_safety_guard",
                    "stop_update": update_idx,
                    "stop_reason": stop_reason,
                    "final_kl_coef": kl_ctl.value,
                },
                output_dir / "run_status.json",
            )
            print(f"Stopping PPO early: {stop_reason}")
            break

        if update_idx == 1 or (sample_every > 0 and update_idx % sample_every == 0):
            sample_rows = []
            for prompt, response, meta, score in zip(
                rollout.prompts or [],
                rollout.responses or [],
                rollout.metadata or [],
                rollout.scores,
            ):
                sample_rows.append(
                    {
                        "update": update_idx,
                        "domain": meta.get("domain", "unknown")
                        if isinstance(meta, dict)
                        else "unknown",
                        "language": meta.get("language", "unknown")
                        if isinstance(meta, dict)
                        else "unknown",
                        "prompt": prompt,
                        "response": response,
                        "reward_score": float(score.item()),
                    }
                )
            save_jsonl(
                sample_rows, output_dir / f"samples/update_{update_idx:05d}.jsonl"
            )

        if save_every > 0 and update_idx % save_every == 0:
            ckpt_path = checkpoint_root / f"update_{update_idx:05d}"
            policy.save_rlhf_pretrained(ckpt_path, tokenizer=tokenizer)
            checkpoint_manifest.append(
                {
                    "update": update_idx,
                    "path": str(ckpt_path),
                    "reward_model_score": float(record.get("reward_model_score", 0.0)),
                    "total_reward": float(record.get("total_reward", 0.0)),
                    "abs_ref_logratio": float(record.get("abs_ref_logratio", 0.0)),
                    "clip_fraction": float(record.get("clip_fraction", 0.0)),
                    "mean_response_tokens": float(
                        record.get("mean_response_tokens", 0.0)
                    ),
                    "empty_response_rate": float(
                        record.get("empty_response_rate", 0.0)
                    ),
                }
            )
            write_json(
                {"checkpoints": checkpoint_manifest}, checkpoint_root / "manifest.json"
            )
            jsonl_to_csv(
                output_dir / "ppo_metrics.jsonl", output_dir / "ppo_metrics.csv"
            )
            print(f"Saved PPO checkpoint: {ckpt_path}")

    final_checkpoint = output_dir / "checkpoint_final"
    policy.save_rlhf_pretrained(final_checkpoint, tokenizer=tokenizer)
    checkpoint_manifest.append({"update": "final", "path": str(final_checkpoint)})
    write_json({"checkpoints": checkpoint_manifest}, checkpoint_root / "manifest.json")
    jsonl_to_csv(output_dir / "ppo_metrics.jsonl", output_dir / "ppo_metrics.csv")
    rows = read_jsonl(output_dir / "ppo_metrics.jsonl")
    plot_paths = save_metric_plots(
        rows,
        output_dir / "plots",
        x_key="update",
        y_keys=[
            "reward_model_score",
            "total_reward",
            "objective_kl",
            "kl_coef",
            "approx_kl",
            "abs_ref_logratio",
            "clip_fraction",
            "loss",
            "policy_loss",
            "value_loss",
            "mean_response_tokens",
            "empty_response_rate",
        ],
        prefix="ppo",
    )
    summary = {
        "total_updates_requested": total_updates,
        "total_updates_completed": len(rows),
        "final_kl_coef": kl_ctl.value,
        "checkpoint_manifest": str(checkpoint_root / "manifest.json"),
        "plot_paths": plot_paths,
    }
    write_json(summary, output_dir / "run_summary.json")
    status = "stopped_by_safety_guard" if stopped_by_safety_guard else "completed"
    finalize_experiment(output_dir, status=status, summary=summary)
    return output_dir
