import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from .config import load_config, save_config
from .data import build_prompt_records, load_helpsteer3_preference, save_jsonl
from .experiment import finalize_experiment, initialize_experiment
from .lm_policy import TokenPolicyWithValue
from .metrics import write_csv, write_json
from .reward_model import RewardModel
from .rollout import GenerationConfig
from .trl_common import resolve_dtype
from .trl_models import load_reward_center


class HFSequenceRewardAdapter(nn.Module):
    """Expose a Transformers sequence classifier through the legacy scalar API."""

    def __init__(self, model, offset=0.0):
        super().__init__()
        self.model = model
        self.offset = float(offset)

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        return output.logits.squeeze(-1) - self.offset


def _device_from_cfg(cfg):
    name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _resolve_num_prompts(value):
    """Resolve eval.num_prompts; accepts integers or all/full/null/-1 for complete split."""
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"all", "full", "none", "null", "validation", "-1"}:
            return None
        value = lowered
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"eval.num_prompts must be an integer or 'all', got {value!r}"
        ) from exc
    return None if n <= 0 else n


def _safe_label(label):
    label = str(label).strip().lower()
    label = re.sub(r"[^a-z0-9_]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    if not label:
        raise ValueError("Policy label cannot be empty after sanitization.")
    return label


def _none_if_empty(value):
    if value in {None, "", "none", "None", "null", "NULL"}:
        return None
    return str(value)


def _looks_like_local_path(value):
    expanded = Path(value).expanduser()
    return (
        expanded.is_absolute()
        or value.startswith(("./", "../", "~"))
        or len(expanded.parts) > 2
    )


def _looks_like_policy_checkpoint(path):
    """Return True only for actual saved policy checkpoints.

    This prevents the dangerous failure mode where a typo such as
    `checkpoints/checkpoint_00250` silently loads the base model because no
    adapter exists at that location.
    """
    return (
        path.exists()
        and (path / "adapter_or_model").exists()
        and (path / "value_head.pt").exists()
    )


def _checkpoint_candidates_from_path(path):
    candidates = [path]

    # If the user accidentally points to an output dir, try common children.
    candidates.extend(
        [
            path / "checkpoint_final",
            path / "checkpoints" / "update_00250",
            path / "checkpoint_00250",
        ]
    )

    # If the user writes checkpoints/checkpoint_00250 but the run saved
    # checkpoints/update_00250, repair that common naming mismatch.
    digits = re.sub(r"\D", "", path.name)
    if digits:
        update_name = f"update_{int(digits):05d}"
        old_name = f"checkpoint_{int(digits):05d}"
        candidates.extend(
            [
                path.parent / update_name,
                path.parent / old_name,
                path.parent.parent / "checkpoints" / update_name,
                path.parent.parent / old_name,
            ]
        )

    # If a manifest exists, add its saved paths.
    possible_manifests = [
        path / "checkpoints" / "manifest.json",
        path.parent / "manifest.json",
        path.parent.parent / "manifest.json",
    ]
    for manifest in possible_manifests:
        if not manifest.exists():
            continue
        try:
            import json

            data = json.loads(manifest.read_text(encoding="utf-8"))
            for item in data.get("checkpoints", []):
                ckpt_path = item.get("path")
                if ckpt_path:
                    candidates.append(Path(ckpt_path))
        except Exception:
            pass

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _resolve_checkpoint_dir(spec):
    """Resolve a policy checkpoint specification.

    Supported forms:
      - checkpoint_dir: null                 -> base model
      - checkpoint_dir: path/to/checkpoint   -> exact checkpoint
      - output_dir + checkpoint: update_00250/checkpoint_00250/final

    The resolver accepts both newer checkpoints/update_XXXXX and older
    checkpoint_XXXXX layouts.  It now validates that a candidate really contains
    an adapter/model and value head; otherwise evaluation fails loudly instead of
    silently falling back to the base model.
    """
    checkpoint_format = str(spec.get("format", "legacy")).lower()
    checkpoint_dir = _none_if_empty(
        spec.get(
            "model_path", spec.get("checkpoint_dir", spec.get("policy_checkpoint_dir"))
        )
    )
    if checkpoint_format in {"hf", "huggingface", "trl", "peft"}:
        if checkpoint_dir is None:
            return None
        path = Path(checkpoint_dir).expanduser()
        if path.exists():
            return str(path)
        if not _looks_like_local_path(checkpoint_dir):
            return checkpoint_dir
        raise FileNotFoundError(
            f"Local Hugging Face checkpoint does not exist for "
            f"label={spec.get('label')!r}: {checkpoint_dir}. "
            "Check the resolved --set policies[index].checkpoint_dir override."
        )
    if checkpoint_dir:
        path = Path(checkpoint_dir)
        for candidate in _checkpoint_candidates_from_path(path):
            if _looks_like_policy_checkpoint(candidate):
                return str(candidate)
        tried = "\n  - ".join(
            str(c) for c in _checkpoint_candidates_from_path(path)[:12]
        )
        raise FileNotFoundError(
            f"Could not resolve policy checkpoint for label={spec.get('label')!r}. "
            f"Requested: {path}\nTried:\n  - {tried}"
        )

    output_dir = _none_if_empty(spec.get("output_dir"))
    checkpoint = _none_if_empty(spec.get("checkpoint"))
    if not output_dir:
        return None
    root = Path(output_dir)
    if checkpoint is None or checkpoint in {"final", "checkpoint_final"}:
        candidates = [root / "checkpoint_final"]
    else:
        ckpt = str(checkpoint)
        update_digits = re.sub(r"\D", "", ckpt)
        update_name = f"update_{int(update_digits):05d}" if update_digits else ckpt
        old_name = f"checkpoint_{int(update_digits):05d}" if update_digits else ckpt
        candidates = [
            root / ckpt,
            root / "checkpoints" / ckpt,
            root / "checkpoints" / update_name,
            root / old_name,
        ]
    for candidate in candidates:
        if _looks_like_policy_checkpoint(candidate):
            return str(candidate)
    tried = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not resolve policy checkpoint for label={spec.get('label')!r}. "
        f"output_dir={output_dir!r}, checkpoint={checkpoint!r}\nTried:\n  - {tried}"
    )


def _policy_specs(cfg):
    specs = [dict(x) for x in cfg.get("policies", [])]
    if not specs:
        raise ValueError(
            "Policy-suite eval requires a top-level `policies:` list in the config."
        )
    labels = [
        _safe_label(str(s.get("label", f"policy_{i}"))) for i, s in enumerate(specs)
    ]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate policy labels after sanitization: {labels}")
    out = []
    for label, spec in zip(labels, specs):
        spec["label"] = label
        spec["checkpoint_dir"] = _resolve_checkpoint_dir(spec)
        out.append(spec)
    print("Resolved evaluation policies:")
    for spec in out:
        source = spec.get("checkpoint_dir") or str(
            cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct")
        )
        print(f"  {spec['label']}: {source}")
    return out


def _eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(x) for x in eos if x is not None}
    return {int(eos)}


def _response_lengths_and_eos(response_ids, tokenizer):
    eos_ids = _eos_token_ids(tokenizer)
    lengths = []
    hit = []
    for row in response_ids.detach().cpu().tolist():
        keep = len(row)
        hit_eos = False
        if eos_ids:
            for idx, token_id in enumerate(row):
                if int(token_id) in eos_ids:
                    keep = idx + 1
                    hit_eos = True
                    break
        lengths.append(max(0, keep))
        hit.append(hit_eos)
    return (
        torch.tensor(lengths, device=response_ids.device, dtype=torch.long),
        torch.tensor(hit, device=response_ids.device, dtype=torch.bool),
    )


def _build_full_attention(prompt_attention, generated, prompt_width, response_lengths):
    full_attention = torch.zeros_like(generated, dtype=torch.long)
    full_attention[:, :prompt_width] = prompt_attention.long()
    if generated.size(1) > prompt_width:
        pos = torch.arange(
            generated.size(1) - prompt_width, device=generated.device
        ).unsqueeze(0)
        full_attention[:, prompt_width:] = (pos < response_lengths.unsqueeze(1)).long()
    return full_attention


def _load_policy(cfg, spec, device, tokenizer=None):
    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    checkpoint_dir = spec.get("checkpoint_dir")
    checkpoint_format = str(spec.get("format", "legacy")).lower()
    torch_dtype = str(spec.get("torch_dtype", cfg.model.get("torch_dtype", "auto")))
    device_map = spec.get("device_map", cfg.model.get("policy_device_map"))
    load_in_4bit = bool(
        spec.get(
            "load_in_4bit",
            cfg.model.get("policy_load_in_4bit", cfg.model.get("load_in_4bit", False)),
        )
    )
    load_in_8bit = bool(
        spec.get("load_in_8bit", cfg.model.get("policy_load_in_8bit", False))
    )
    trust_remote_code = bool(cfg.model.get("trust_remote_code", False))
    local_files_only = bool(cfg.model.get("local_files_only", False))

    if checkpoint_format in {"hf", "huggingface", "trl", "peft"}:
        from transformers import AutoModelForCausalLM

        model_path = checkpoint_dir or spec.get("model_path") or model_name
        kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        dtype = resolve_dtype(torch_dtype)
        if dtype != "auto":
            kwargs["dtype"] = dtype
        if spec.get("attn_implementation"):
            kwargs["attn_implementation"] = spec["attn_implementation"]
        adapter_path = _none_if_empty(spec.get("adapter_path"))
        if adapter_path:
            base_path = str(spec.get("base_model_path", model_path))
            policy = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
            from peft import PeftModel

            policy = PeftModel.from_pretrained(policy, adapter_path, is_trainable=False)
        else:
            policy = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
        if tokenizer is not None:
            from .trl_common import resize_embeddings_if_needed

            resize_embeddings_if_needed(policy, tokenizer)
            policy.config.pad_token_id = tokenizer.pad_token_id
            policy.config.eos_token_id = tokenizer.eos_token_id
        if device_map is None:
            policy.to(device)
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        return policy

    if checkpoint_dir:
        policy = TokenPolicyWithValue.load_rlhf_pretrained(
            checkpoint_dir,
            base_model_name=model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
        )
    else:
        policy = TokenPolicyWithValue.from_model_name(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            lora=None,
            gradient_checkpointing=False,
            trust_remote_code=trust_remote_code,
        )
    if device_map is None:
        policy.to(device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _load_reward_model(cfg, device):
    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    checkpoint_format = str(cfg.reward_model.get("format", "legacy")).lower()
    if checkpoint_format in {"hf", "huggingface", "trl"}:
        from transformers import AutoModelForSequenceClassification

        checkpoint_dir = str(cfg.reward_model.get("checkpoint_dir"))
        kwargs = {
            "num_labels": 1,
            "trust_remote_code": bool(cfg.model.get("trust_remote_code", False)),
            "local_files_only": bool(cfg.model.get("local_files_only", False)),
        }
        dtype = resolve_dtype(
            str(
                cfg.reward_model.get(
                    "torch_dtype", cfg.model.get("torch_dtype", "auto")
                )
            )
        )
        if dtype != "auto":
            kwargs["dtype"] = dtype
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir, **kwargs
        )
        model.config.pad_token_id = cfg.reward_model.get(
            "pad_token_id", model.config.pad_token_id
        )
        if cfg.reward_model.get("device_map") is None:
            model.to(device)
        offset = load_reward_center(cfg.reward_model.get("reward_center_path"))
        reward_model = HFSequenceRewardAdapter(model, offset=offset)
        reward_model.eval()
        for parameter in reward_model.parameters():
            parameter.requires_grad_(False)
        return reward_model

    reward_model = RewardModel.load_rlhf_pretrained(
        str(cfg.reward_model.get("checkpoint_dir")),
        base_model_name=model_name,
        torch_dtype=str(
            cfg.reward_model.get("torch_dtype", cfg.model.get("torch_dtype", "auto"))
        ),
        device_map=cfg.reward_model.get("device_map"),
        load_in_4bit=bool(cfg.reward_model.get("load_in_4bit", False)),
        load_in_8bit=bool(cfg.reward_model.get("load_in_8bit", False)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)
    if cfg.reward_model.get("device_map") is None:
        reward_model.to(device)
    return reward_model


def _gen_kwargs_from_config(generation, tokenizer, input_ids, attention_mask):
    pad_id = int(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(generation.max_new_tokens),
        do_sample=bool(generation.do_sample),
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=float(generation.repetition_penalty),
        no_repeat_ngram_size=int(generation.no_repeat_ngram_size),
    )
    if int(getattr(generation, "min_new_tokens", 0)) > 0:
        gen_kwargs["min_new_tokens"] = int(generation.min_new_tokens)
    if bool(generation.do_sample):
        gen_kwargs["temperature"] = float(generation.temperature)
        gen_kwargs["top_p"] = float(generation.top_p)
        gen_kwargs["top_k"] = 0
        gen_kwargs["typical_p"] = 1.0
        gen_kwargs["epsilon_cutoff"] = 0.0
        gen_kwargs["eta_cutoff"] = 0.0
    return gen_kwargs


@torch.inference_mode()
def _generate_and_score(
    policy,
    reward_model,
    tokenizer,
    prompts,
    *,
    generation,
    device,
):
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(generation.max_prompt_length),
    )
    tokenizer.padding_side = old_padding_side

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_width = int(input_ids.size(1))
    prompt_tokens = attention_mask.sum(dim=1).detach().cpu().tolist()

    generated = policy.generate(
        **_gen_kwargs_from_config(generation, tokenizer, input_ids, attention_mask)
    )
    response_ids = generated[:, prompt_width:]
    response_lengths, hit_eos = _response_lengths_and_eos(response_ids, tokenizer)
    full_attention = _build_full_attention(
        attention_mask, generated, prompt_width, response_lengths
    )
    scores = reward_model(generated, full_attention).detach().float().cpu().tolist()

    rows = []
    for i, (ids, keep) in enumerate(
        zip(response_ids, response_lengths.detach().cpu().tolist())
    ):
        keep = int(keep)
        text = tokenizer.decode(ids[:keep], skip_special_tokens=True).strip()
        rows.append(
            {
                "response": text,
                "reward": float(scores[i]),
                "response_tokens": keep,
                "response_chars": len(text),
                "prompt_tokens": int(prompt_tokens[i]),
                "total_tokens": int(full_attention[i].sum().item()),
                "hit_eos": bool(hit_eos[i].item()),
                "cap_hit": bool(
                    keep >= int(generation.max_new_tokens)
                    and not bool(hit_eos[i].item())
                ),
                "empty": not bool(text.strip()),
            }
        )
    return rows


def _numeric(values):
    out = []
    for value in values:
        try:
            if value is not None and not (
                isinstance(value, float) and math.isnan(value)
            ):
                out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def _stats(values):
    vals = _numeric(values)
    if not vals:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    median = (
        vals_sorted[n // 2]
        if n % 2
        else 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
    )
    return {
        "mean": float(sum(vals_sorted) / n),
        "median": float(median),
        "min": float(vals_sorted[0]),
        "max": float(vals_sorted[-1]),
    }


def _winner_for_rewards(rewards, tie_epsilon):
    max_reward = max(rewards.values())
    winners = [
        label
        for label, reward in rewards.items()
        if abs(float(reward) - max_reward) <= tie_epsilon
    ]
    return winners[0] if len(winners) == 1 else "tie"


def _add_comparisons(rows, labels, tie_epsilon):
    for row in rows:
        rewards = {label: float(row[f"{label}_reward"]) for label in labels}
        row["winner"] = _winner_for_rewards(rewards, tie_epsilon)
        ranked = sorted(labels, key=lambda label: rewards[label], reverse=True)
        row["reward_rank"] = ">".join(ranked)
        row["reward_spread"] = max(rewards.values()) - min(rewards.values())
        for a, b in itertools.combinations(labels, 2):
            delta = rewards[b] - rewards[a]
            if abs(delta) <= tie_epsilon:
                winner = "tie"
            else:
                winner = b if delta > 0 else a
            row[f"delta_{b}_minus_{a}"] = delta
            row[f"winner_{a}_vs_{b}"] = winner


def _summarize(rows, labels, tie_epsilon):
    if not rows:
        return {"num_examples": 0, "labels": labels}

    winner_counts = {label: 0 for label in labels}
    winner_counts["tie"] = 0
    domain_winner_counts = {}
    for row in rows:
        winner = str(row.get("winner", "tie"))
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        domain = str(row.get("domain", "unknown"))
        domain_winner_counts.setdefault(domain, {label: 0 for label in labels})
        domain_winner_counts[domain].setdefault("tie", 0)
        domain_winner_counts[domain][winner] = (
            domain_winner_counts[domain].get(winner, 0) + 1
        )

    per_policy = {}
    for label in labels:
        per_policy[label] = {
            "reward": _stats([r.get(f"{label}_reward") for r in rows]),
            "response_tokens": _stats(
                [r.get(f"{label}_response_tokens") for r in rows]
            ),
            "response_chars": _stats([r.get(f"{label}_response_chars") for r in rows]),
            "cap_hit_rate": float(
                sum(1 for r in rows if r.get(f"{label}_cap_hit")) / len(rows)
            ),
            "empty_rate": float(
                sum(1 for r in rows if r.get(f"{label}_empty")) / len(rows)
            ),
            "overall_win_rate": float(winner_counts.get(label, 0) / len(rows)),
        }

    pairwise = {}
    pairwise_rows = []
    for a, b in itertools.combinations(labels, 2):
        key = f"{a}_vs_{b}"
        counts = {a: 0, b: 0, "tie": 0}
        domain_counts = {}
        deltas = []
        for row in rows:
            winner = str(row.get(f"winner_{a}_vs_{b}", "tie"))
            counts[winner] = counts.get(winner, 0) + 1
            domain = str(row.get("domain", "unknown"))
            domain_counts.setdefault(domain, {a: 0, b: 0, "tie": 0})
            domain_counts[domain][winner] = domain_counts[domain].get(winner, 0) + 1
            deltas.append(row.get(f"delta_{b}_minus_{a}", 0.0))
        non_ties = max(counts.get(a, 0) + counts.get(b, 0), 1)
        pairwise[key] = {
            "a": a,
            "b": b,
            "winner_counts": counts,
            "domain_winner_counts": domain_counts,
            f"{b}_minus_{a}": _stats(deltas),
            f"{a}_win_rate": float(counts.get(a, 0) / len(rows)),
            f"{b}_win_rate": float(counts.get(b, 0) / len(rows)),
            f"{b}_win_rate_excluding_ties": float(counts.get(b, 0) / non_ties),
        }
        pairwise_rows.append(
            {
                "comparison": key,
                "a": a,
                "b": b,
                f"{a}_wins": counts.get(a, 0),
                f"{b}_wins": counts.get(b, 0),
                "ties": counts.get("tie", 0),
                f"{a}_win_rate": counts.get(a, 0) / len(rows),
                f"{b}_win_rate": counts.get(b, 0) / len(rows),
                f"mean_delta_{b}_minus_{a}": pairwise[key][f"{b}_minus_{a}"]["mean"],
                f"median_delta_{b}_minus_{a}": pairwise[key][f"{b}_minus_{a}"][
                    "median"
                ],
            }
        )

    return {
        "num_examples": len(rows),
        "labels": labels,
        "tie_epsilon": tie_epsilon,
        "winner_counts": winner_counts,
        "domain_winner_counts": domain_winner_counts,
        "per_policy": per_policy,
        "pairwise": pairwise,
        "pairwise_rows": pairwise_rows,
    }


def _write_excel_if_available(rows, path):
    try:
        import pandas as pd

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_excel(path, index=False)
    except Exception:
        return


def _compact(text, n=650):
    s = str(text or "").replace("\n", "<br>")
    return s if len(s) <= n else s[:n] + "..."


def _write_markdown(rows, labels, path, n):
    lines = [
        "# Policy-suite evaluation preview\n\n",
        "This Markdown preview is intentionally truncated for readability. ",
        "Use `policy_suite_samples.jsonl` or `policy_suite_samples.csv` for full responses.\n\n",
        f"Policies: **{', '.join(labels)}**\n\n",
    ]
    show_rows = rows[:n]
    for row in show_rows:
        lines.append(
            f"## idx {row.get('idx')} — domain: {row.get('domain')} — winner: {row.get('winner')}\n\n"
        )
        lines.append(f"**Prompt**\n\n{_compact(row.get('prompt'), 1000)}\n\n")
        for label in labels:
            lines.append(
                f"**{label} reward:** `{float(row.get(f'{label}_reward', 0.0)):.4f}`; "
            )
            lines.append(
                f"tokens: `{int(row.get(f'{label}_response_tokens', 0))}`; cap_hit: `{row.get(f'{label}_cap_hit')}`\n\n"
            )
            lines.append(f"{_compact(row.get(f'{label}_response'), 1200)}\n\n")
        lines.append("---\n\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_plots(rows, labels, output_dir):
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Reward distributions.
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    for label in labels:
        vals = _numeric([r.get(f"{label}_reward") for r in rows])
        if vals:
            ax.hist(vals, bins=50, alpha=0.35, label=label)
    ax.set_title("Reward distributions by policy")
    ax.set_xlabel("reward")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "suite_reward_distributions.png", dpi=160)
    plt.close(fig)

    # Response length distributions.
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    for label in labels:
        vals = _numeric([r.get(f"{label}_response_tokens") for r in rows])
        if vals:
            ax.hist(vals, bins=50, alpha=0.35, label=label)
    ax.set_title("Response-token distributions by policy")
    ax.set_xlabel("response tokens")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "suite_response_token_distributions.png", dpi=160)
    plt.close(fig)

    # Overall winner counts.
    counts = {
        label: sum(1 for r in rows if r.get("winner") == label) for label in labels
    }
    counts["tie"] = sum(1 for r in rows if r.get("winner") == "tie")
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    ax.bar(keys, vals)
    ax.set_title("Overall reward winner counts")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "suite_overall_winner_counts.png", dpi=160)
    plt.close(fig)

    # Pairwise deltas.
    for a, b in itertools.combinations(labels, 2):
        vals = _numeric([r.get(f"delta_{b}_minus_{a}") for r in rows])
        if not vals:
            continue
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.hist(vals, bins=50)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"Reward delta: {b} - {a}")
        ax.set_xlabel(f"delta_{b}_minus_{a}")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / f"suite_delta_{b}_minus_{a}.png", dpi=160)
        plt.close(fig)

    # Winner by domain: one grouped bar per domain.
    domains = sorted({str(r.get("domain", "unknown")) for r in rows})
    if domains:
        fig = plt.figure(figsize=(max(7, 1.5 * len(domains)), 4))
        ax = fig.add_subplot(111)
        x = list(range(len(domains)))
        width = 0.8 / max(len(labels), 1)
        for j, label in enumerate(labels):
            rates = []
            for domain in domains:
                subset = [r for r in rows if str(r.get("domain", "unknown")) == domain]
                rates.append(
                    sum(1 for r in subset if r.get("winner") == label)
                    / max(len(subset), 1)
                )
            offsets = [v + (j - (len(labels) - 1) / 2) * width for v in x]
            ax.bar(offsets, rates, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("overall winner rate")
        ax.set_title("Reward winner rate by domain")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "suite_winner_rate_by_domain.png", dpi=160)
        plt.close(fig)


def _write_summary_markdown(summary, output_dir):
    lines = ["# Policy-suite evaluation summary\n\n"]
    lines.append(f"Examples: `{summary.get('num_examples', 0)}`\n\n")
    lines.append(f"Policies: `{', '.join(summary.get('labels', []))}`\n\n")
    lines.append("## Overall winner counts\n\n")
    lines.append(
        "| policy | wins | win rate | mean reward | median response tokens | cap-hit rate | empty rate |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    n = max(int(summary.get("num_examples", 0)), 1)
    for label in summary.get("labels", []):
        wins = int(summary.get("winner_counts", {}).get(label, 0))
        per = summary.get("per_policy", {}).get(label, {})
        lines.append(
            f"| {label} | {wins} | {wins / n:.4f} | "
            f"{per.get('reward', {}).get('mean', 0.0):.4f} | "
            f"{per.get('response_tokens', {}).get('median', 0.0):.1f} | "
            f"{per.get('cap_hit_rate', 0.0):.4f} | {per.get('empty_rate', 0.0):.4f} |\n"
        )
    if int(summary.get("winner_counts", {}).get("tie", 0)):
        ties = int(summary["winner_counts"].get("tie", 0))
        lines.append(f"| tie | {ties} | {ties / n:.4f} |  |  |  |  |\n")
    lines.append("\n## Pairwise comparisons\n\n")
    lines.append(
        "| comparison | left wins | right wins | ties | right win rate | mean right-left reward delta |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for comp, data in summary.get("pairwise", {}).items():
        a, b = data["a"], data["b"]
        counts = data.get("winner_counts", {})
        delta_key = f"{b}_minus_{a}"
        lines.append(
            f"| {a} vs {b} | {int(counts.get(a, 0))} | {int(counts.get(b, 0))} | {int(counts.get('tie', 0))} | "
            f"{data.get(f'{b}_win_rate', 0.0):.4f} | {data.get(delta_key, {}).get('mean', 0.0):.4f} |\n"
        )
    output_dir.joinpath("policy_suite_summary.md").write_text(
        "".join(lines), encoding="utf-8"
    )


def _atomic_write_text(path, text):
    """Write text atomically enough for Colab/Drive-style interrupted runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path, records):
    """Append records to a JSONL file and flush to disk immediately."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass


def _read_jsonl_by_idx(path, *, expected_signature=None):
    """Read a possibly-duplicated JSONL shard. Later rows overwrite earlier rows."""
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if (
                    expected_signature is not None
                    and rec.get("_eval_signature") != expected_signature
                ):
                    continue
                out[int(rec["idx"])] = rec
            except Exception:
                # Ignore a final torn line from an interrupted Colab/Drive write.
                continue
    return out


def _evaluation_signature(cfg, specs, generation, records):
    """Fingerprint inputs that determine cached policy responses and rewards."""
    record_digest = hashlib.sha256()
    for record in records:
        record_digest.update(
            json.dumps(
                {
                    "prompt": record.get("prompt"),
                    "domain": record.get("domain"),
                    "language": record.get("language"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        record_digest.update(b"\n")

    payload = {
        "signature_version": 1,
        "model": dict(cfg.model),
        "reward_model": dict(cfg.reward_model),
        "policies": specs,
        "generation": asdict(generation),
        "records_sha256": record_digest.hexdigest(),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy_shard_path(output_dir, label, partial_subdir):
    return output_dir / partial_subdir / f"{_safe_label(label)}.jsonl"


def _write_progress_manifest(
    output_dir,
    *,
    eval_signature,
    labels,
    completed_by_label,
    total_records,
    partial_subdir,
):
    manifest = {
        "eval_signature": eval_signature,
        "total_records": int(total_records),
        "partial_subdir": partial_subdir,
        "labels": labels,
        "completed_by_label": {
            label: int(completed_by_label.get(label, 0)) for label in labels
        },
        "complete": all(
            int(completed_by_label.get(label, 0)) >= int(total_records)
            for label in labels
        ),
    }
    _atomic_write_text(
        output_dir / "policy_suite_progress.json", json.dumps(manifest, indent=2)
    )


def _finalize_policy_suite_outputs(
    *,
    output_dir,
    rows,
    labels,
    tie_epsilon,
    num_demo_rows,
):
    """Write final combined artifacts from already-populated rows."""
    missing = []
    for label in labels:
        for row in rows:
            if f"{label}_response" not in row or f"{label}_reward" not in row:
                missing.append((label, row.get("idx")))
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break
    if missing:
        raise RuntimeError(
            "Cannot finalize policy-suite eval because some policy outputs are missing. "
            f"Examples: {missing}. Re-run with resume enabled to complete them."
        )

    _add_comparisons(rows, labels, tie_epsilon)
    summary = _summarize(rows, labels, tie_epsilon)
    pairwise_rows = summary.pop("pairwise_rows", [])

    save_jsonl(rows, output_dir / "policy_suite_samples.jsonl")
    write_csv(rows, output_dir / "policy_suite_samples.csv")
    write_json(summary, output_dir / "policy_suite_summary.json")
    write_csv(pairwise_rows, output_dir / "policy_suite_pairwise_summary.csv")
    _write_excel_if_available(rows, output_dir / "policy_suite_samples.xlsx")
    _write_markdown(rows, labels, output_dir / "policy_suite_demo.md", num_demo_rows)
    _write_summary_markdown(summary, output_dir)
    _write_plots(rows, labels, output_dir / "plots")


def run_policy_suite_eval(
    config_path,
    *,
    output_dir=None,
    override_values=None,
):
    cfg = load_config(config_path)
    if override_values:
        from .config import apply_overrides
        from .trl_common import parse_cli_overrides

        cfg = apply_overrides(cfg, parse_cli_overrides(override_values))
    output_dir = Path(
        output_dir
        or cfg.eval.get("output_dir", "outputs/rlhf/qwen25_05b_helpsteer3_eval_suite")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="rlhf_policy_suite_eval",
        config_path=config_path,
        extra={"model_name": str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))},
    )

    from transformers import AutoTokenizer

    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    tokenizer_path = str(cfg.model.get("tokenizer_path", model_name))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        local_files_only=bool(cfg.model.get("local_files_only", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = _device_from_cfg(cfg.eval)
    raw = load_helpsteer3_preference(str(cfg.data.get("eval_split", "validation")))
    records = build_prompt_records(
        raw,
        tokenizer,
        max_samples=_resolve_num_prompts(cfg.eval.get("num_prompts", 200)),
        seed=int(cfg.eval.get("seed", 839)),
        shuffle=bool(cfg.eval.get("shuffle", True)),
    )
    prompts = [r["prompt"] for r in records]
    specs = _policy_specs(cfg)
    labels = [s["label"] for s in specs]
    generation = GenerationConfig(**dict(cfg.get("generation", {})))
    eval_signature = _evaluation_signature(cfg, specs, generation, records)
    batch_size = int(cfg.eval.get("batch_size", 2))
    tie_epsilon = float(cfg.eval.get("tie_epsilon", 0.0))
    load_mode = str(cfg.eval.get("load_mode", "resident")).lower().strip()

    resume = bool(cfg.eval.get("resume", True))
    finalize_only = bool(cfg.eval.get("finalize_only", False))
    partial_subdir = str(cfg.eval.get("partial_subdir", "partial_policy_outputs"))

    rows = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "idx": idx,
                "domain": record.get("domain", "unknown"),
                "language": record.get("language", "unknown"),
                "prompt": record["prompt"],
            }
        )

    # Hydrate rows from per-policy shards written by previous interrupted runs.
    completed_by_label = {}
    for label in labels:
        existing = (
            _read_jsonl_by_idx(
                _policy_shard_path(output_dir, label, partial_subdir),
                expected_signature=eval_signature,
            )
            if resume
            else {}
        )
        for idx, rec in existing.items():
            if 0 <= idx < len(rows):
                for key, value in rec.items():
                    if key not in {"idx", "_eval_signature"}:
                        rows[idx][key] = value
        completed_by_label[label] = sum(
            1 for row in rows if f"{label}_response" in row and f"{label}_reward" in row
        )

    _write_progress_manifest(
        output_dir,
        eval_signature=eval_signature,
        labels=labels,
        completed_by_label=completed_by_label,
        total_records=len(rows),
        partial_subdir=partial_subdir,
    )

    if finalize_only:
        _finalize_policy_suite_outputs(
            output_dir=output_dir,
            rows=rows,
            labels=labels,
            tie_epsilon=tie_epsilon,
            num_demo_rows=int(cfg.eval.get("num_demo_rows", 12)),
        )
        finalize_experiment(output_dir)
        return output_dir

    reward_model = _load_reward_model(cfg, device)

    def fill_policy_outputs(policy, label):
        shard_path = _policy_shard_path(output_dir, label, partial_subdir)
        pending_indices = [
            idx
            for idx, row in enumerate(rows)
            if not (resume and f"{label}_response" in row and f"{label}_reward" in row)
        ]
        if not pending_indices:
            print(f"eval {label}: already complete ({len(rows)}/{len(rows)}); skipping")
            return

        progress_every = int(cfg.eval.get("progress_every", 5))
        total_rows = len(rows)
        pbar = tqdm(
            range(0, len(pending_indices), batch_size),
            desc=f"eval {label}",
            dynamic_ncols=True,
            mininterval=5,
            leave=True,
        )
        for batch_number, offset in enumerate(pbar, start=1):
            batch_indices = pending_indices[offset : offset + batch_size]
            batch_prompts = [prompts[i] for i in batch_indices]
            outputs = _generate_and_score(
                policy,
                reward_model,
                tokenizer,
                batch_prompts,
                generation=generation,
                device=device,
            )
            shard_records = []
            for idx, out in zip(batch_indices, outputs):
                row = rows[idx]
                update = {
                    "idx": int(idx),
                    "domain": row.get("domain", "unknown"),
                    "language": row.get("language", "unknown"),
                    f"{label}_response": out["response"],
                    f"{label}_reward": out["reward"],
                    f"{label}_response_tokens": out["response_tokens"],
                    f"{label}_response_chars": out["response_chars"],
                    f"{label}_prompt_tokens": out["prompt_tokens"],
                    f"{label}_total_tokens": out["total_tokens"],
                    f"{label}_hit_eos": out["hit_eos"],
                    f"{label}_cap_hit": out["cap_hit"],
                    f"{label}_empty": out["empty"],
                }
                row.update(update)
                shard_records.append({"_eval_signature": eval_signature, **update})
            _append_jsonl(shard_path, shard_records)
            completed_by_label[label] = sum(
                1
                for row in rows
                if f"{label}_response" in row and f"{label}_reward" in row
            )
            _write_progress_manifest(
                output_dir,
                eval_signature=eval_signature,
                labels=labels,
                completed_by_label=completed_by_label,
                total_records=len(rows),
                partial_subdir=partial_subdir,
            )
            pbar.set_postfix(done=completed_by_label[label], total=len(rows))
            completed = completed_by_label[label]
            if (
                batch_number == 1
                or batch_number % progress_every == 0
                or completed == total_rows
            ):
                print(
                    f"[eval {label}] {completed}/{total_rows} complete "
                    f"({100.0 * completed / max(total_rows, 1):.1f}%)",
                    flush=True,
                )

    def label_complete(label):
        return all(
            f"{label}_response" in row and f"{label}_reward" in row for row in rows
        )

    if load_mode == "resident":
        policies = [
            (spec["label"], _load_policy(cfg, spec, device, tokenizer))
            for spec in specs
            if not label_complete(spec["label"])
        ]
        for label, policy in policies:
            fill_policy_outputs(policy, label)
    elif load_mode == "sequential":
        for spec in specs:
            label = spec["label"]
            if label_complete(label):
                print(
                    f"eval {label}: already complete ({len(rows)}/{len(rows)}); skipping load"
                )
                continue
            policy = _load_policy(cfg, spec, device, tokenizer)
            fill_policy_outputs(policy, label)
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        raise ValueError("eval.load_mode must be 'resident' or 'sequential'.")

    _finalize_policy_suite_outputs(
        output_dir=output_dir,
        rows=rows,
        labels=labels,
        tie_epsilon=tie_epsilon,
        num_demo_rows=int(cfg.eval.get("num_demo_rows", 12)),
    )
    finalize_experiment(output_dir)
    return output_dir
