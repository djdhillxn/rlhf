from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import load_config, save_config
from .data import build_prompt_records, load_helpsteer3_preference, save_jsonl
from .experiment import finalize_experiment, initialize_experiment
from .lm_policy import FrozenCausalLM, TokenPolicyWithValue
from .metrics import write_csv, write_json
from .reward_model import RewardModel
from .rollout import GenerationConfig, collect_lm_rollouts


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


def _load_policy_or_base(cfg, checkpoint_dir, device):
    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    if checkpoint_dir:
        policy = TokenPolicyWithValue.load_rlhf_pretrained(
            checkpoint_dir,
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
            lora=None,
            gradient_checkpointing=False,
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    if cfg.model.get("policy_device_map") is None:
        policy.to(device)
    policy.eval()
    return policy


def _comparison_cfg(cfg):
    """Return backward-compatible before/after comparison settings.

    Old configs only had eval.policy_checkpoint_dir, and compared base Qwen vs PPO.
    New configs may set eval.baseline_checkpoint_dir and labels, letting us compare
    base-vs-SFT, base-vs-PPO, or SFT-vs-PPO with the same evaluator.
    """
    candidate_checkpoint = cfg.eval.get(
        "candidate_checkpoint_dir", cfg.eval.get("policy_checkpoint_dir")
    )
    baseline_checkpoint = cfg.eval.get("baseline_checkpoint_dir", None)
    baseline_label = str(cfg.eval.get("baseline_label", "base"))
    candidate_label = str(
        cfg.eval.get("candidate_label", cfg.eval.get("policy_label", "ppo"))
    )
    return {
        "baseline_checkpoint_dir": None
        if baseline_checkpoint in {None, "", "none", "null"}
        else str(baseline_checkpoint),
        "candidate_checkpoint_dir": None
        if candidate_checkpoint in {None, "", "none", "null"}
        else str(candidate_checkpoint),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
    }


def run_before_after_eval(config_path, *, output_dir=None):
    cfg = load_config(config_path)
    output_dir = Path(
        output_dir
        or cfg.eval.get("output_dir", "outputs/rlhf/qwen25_05b_helpsteer3_eval")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="rlhf_before_after_eval",
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
    device = _device_from_cfg(cfg.eval)

    raw = load_helpsteer3_preference(str(cfg.data.get("eval_split", "validation")))
    records = build_prompt_records(
        raw,
        tokenizer,
        max_samples=_resolve_num_prompts(cfg.eval.get("num_prompts", 100)),
        seed=int(cfg.eval.get("seed", 839)),
        shuffle=bool(cfg.eval.get("shuffle", True)),
    )
    prompts = [r["prompt"] for r in records]
    cmp_cfg = _comparison_cfg(cfg)

    baseline_policy = _load_policy_or_base(
        cfg, cmp_cfg["baseline_checkpoint_dir"], device
    )
    candidate_policy = _load_policy_or_base(
        cfg, cmp_cfg["candidate_checkpoint_dir"], device
    )
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

    reward_model = RewardModel.load_rlhf_pretrained(
        str(cfg.reward_model.get("checkpoint_dir")),
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
    batch_size = int(cfg.eval.get("batch_size", 4))
    rows = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="eval batches"):
        batch_prompts = prompts[start : start + batch_size]
        meta = records[start : start + batch_size]
        baseline = collect_lm_rollouts(
            baseline_policy,
            reference,
            reward_model,
            tokenizer,
            batch_prompts,
            generation=generation,
            kl_coef=0.0,
            device=device,
            metadata=meta,
        )
        candidate = collect_lm_rollouts(
            candidate_policy,
            reference,
            reward_model,
            tokenizer,
            batch_prompts,
            generation=generation,
            kl_coef=0.0,
            device=device,
            metadata=meta,
        )
        baseline_tokens = baseline.response_mask.sum(dim=1).detach().cpu().tolist()
        candidate_tokens = candidate.response_mask.sum(dim=1).detach().cpu().tolist()
        for i, prompt in enumerate(batch_prompts):
            baseline_score = float(baseline.scores[i].item())
            candidate_score = float(candidate.scores[i].item())
            winner = (
                cmp_cfg["candidate_label"]
                if candidate_score > baseline_score
                else cmp_cfg["baseline_label"]
            )
            baseline_response = (baseline.responses or [""])[i]
            candidate_response = (candidate.responses or [""])[i]
            row = {
                "idx": start + i,
                "domain": meta[i].get("domain", "unknown"),
                "language": meta[i].get("language", "unknown"),
                "prompt": prompt,
                "baseline_label": cmp_cfg["baseline_label"],
                "candidate_label": cmp_cfg["candidate_label"],
                "baseline_response": baseline_response,
                "candidate_response": candidate_response,
                "baseline_reward": baseline_score,
                "candidate_reward": candidate_score,
                "baseline_response_tokens": int(baseline_tokens[i]),
                "candidate_response_tokens": int(candidate_tokens[i]),
                "baseline_response_chars": len(str(baseline_response)),
                "candidate_response_chars": len(str(candidate_response)),
                "reward_delta": candidate_score - baseline_score,
                "winner": winner,
            }
            # Backward-compatible aliases used by older notebooks/report snippets.
            row.update(
                {
                    "base_response": baseline_response,
                    "ppo_response": candidate_response,
                    "base_reward": baseline_score,
                    "ppo_reward": candidate_score,
                    "base_response_tokens": int(baseline_tokens[i]),
                    "ppo_response_tokens": int(candidate_tokens[i]),
                    "base_response_chars": len(str(baseline_response)),
                    "ppo_response_chars": len(str(candidate_response)),
                }
            )
            rows.append(row)

    save_jsonl(rows, output_dir / "before_after_samples.jsonl")
    write_csv(rows, output_dir / "before_after_samples.csv")
    _write_eval_summary(rows, output_dir)
    _write_eval_plots(rows, output_dir / "plots")
    _write_excel_if_available(rows, output_dir / "before_after_samples.xlsx")
    _write_markdown_table(
        rows[: int(cfg.eval.get("num_demo_rows", 12))],
        output_dir / "before_after_demo.md",
    )
    finalize_experiment(output_dir)
    return output_dir


def _safe_len(text):
    return len(str(text or ""))


def _numeric(values):
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def _stats(values):
    import statistics

    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _write_eval_summary(rows, output_dir):
    """Write compact report-friendly aggregate metrics for pairwise eval."""
    output_dir = Path(output_dir)
    if not rows:
        write_json({"num_examples": 0}, output_dir / "eval_summary.json")
        return
    winner_counts = {}
    domain_counts = {}
    reward_deltas = []
    baseline_rewards = []
    candidate_rewards = []
    baseline_chars = []
    candidate_chars = []
    baseline_tokens = []
    candidate_tokens = []
    baseline_label = str(rows[0].get("baseline_label", "base"))
    candidate_label = str(rows[0].get("candidate_label", "candidate"))
    for row in rows:
        winner = str(row.get("winner", "unknown"))
        domain = str(row.get("domain", "unknown"))
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        domain_counts.setdefault(domain, {})[winner] = (
            domain_counts.setdefault(domain, {}).get(winner, 0) + 1
        )
        reward_deltas.extend(_numeric([row.get("reward_delta", 0.0)]))
        baseline_rewards.extend(
            _numeric([row.get("baseline_reward", row.get("base_reward", 0.0))])
        )
        candidate_rewards.extend(
            _numeric([row.get("candidate_reward", row.get("ppo_reward", 0.0))])
        )
        baseline_chars.append(
            int(
                row.get(
                    "baseline_response_chars",
                    _safe_len(
                        row.get("baseline_response", row.get("base_response", ""))
                    ),
                )
            )
        )
        candidate_chars.append(
            int(
                row.get(
                    "candidate_response_chars",
                    _safe_len(
                        row.get("candidate_response", row.get("ppo_response", ""))
                    ),
                )
            )
        )
        baseline_tokens.extend(
            [
                int(x)
                for x in _numeric(
                    [
                        row.get(
                            "baseline_response_tokens",
                            row.get("base_response_tokens", 0),
                        )
                    ]
                )
            ]
        )
        candidate_tokens.extend(
            [
                int(x)
                for x in _numeric(
                    [
                        row.get(
                            "candidate_response_tokens",
                            row.get("ppo_response_tokens", 0),
                        )
                    ]
                )
            ]
        )

    summary = {
        "num_examples": len(rows),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "comparison": f"{baseline_label}_vs_{candidate_label}",
        "winner_counts": winner_counts,
        "domain_winner_counts": domain_counts,
        "baseline_reward": _stats(baseline_rewards),
        "candidate_reward": _stats(candidate_rewards),
        "reward_delta": _stats(reward_deltas),
        "baseline_response_chars": _stats(baseline_chars),
        "candidate_response_chars": _stats(candidate_chars),
        "baseline_response_tokens": _stats(baseline_tokens),
        "candidate_response_tokens": _stats(candidate_tokens),
        "candidate_win_rate": float(
            winner_counts.get(candidate_label, 0) / max(len(rows), 1)
        ),
        # Legacy key expected by older notebooks when candidate_label == ppo.
        "ppo_win_rate": float(
            winner_counts.get(candidate_label, 0) / max(len(rows), 1)
        ),
    }
    write_json(summary, output_dir / "eval_summary.json")


def _write_eval_plots(rows, output_dir):
    """Write lightweight reward/length distribution plots for reports."""
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_label = str(rows[0].get("baseline_label", "base"))
    candidate_label = str(rows[0].get("candidate_label", "candidate"))

    reward_delta = _numeric([r.get("reward_delta") for r in rows])
    if reward_delta:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.hist(reward_delta, bins=50)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"Reward delta: {candidate_label} - {baseline_label}")
        ax.set_xlabel("reward_delta")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / "eval_reward_delta_hist.png", dpi=160)
        plt.close(fig)

    baseline_rewards = _numeric(
        [r.get("baseline_reward", r.get("base_reward")) for r in rows]
    )
    candidate_rewards = _numeric(
        [r.get("candidate_reward", r.get("ppo_reward")) for r in rows]
    )
    if (
        baseline_rewards
        and candidate_rewards
        and len(baseline_rewards) == len(candidate_rewards)
    ):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.scatter(baseline_rewards, candidate_rewards, s=8, alpha=0.5)
        lo = min(min(baseline_rewards), min(candidate_rewards))
        hi = max(max(baseline_rewards), max(candidate_rewards))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xlabel(f"{baseline_label} reward")
        ax.set_ylabel(f"{candidate_label} reward")
        ax.set_title("Reward scatter")
        fig.tight_layout()
        fig.savefig(output_dir / "eval_reward_scatter.png", dpi=160)
        plt.close(fig)

    for key, title in [
        ("baseline_response_tokens", f"{baseline_label} response tokens"),
        ("candidate_response_tokens", f"{candidate_label} response tokens"),
        ("baseline_response_chars", f"{baseline_label} response chars"),
        ("candidate_response_chars", f"{candidate_label} response chars"),
    ]:
        vals = _numeric([r.get(key) for r in rows])
        if vals:
            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(111)
            ax.hist(vals, bins=50)
            ax.set_title(title)
            ax.set_xlabel(key)
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(output_dir / f"eval_{key}_hist.png", dpi=160)
            plt.close(fig)

    domains = sorted({str(r.get("domain", "unknown")) for r in rows})
    if domains:
        candidate_rates = []
        counts = []
        for domain in domains:
            subset = [r for r in rows if str(r.get("domain", "unknown")) == domain]
            counts.append(len(subset))
            candidate_rates.append(
                sum(1 for r in subset if r.get("winner") == candidate_label)
                / max(len(subset), 1)
            )
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.bar(domains, candidate_rates)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"{candidate_label} win rate")
        ax.set_title("Win rate by domain")
        for idx, count in enumerate(counts):
            ax.text(
                idx,
                candidate_rates[idx],
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(output_dir / "eval_candidate_win_rate_by_domain.png", dpi=160)
        plt.close(fig)


def _write_excel_if_available(rows, path):
    """Write an XLSX copy when pandas/openpyxl are available; silently skip otherwise."""
    try:
        import pandas as pd

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_excel(path, index=False)
    except Exception:
        return


def _compact(text, n=900):
    s = str(text or "").replace("\n", "<br>")
    return s if len(s) <= n else s[:n] + "..."


def _write_markdown_table(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# Before/after examples\n\nNo rows.\n", encoding="utf-8")
        return
    baseline_label = str(rows[0].get("baseline_label", "base"))
    candidate_label = str(rows[0].get("candidate_label", "candidate"))
    lines = [
        "# Before/after examples\n\n",
        "This Markdown preview is intentionally truncated for readability. ",
        "Use `before_after_samples.jsonl` or the curation notebook for complete responses.\n\n",
        f"Comparison: **{baseline_label}** vs **{candidate_label}**\n\n",
        "| idx | domain | winner | reward delta | prompt | baseline response | candidate response |\n",
        "|---:|---|---|---:|---|---|---|\n",
    ]
    for row in rows:
        lines.append(
            "| {idx} | {domain} | {winner} | {delta:.3f} | {prompt} | {base} | {cand} |\n".format(
                idx=row.get("idx", ""),
                domain=row.get("domain", ""),
                winner=row.get("winner", ""),
                delta=float(row.get("reward_delta", 0.0)),
                prompt=_compact(row.get("prompt", ""), 260),
                base=_compact(
                    row.get("baseline_response", row.get("base_response", "")), 420
                ),
                cand=_compact(
                    row.get("candidate_response", row.get("ppo_response", "")), 420
                ),
            )
        )
    path.write_text("".join(lines), encoding="utf-8")
