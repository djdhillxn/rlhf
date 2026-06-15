#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd

BAD_PATTERNS = [
    r"erot", r"adult", r"sex", r"porn", r"cunt", r"busty", r"voyeur", r"hooker", r"libertin",
    r"blackColor", r"didReceiveMemoryWarning", r"SimpleName", r"HTTPHeader", r"numel",
]
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _bad_hits(text: str) -> int:
    lower = text.lower()
    return sum(1 for pat in BAD_PATTERNS if re.search(pat, lower))


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(NON_ASCII_RE.findall(text)) / max(len(text), 1)


def inspect_run(root: Path) -> dict:
    reward_dir = root / "qwen25_05b_helpsteer3_reward"
    ppo_dir = root / "qwen25_05b_helpsteer3_ppo"
    eval_dir = root / "qwen25_05b_helpsteer3_eval"
    report: dict = {"root": str(root)}

    reward_final = _load_json(reward_dir / "final_eval_metrics.json")
    if reward_final:
        report["reward_model"] = reward_final

    ppo_csv = ppo_dir / "ppo_metrics.csv"
    if ppo_csv.exists():
        ppo = pd.read_csv(ppo_csv)
        summary = {"num_updates": int(len(ppo))}
        for col in [
            "reward_model_score", "objective_kl", "abs_ref_logratio", "kl_coef", "clip_fraction",
            "approx_kl", "mean_response_tokens", "mean_response_chars", "value_loss", "policy_loss",
        ]:
            if col in ppo.columns:
                s = pd.to_numeric(ppo[col], errors="coerce").dropna()
                if len(s):
                    summary[col] = {
                        "first": float(s.iloc[0]), "last": float(s.iloc[-1]), "min": float(s.min()),
                        "max": float(s.max()), "mean": float(s.mean()), "median": float(s.median()),
                    }
        flags = []
        if "kl_coef" in ppo.columns and float(pd.to_numeric(ppo["kl_coef"]).min()) < 0.005:
            flags.append("KL coefficient collapsed below 0.005; policy can drift away from reference.")
        if "mean_response_tokens" in ppo.columns and float(pd.to_numeric(ppo["mean_response_tokens"]).tail(20).mean()) > 0.95 * float(pd.to_numeric(ppo["mean_response_tokens"]).max()):
            flags.append("Responses are frequently hitting max_new_tokens near the end of training.")
        if "clip_fraction" in ppo.columns and float(pd.to_numeric(ppo["clip_fraction"]).tail(20).mean()) > 0.35:
            flags.append("High late PPO clip fraction; updates are too aggressive or policy is unstable.")
        summary["flags"] = flags
        report["ppo"] = summary

    eval_csv = eval_dir / "before_after_samples.csv"
    if eval_csv.exists():
        ev = pd.read_csv(eval_csv)
        ppo_responses = ev.get("ppo_response", pd.Series(dtype=str)).fillna("").astype(str)
        base_responses = ev.get("base_response", pd.Series(dtype=str)).fillna("").astype(str)
        eval_summary = {
            "num_examples": int(len(ev)),
            "winner_counts": ev["winner"].value_counts().to_dict() if "winner" in ev.columns else {},
        }
        for col in ["base_reward", "ppo_reward", "reward_delta"]:
            if col in ev.columns:
                s = pd.to_numeric(ev[col], errors="coerce").dropna()
                if len(s):
                    eval_summary[col] = {
                        "mean": float(s.mean()), "median": float(s.median()),
                        "min": float(s.min()), "max": float(s.max()),
                    }
        eval_summary["ppo_non_ascii_ratio_mean"] = float(ppo_responses.map(_non_ascii_ratio).mean()) if len(ppo_responses) else 0.0
        eval_summary["base_non_ascii_ratio_mean"] = float(base_responses.map(_non_ascii_ratio).mean()) if len(base_responses) else 0.0
        eval_summary["ppo_cjk_response_count"] = int(ppo_responses.map(lambda x: bool(CJK_RE.search(x))).sum())
        eval_summary["ppo_bad_pattern_response_count"] = int(ppo_responses.map(lambda x: _bad_hits(x) > 0).sum())
        eval_summary["flags"] = []
        if eval_summary["winner_counts"].get("base", 0) > eval_summary["winner_counts"].get("ppo", 0):
            eval_summary["flags"].append("Base model wins more examples than PPO under the reward model.")
        if eval_summary["ppo_bad_pattern_response_count"] > 0:
            eval_summary["flags"].append("PPO responses contain known degenerate/toxic/debug-token patterns.")
        if eval_summary["ppo_cjk_response_count"] > 0:
            eval_summary["flags"].append("PPO responses contain CJK characters; verify multilingual prompts or drift.")
        report["evaluation"] = eval_summary
    return report


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# RLHF Run Inspection\n"]
    reward = report.get("reward_model")
    if reward:
        lines += [
            "## Reward model\n",
            f"- Validation pairwise accuracy: **{reward.get('accuracy', float('nan')):.4f}**\n",
            f"- Validation loss: **{reward.get('loss', float('nan')):.4f}**\n",
            f"- Average reward margin: **{reward.get('avg_margin', float('nan')):.4f}**\n",
        ]
    ppo = report.get("ppo")
    if ppo:
        lines += ["\n## PPO training\n", f"- Updates: **{ppo.get('num_updates')}**\n"]
        for key in ["reward_model_score", "objective_kl", "abs_ref_logratio", "kl_coef", "clip_fraction", "mean_response_tokens"]:
            if key in ppo:
                v = ppo[key]
                lines.append(f"- {key}: first={v['first']:.4f}, last={v['last']:.4f}, min={v['min']:.4f}, max={v['max']:.4f}, mean={v['mean']:.4f}\n")
        for flag in ppo.get("flags", []):
            lines.append(f"  - ⚠️ {flag}\n")
    ev = report.get("evaluation")
    if ev:
        lines += ["\n## Before/after evaluation\n", f"- Examples: **{ev.get('num_examples')}**\n", f"- Winner counts: `{ev.get('winner_counts')}`\n"]
        if "reward_delta" in ev:
            lines.append(f"- Mean reward delta: **{ev['reward_delta']['mean']:.4f}**\n")
        lines.append(f"- PPO responses with bad/debug patterns: **{ev.get('ppo_bad_pattern_response_count')}**\n")
        lines.append(f"- PPO responses with CJK characters: **{ev.get('ppo_cjk_response_count')}**\n")
        for flag in ev.get("flags", []):
            lines.append(f"  - ⚠️ {flag}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/rlhf", help="RLHF output root directory")
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    parser.add_argument("--md", default=None, help="Optional Markdown report path")
    args = parser.parse_args()
    report = inspect_run(Path(args.root))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.md:
        write_markdown(report, Path(args.md))


if __name__ == "__main__":
    main()
