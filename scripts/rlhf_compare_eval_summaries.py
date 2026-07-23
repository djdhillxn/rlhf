#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def _load(path):
    summary_path = path / "eval_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["eval_dir"] = str(path)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple RLHF eval_summary.json files into one table."
    )
    parser.add_argument("eval_dirs", nargs="*", help="Evaluation output directories")
    parser.add_argument(
        "--root", default="outputs/rlhf", help="Used when eval_dirs is empty"
    )
    parser.add_argument("--out-csv", default="outputs/rlhf/eval_comparison_summary.csv")
    parser.add_argument(
        "--out-json", default="outputs/rlhf/eval_comparison_summary.json"
    )
    args = parser.parse_args()
    ensure_repo_root_on_path()
    from rlhf.metrics import write_csv, write_json

    dirs = [Path(x) for x in args.eval_dirs]
    if not dirs:
        root = Path(args.root)
        names = [
            "qwen25_05b_helpsteer3_eval_full",
            "qwen25_05b_helpsteer3_eval_sft_full",
            "qwen25_05b_helpsteer3_eval_sft_vs_ppo_full",
            "qwen25_05b_helpsteer3_eval",
            "qwen25_05b_helpsteer3_eval_sft",
            "qwen25_05b_helpsteer3_eval_sft_vs_ppo",
        ]
        dirs = [
            root / name
            for name in names
            if (root / name / "eval_summary.json").exists()
        ]
    rows = []
    for d in dirs:
        s = _load(d)
        baseline_label = s.get("baseline_label", "base")
        candidate_label = s.get("candidate_label", "candidate")
        winners = s.get("winner_counts", {})
        delta = s.get("reward_delta", {})
        rows.append(
            {
                "eval_dir": str(d),
                "comparison": s.get(
                    "comparison", f"{baseline_label}_vs_{candidate_label}"
                ),
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
                "num_examples": s.get("num_examples", 0),
                "baseline_wins": winners.get(baseline_label, 0),
                "candidate_wins": winners.get(candidate_label, 0),
                "candidate_win_rate": s.get(
                    "candidate_win_rate", s.get("ppo_win_rate", 0.0)
                ),
                "reward_delta_mean": delta.get("mean", 0.0),
                "reward_delta_median": delta.get("median", 0.0),
                "baseline_reward_mean": s.get(
                    "baseline_reward", s.get("base_reward", {})
                ).get("mean", 0.0),
                "candidate_reward_mean": s.get(
                    "candidate_reward", s.get("ppo_reward", {})
                ).get("mean", 0.0),
                "baseline_response_tokens_mean": s.get(
                    "baseline_response_tokens", s.get("base_response_tokens", {})
                ).get("mean", 0.0),
                "candidate_response_tokens_mean": s.get(
                    "candidate_response_tokens", s.get("ppo_response_tokens", {})
                ).get("mean", 0.0),
            }
        )
    write_csv(rows, args.out_csv)
    write_json({"rows": rows}, args.out_json)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
