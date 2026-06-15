import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(description="Create CSV files and PNG learning curves from RLHF JSONL logs.")
    parser.add_argument("--run-dir", required=True, help="RLHF output directory, e.g. outputs/rlhf/qwen25_05b_helpsteer3_ppo")
    parser.add_argument("--kind", choices=["reward", "ppo"], required=True)
    args = parser.parse_args()
    ensure_repo_root_on_path()

    from rlhf.metrics import jsonl_to_csv, read_jsonl, save_metric_plots, write_json

    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "reward":
        train_jsonl = run_dir / "train_metrics.jsonl"
        eval_jsonl = run_dir / "eval_metrics.jsonl"
        jsonl_to_csv(train_jsonl, run_dir / "train_metrics.csv")
        jsonl_to_csv(eval_jsonl, run_dir / "eval_metrics.csv")
        paths = []
        paths.extend(
            save_metric_plots(
                read_jsonl(train_jsonl),
                plots_dir,
                x_key="step",
                y_keys=["loss", "accuracy_batch", "reward_margin_batch"],
                prefix="reward_train",
            )
        )
        paths.extend(
            save_metric_plots(
                read_jsonl(eval_jsonl),
                plots_dir,
                x_key="step",
                y_keys=["loss", "accuracy", "avg_margin"],
                prefix="reward_eval",
            )
        )
    else:
        metrics_jsonl = run_dir / "ppo_metrics.jsonl"
        jsonl_to_csv(metrics_jsonl, run_dir / "ppo_metrics.csv")
        paths = save_metric_plots(
            read_jsonl(metrics_jsonl),
            plots_dir,
            x_key="update",
            y_keys=[
                "reward_model_score",
                "total_reward",
                "objective_kl",
                "kl_coef",
                "approx_kl",
                "clip_fraction",
                "loss",
                "policy_loss",
                "value_loss",
                "mean_response_tokens",
            ],
            prefix="ppo",
        )

    write_json({"plot_paths": paths}, run_dir / "plot_summary.json")
    print(f"Wrote {len(paths)} plots under {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
