import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(description="Train Qwen reward model on HelpSteer3 preference pairs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None, help="Optional reward checkpoint to continue training from.")
    parser.add_argument(
        "--clear-existing-metrics",
        action="store_true",
        help="Clear metric/artifact files in the output dir before starting this run.",
    )
    args = parser.parse_args()
    ensure_repo_root_on_path()
    from rlhf.config import load_config
    from rlhf.train_reward_model import run_reward_training

    cfg_path = args.config
    temp_path = None
    if args.resume_from_checkpoint or args.clear_existing_metrics:
        import tempfile
        from pathlib import Path as _Path

        cfg = load_config(args.config)
        if args.resume_from_checkpoint:
            cfg.train["resume_from_checkpoint"] = args.resume_from_checkpoint
        if args.clear_existing_metrics:
            cfg.train["clear_existing_metrics"] = True
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        temp_path = _Path(handle.name)
        handle.close()
        from rlhf.config import save_config

        save_config(cfg, temp_path)
        cfg_path = temp_path

    out = run_reward_training(cfg_path, output_dir=args.output_dir)
    if temp_path is not None:
        try:
            temp_path.unlink()
        except Exception:
            pass
    print(f"Reward model saved to: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
