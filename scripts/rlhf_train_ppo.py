import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(description="Run token-level PPO RLHF on Qwen with HelpSteer3 prompts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    ensure_repo_root_on_path()
    from rlhf.train_ppo import run_ppo_training

    out = run_ppo_training(args.config, output_dir=args.output_dir)
    print(f"PPO policy saved to: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
