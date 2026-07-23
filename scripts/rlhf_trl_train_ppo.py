#!/usr/bin/env python3
import argparse

from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(
        description="Run N+-style PPO alignment with Hugging Face TRL."
    )
    parser.add_argument(
        "--config", default="configs/trl/qwen25_05b_helpsteer3_ppo.yaml"
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    ensure_repo_root_on_path()

    from rlhf.trl_common import load_config_with_overrides
    from rlhf.trl_train_ppo import run_trl_ppo

    cfg = load_config_with_overrides(args.config, args.set)
    print(f"TRL PPO output: {run_trl_ppo(cfg, config_path=args.config).resolve()}")


if __name__ == "__main__":
    main()
