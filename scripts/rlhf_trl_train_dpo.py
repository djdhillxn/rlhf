#!/usr/bin/env python3
import argparse

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(
        description="Train HelpSteer3 preferences with Hugging Face TRL DPO."
    )
    parser.add_argument(
        "--config", default="configs/trl/qwen25_05b_helpsteer3_dpo.yaml"
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    ensure_repo_root_on_path()

    from rlhf.trl_common import load_config_with_overrides
    from rlhf.trl_train_dpo import run_trl_dpo

    cfg = load_config_with_overrides(args.config, args.set)
    output = run_trl_dpo(cfg, config_path=args.config)
    print(f"TRL DPO output: {output.resolve()}")


if __name__ == "__main__":
    main()
