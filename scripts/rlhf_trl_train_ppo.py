#!/usr/bin/env python3
import argparse

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(
        description="Run N+-style PPO alignment with Hugging Face TRL."
    )
    parser.add_argument(
        "--config", default="configs/trl/qwen25_05b_helpsteer3_ppo.yaml"
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--export-checkpoint",
        help="Merge and export the policy adapter from a completed guarded checkpoint.",
    )
    parser.add_argument(
        "--export-output-dir",
        help="Destination for --export-checkpoint.",
    )
    parser.add_argument(
        "--prepare-guardrails-only",
        action="store_true",
        help="Calibrate and persist reward/EOS/repetition guardrails without training.",
    )
    args = parser.parse_args()
    if args.prepare_guardrails_only and args.export_checkpoint:
        parser.error(
            "--prepare-guardrails-only and --export-checkpoint are mutually exclusive"
        )
    ensure_repo_root_on_path()

    from rlhf.trl_common import load_config_with_overrides
    from rlhf.trl_train_ppo import (
        export_ppo_policy,
        prepare_ppo_guardrails,
        run_trl_ppo,
    )

    cfg = load_config_with_overrides(args.config, args.set)
    if args.prepare_guardrails_only:
        output = prepare_ppo_guardrails(cfg)
        print(f"PPO reward guardrails: {output.resolve()}")
        return
    if args.export_checkpoint:
        if not args.export_output_dir:
            parser.error("--export-checkpoint requires --export-output-dir")
        output = export_ppo_policy(cfg, args.export_checkpoint, args.export_output_dir)
        print(f"Exported PPO policy: {output.resolve()}")
        return
    print(f"TRL PPO output: {run_trl_ppo(cfg, config_path=args.config).resolve()}")


if __name__ == "__main__":
    main()
