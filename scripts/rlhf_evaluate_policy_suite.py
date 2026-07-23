import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base/SFT/PPO or any policy suite in one reward-model pass."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config field. May be repeated.",
    )
    args = parser.parse_args()
    ensure_repo_root_on_path()
    from rlhf.evaluate_suite import run_policy_suite_eval

    out = run_policy_suite_eval(
        args.config, output_dir=args.output_dir, override_values=args.set
    )
    print(f"Policy-suite evaluation artifacts saved to: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
