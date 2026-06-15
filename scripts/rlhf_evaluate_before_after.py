import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs PPO-aligned Qwen responses.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    ensure_repo_root_on_path()
    from rlhf.evaluate import run_before_after_eval

    out = run_before_after_eval(args.config, output_dir=args.output_dir)
    print(f"Evaluation artifacts saved to: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
