#!/usr/bin/env python3
import argparse

from _bootstrap import ensure_repo_root_on_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare response-safe HelpSteer3 datasets for TRL.")
    parser.add_argument("--config", default="configs/trl/qwen25_05b_helpsteer3_sft.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    ensure_repo_root_on_path()

    from rlhf.trl_common import load_config_with_overrides, load_tokenizer
    from rlhf.trl_data import prepare_helpsteer3_for_trl

    cfg = load_config_with_overrides(args.config, args.set)
    tokenizer = load_tokenizer(
        str(cfg.model.name),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        padding_side="right",
    )
    report = prepare_helpsteer3_for_trl(cfg.data, tokenizer)
    print(f"Prepared TRL datasets under {report['cache_dir']}")


if __name__ == "__main__":
    main()
