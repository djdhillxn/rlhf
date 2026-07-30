#!/usr/bin/env python3
import argparse

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare response-safe HelpSteer3 datasets for TRL."
    )
    parser.add_argument(
        "--config", default="configs/trl/qwen25_05b_helpsteer3_sft.yaml"
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    ensure_repo_root_on_path()

    from rlhf.trl_common import load_config_with_overrides, load_tokenizer
    from rlhf.trl_data import prepare_helpsteer3_for_trl

    cfg = load_config_with_overrides(args.config, args.set)
    tokenizer_source = cfg.model.get("sft_model_path", cfg.model.get("name"))
    if not tokenizer_source:
        raise ValueError("Config must define model.name or model.sft_model_path.")
    tokenizer = load_tokenizer(
        str(tokenizer_source),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        padding_side="right",
    )
    report = prepare_helpsteer3_for_trl(cfg.data, tokenizer)
    print(f"Prepared TRL datasets under {report['cache_dir']}")


if __name__ == "__main__":
    main()
