import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(
        description="Cache HelpSteer3 preference pairs/prompts as JSONL previews."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/rlhf/helpsteer3_cache")
    args = parser.parse_args()
    ensure_repo_root_on_path()
    from transformers import AutoTokenizer
    from rlhf.data import (
        build_preference_pairs,
        build_prompt_records,
        load_helpsteer3_preference,
        preference_pairs_to_dicts,
        save_jsonl,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_helpsteer3_preference(args.split)
    pairs = build_preference_pairs(raw, tokenizer, max_samples=args.max_samples)
    raw = load_helpsteer3_preference(args.split)
    prompts = build_prompt_records(raw, tokenizer, max_samples=args.max_samples)
    out = Path(args.output_dir)
    save_jsonl(preference_pairs_to_dicts(pairs), out / f"{args.split}_pairs.jsonl")
    save_jsonl(prompts, out / f"{args.split}_prompts.jsonl")
    print(f"Saved {len(pairs)} pairs and {len(prompts)} prompts to {out.resolve()}")


if __name__ == "__main__":
    main()
