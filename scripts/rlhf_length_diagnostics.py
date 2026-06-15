#!/usr/bin/env python3
"""
Token-length diagnostics for HelpSteer3 SFT / reward-model training.

What this measures:
  - prompt token length under Qwen chat template
  - prompt + chosen response token length
  - prompt + rejected response token length
  - how much data would be truncated at limits like 1024, 2048, 3072, 4096
  - how many response tokens are lost because the prompt + response exceeds the cap
  - domain-wise truncation rates

Run from repo root:
  python3 scripts/rlhf_length_diagnostics.py \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --splits train validation \
    --limits 1024 2048 3072 4096 \
    --output-dir outputs/rlhf/length_diagnostics
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import os
import sys

# Make the src-layout package importable when running from a checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def load_preference_split(split: str):
    """Prefer repo loader if available; otherwise fall back to datasets.load_dataset."""
    try:
        from rlhf.data import load_helpsteer3_preference

        return list(load_helpsteer3_preference(split))
    except Exception as exc:
        print(f"[warn] repo loader failed for split={split}: {exc}")
        print("[warn] falling back to datasets.load_dataset")

    from datasets import load_dataset

    # HelpSteer3 has changed loader behavior across datasets versions, so keep fallbacks.
    attempts = [
        lambda: load_dataset("nvidia/HelpSteer3", "preference", split=split),
        lambda: load_dataset("nvidia/HelpSteer3", split=split),
    ]

    last_exc = None
    for attempt in attempts:
        try:
            return list(attempt())
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Could not load HelpSteer3 preference split={split}") from last_exc


def normalize_role(role):
    if role is None:
        return "user"
    role = str(role).lower().strip()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "gpt", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return "user"


def normalize_context(context):
    """
    Normalize HelpSteer3 context into chat-template messages.

    We defensively remove trailing assistant messages because for generation/SFT/RM prompt construction
    we want context up to the user query, then append chosen/rejected as the target assistant answer.
    """
    messages = []

    if context is None:
        return messages

    if isinstance(context, str):
        text = context.strip()
        if text:
            messages.append({"role": "user", "content": text})
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, dict):
                role = normalize_role(item.get("role") or item.get("from"))
                content = item.get("content")
                if content is None:
                    content = item.get("value", "")
                content = str(content).strip()
                if content:
                    messages.append({"role": role, "content": content})
            else:
                content = str(item).strip()
                if content:
                    messages.append({"role": "user", "content": content})
    else:
        text = str(context).strip()
        if text:
            messages.append({"role": "user", "content": text})

    # Remove trailing assistant turns. The candidate response will be appended separately.
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()

    # If no system message exists, Qwen tokenizer's chat template will inject its default system prompt.
    return messages

def get_preference_score(row):
    for key in [
        "preference_score",
        "overall_preference_score",
        "overall_preference",
        "preference",
        "score",
    ]:
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except Exception:
                pass
    return None

def get_responses(row):
    response1 = None
    response2 = None

    for key in ["response1", "response_1", "answer1", "answer_1", "output1", "output_1"]:
        if key in row and row[key] is not None:
            response1 = row[key]
            break

    for key in ["response2", "response_2", "answer2", "answer_2", "output2", "output_2"]:
        if key in row and row[key] is not None:
            response2 = row[key]
            break

    response1 = "" if response1 is None else str(response1)
    response2 = "" if response2 is None else str(response2)
    return response1, response2

def chosen_rejected_from_row(row):
    score = get_preference_score(row)
    r1, r2 = get_responses(row)

    if score is None:
        return None

    # HelpSteer3 preference convention:
    # negative => response1 preferred; positive => response2 preferred; zero => tie.
    if score < 0:
        return r1, r2, score
    if score > 0:
        return r2, r1, score
    return None


def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def render_full(tokenizer, messages, response):
    full_messages = list(messages) + [{"role": "assistant", "content": response}]
    return tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def token_count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def compute_limit_stats(row, limits):
    """
    Given a length row, compute truncation and lost-response-token stats for each limit.
    """
    out = {}

    prompt_tokens = row["prompt_tokens"]
    chosen_full = row["chosen_full_tokens"]
    rejected_full = row["rejected_full_tokens"]
    chosen_resp = row["chosen_response_tokens_est"]
    rejected_resp = row["rejected_response_tokens_est"]

    for limit in limits:
        key = str(limit)

        chosen_visible_resp = max(0, min(chosen_resp, limit - prompt_tokens))
        rejected_visible_resp = max(0, min(rejected_resp, limit - prompt_tokens))

        out[f"prompt_exceeds_{key}"] = prompt_tokens >= limit
        out[f"sft_chosen_truncated_{key}"] = chosen_full > limit
        out[f"rm_chosen_truncated_{key}"] = chosen_full > limit
        out[f"rm_rejected_truncated_{key}"] = rejected_full > limit
        out[f"rm_any_truncated_{key}"] = (chosen_full > limit) or (rejected_full > limit)

        out[f"sft_chosen_response_tokens_visible_{key}"] = chosen_visible_resp
        out[f"sft_chosen_response_tokens_lost_{key}"] = max(0, chosen_resp - chosen_visible_resp)

        out[f"rm_chosen_response_tokens_visible_{key}"] = chosen_visible_resp
        out[f"rm_chosen_response_tokens_lost_{key}"] = max(0, chosen_resp - chosen_visible_resp)
        out[f"rm_rejected_response_tokens_visible_{key}"] = rejected_visible_resp
        out[f"rm_rejected_response_tokens_lost_{key}"] = max(0, rejected_resp - rejected_visible_resp)

        out[f"sft_no_response_tokens_visible_{key}"] = chosen_visible_resp <= 0
        out[f"rm_chosen_no_response_tokens_visible_{key}"] = chosen_visible_resp <= 0
        out[f"rm_rejected_no_response_tokens_visible_{key}"] = rejected_visible_resp <= 0

    return out


def summarize(df, limits, group_cols=None):
    if group_cols is None:
        group_cols = []

    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    for group_key, g in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        base = {}
        for col, val in zip(group_cols, group_key):
            base[col] = val

        base["n"] = int(len(g))
        base["prompt_tokens_mean"] = float(g["prompt_tokens"].mean())
        base["prompt_tokens_p50"] = float(g["prompt_tokens"].quantile(0.50))
        base["prompt_tokens_p90"] = float(g["prompt_tokens"].quantile(0.90))
        base["chosen_full_tokens_p50"] = float(g["chosen_full_tokens"].quantile(0.50))
        base["chosen_full_tokens_p90"] = float(g["chosen_full_tokens"].quantile(0.90))
        base["chosen_full_tokens_p95"] = float(g["chosen_full_tokens"].quantile(0.95))
        base["chosen_full_tokens_p99"] = float(g["chosen_full_tokens"].quantile(0.99))
        base["rejected_full_tokens_p50"] = float(g["rejected_full_tokens"].quantile(0.50))
        base["rejected_full_tokens_p90"] = float(g["rejected_full_tokens"].quantile(0.90))
        base["rejected_full_tokens_p95"] = float(g["rejected_full_tokens"].quantile(0.95))
        base["rejected_full_tokens_p99"] = float(g["rejected_full_tokens"].quantile(0.99))

        total_chosen_resp = max(float(g["chosen_response_tokens_est"].sum()), 1.0)
        total_rejected_resp = max(float(g["rejected_response_tokens_est"].sum()), 1.0)

        for limit in limits:
            key = str(limit)
            base[f"prompt_exceeds_rate_{key}"] = float(g[f"prompt_exceeds_{key}"].mean())
            base[f"sft_chosen_trunc_rate_{key}"] = float(g[f"sft_chosen_truncated_{key}"].mean())
            base[f"rm_chosen_trunc_rate_{key}"] = float(g[f"rm_chosen_truncated_{key}"].mean())
            base[f"rm_rejected_trunc_rate_{key}"] = float(g[f"rm_rejected_truncated_{key}"].mean())
            base[f"rm_any_trunc_rate_{key}"] = float(g[f"rm_any_truncated_{key}"].mean())

            base[f"sft_no_response_visible_rate_{key}"] = float(g[f"sft_no_response_tokens_visible_{key}"].mean())
            base[f"rm_rejected_no_response_visible_rate_{key}"] = float(
                g[f"rm_rejected_no_response_tokens_visible_{key}"].mean()
            )

            base[f"sft_chosen_response_token_loss_rate_{key}"] = float(
                g[f"sft_chosen_response_tokens_lost_{key}"].sum() / total_chosen_resp
            )
            base[f"rm_chosen_response_token_loss_rate_{key}"] = float(
                g[f"rm_chosen_response_tokens_lost_{key}"].sum() / total_chosen_resp
            )
            base[f"rm_rejected_response_token_loss_rate_{key}"] = float(
                g[f"rm_rejected_response_tokens_lost_{key}"].sum() / total_rejected_resp
            )

        rows.append(base)

    return pd.DataFrame(rows)


def write_markdown(summary_overall, summary_by_domain, output_path, limits):
    def pct(x):
        return f"{100.0 * x:.2f}%"

    lines = []
    lines.append("# HelpSteer3 token-length diagnostics")
    lines.append("")
    lines.append("## Overall")
    lines.append("")

    for _, row in summary_overall.iterrows():
        split = row["split"]
        lines.append(f"### Split: `{split}`")
        lines.append("")
        lines.append(f"- Rows: `{int(row['n'])}`")
        lines.append(f"- Prompt tokens p50/p90: `{row['prompt_tokens_p50']:.1f}` / `{row['prompt_tokens_p90']:.1f}`")
        lines.append(
            f"- Chosen full tokens p50/p90/p95/p99: "
            f"`{row['chosen_full_tokens_p50']:.1f}` / `{row['chosen_full_tokens_p90']:.1f}` / "
            f"`{row['chosen_full_tokens_p95']:.1f}` / `{row['chosen_full_tokens_p99']:.1f}`"
        )
        lines.append("")
        lines.append("| limit | SFT chosen trunc | RM any trunc | SFT chosen resp-token loss | RM rejected resp-token loss | prompt exceeds |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for limit in limits:
            key = str(limit)
            lines.append(
                f"| {limit} | "
                f"{pct(row[f'sft_chosen_trunc_rate_{key}'])} | "
                f"{pct(row[f'rm_any_trunc_rate_{key}'])} | "
                f"{pct(row[f'sft_chosen_response_token_loss_rate_{key}'])} | "
                f"{pct(row[f'rm_rejected_response_token_loss_rate_{key}'])} | "
                f"{pct(row[f'prompt_exceeds_rate_{key}'])} |"
            )
        lines.append("")

    lines.append("## Domain-wise summary")
    lines.append("")
    for split in sorted(summary_by_domain["split"].unique()):
        lines.append(f"### Split: `{split}`")
        lines.append("")
        sub = summary_by_domain[summary_by_domain["split"] == split].copy()
        for limit in limits:
            key = str(limit)
            lines.append(f"#### Limit `{limit}`")
            lines.append("")
            lines.append("| domain | n | SFT chosen trunc | RM any trunc | SFT response-token loss | RM rejected response-token loss |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for _, row in sub.sort_values("domain").iterrows():
                lines.append(
                    f"| {row['domain']} | {int(row['n'])} | "
                    f"{pct(row[f'sft_chosen_trunc_rate_{key}'])} | "
                    f"{pct(row[f'rm_any_trunc_rate_{key}'])} | "
                    f"{pct(row[f'sft_chosen_response_token_loss_rate_{key}'])} | "
                    f"{pct(row[f'rm_rejected_response_token_loss_rate_{key}'])} |"
                )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--limits", nargs="+", type=int, default=[1024, 2048, 3072, 4096])
    parser.add_argument("--output-dir", default="outputs/rlhf/length_diagnostics")
    parser.add_argument("--max-examples-per-split", type=int, default=None)
    parser.add_argument("--save-per-example", action="store_true", help="Save per-example CSV. This can be large.")
    args = parser.parse_args()

    global pd, tqdm
    import pandas as pd
    from tqdm.auto import tqdm

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_rows = []
    skipped = defaultdict(int)

    for split in args.splits:
        raw = load_preference_split(split)
        if args.max_examples_per_split is not None:
            raw = raw[: args.max_examples_per_split]

        for idx, row in enumerate(tqdm(raw, desc=f"tokenizing {split}")):
            pair = chosen_rejected_from_row(row)
            if pair is None:
                skipped[(split, "tie_or_invalid_score")] += 1
                continue

            chosen, rejected, score = pair
            if not chosen.strip() or not rejected.strip():
                skipped[(split, "empty_response")] += 1
                continue

            messages = normalize_context(row.get("context"))
            if not messages:
                skipped[(split, "empty_context")] += 1
                continue

            try:
                prompt_text = render_prompt(tokenizer, messages)
                chosen_full_text = render_full(tokenizer, messages, chosen)
                rejected_full_text = render_full(tokenizer, messages, rejected)

                prompt_tokens = token_count(tokenizer, prompt_text)
                chosen_full_tokens = token_count(tokenizer, chosen_full_text)
                rejected_full_tokens = token_count(tokenizer, rejected_full_text)

                chosen_response_tokens_est = max(0, chosen_full_tokens - prompt_tokens)
                rejected_response_tokens_est = max(0, rejected_full_tokens - prompt_tokens)

                rec = {
                    "split": split,
                    "idx": idx,
                    "domain": row.get("domain", "unknown") or "unknown",
                    "language": row.get("language", "unknown") or "unknown",
                    "preference_score": score,
                    "prompt_chars": len(prompt_text),
                    "chosen_full_chars": len(chosen_full_text),
                    "rejected_full_chars": len(rejected_full_text),
                    "chosen_response_chars": len(chosen),
                    "rejected_response_chars": len(rejected),
                    "prompt_tokens": prompt_tokens,
                    "chosen_full_tokens": chosen_full_tokens,
                    "rejected_full_tokens": rejected_full_tokens,
                    "chosen_response_tokens_est": chosen_response_tokens_est,
                    "rejected_response_tokens_est": rejected_response_tokens_est,
                }
                rec.update(compute_limit_stats(rec, args.limits))
                all_rows.append(rec)
            except Exception as exc:
                skipped[(split, f"error:{type(exc).__name__}")] += 1

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise RuntimeError("No valid rows were produced. Check dataset loading/schema.")

    summary_overall = summarize(df, args.limits, group_cols=["split"])
    summary_by_domain = summarize(df, args.limits, group_cols=["split", "domain"])
    summary_by_language = summarize(df, args.limits, group_cols=["split", "language"])

    summary_overall_path = output_dir / "length_summary_overall.csv"
    summary_by_domain_path = output_dir / "length_summary_by_domain.csv"
    summary_by_language_path = output_dir / "length_summary_by_language.csv"
    skipped_path = output_dir / "length_skipped_counts.json"
    md_path = output_dir / "length_diagnostics.md"

    summary_overall.to_csv(summary_overall_path, index=False)
    summary_by_domain.to_csv(summary_by_domain_path, index=False)
    summary_by_language.to_csv(summary_by_language_path, index=False)
    skipped_json = {f"{k[0]}::{k[1]}": v for k, v in skipped.items()}
    skipped_path.write_text(json.dumps(skipped_json, indent=2), encoding="utf-8")

    if args.save_per_example:
        df.to_csv(output_dir / "lengths_per_example.csv", index=False)

    write_markdown(summary_overall, summary_by_domain, md_path, args.limits)

    print("\nSaved:")
    print(" ", summary_overall_path)
    print(" ", summary_by_domain_path)
    print(" ", summary_by_language_path)
    print(" ", skipped_path)
    print(" ", md_path)
    if args.save_per_example:
        print(" ", output_dir / "lengths_per_example.csv")

    print("\nOverall summary:")
    display_cols = ["split", "n"]
    for limit in args.limits:
        key = str(limit)
        display_cols.extend(
            [
                f"sft_chosen_trunc_rate_{key}",
                f"rm_any_trunc_rate_{key}",
                f"sft_chosen_response_token_loss_rate_{key}",
            ]
        )
    print(summary_overall[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
