import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median


BASE = "base"
SFT = "sft_4096"
PPO = "ppo_4096_ep2_u400"
POLICIES = (BASE, SFT, PPO)
SENSITIVE_TERMS = (
    "asshole",
    "bitch",
    "cunt",
    "fuck",
    "fucking",
    "kill yourself",
    "motherfucker",
    "nigger",
    "porn",
    "rape",
    "shit",
    "suicide",
)


def _load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_user_prompt(rendered_prompt):
    matches = re.findall(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>", str(rendered_prompt), flags=re.S
    )
    if matches:
        return matches[-1].strip()
    return str(rendered_prompt).replace("<|im_start|>assistant\n", "").strip()


def _repetition_metrics(text):
    words = re.findall(r"[\w']+", str(text).lower(), flags=re.UNICODE)
    ngrams = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    counts = Counter(ngrams)
    repeated_fraction = 0.0 if not ngrams else 1.0 - len(counts) / len(ngrams)
    return {
        "word_count": len(words),
        "unique_word_fraction": 0.0 if not words else len(set(words)) / len(words),
        "repeated_4gram_fraction": repeated_fraction,
        "max_4gram_count": max(counts.values(), default=0),
    }


def _sensitive_terms(text):
    lowered = str(text).lower()
    return [
        term
        for term in SENSITIVE_TERMS
        if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", lowered)
    ]


def _enrich(rows):
    for row in rows:
        row["user_prompt"] = _extract_user_prompt(row.get("prompt", ""))
        for policy in POLICIES:
            prefix = f"{policy}_"
            metrics = _repetition_metrics(row.get(f"{policy}_response", ""))
            for key, value in metrics.items():
                row[prefix + key] = value
            row[prefix + "sensitive_terms"] = ",".join(
                _sensitive_terms(row.get(f"{policy}_response", ""))
            )


def _write_csv(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percent(value):
    return f"{100.0 * value:.2f}%"


def _policy_stats(rows, policy):
    rewards = [float(row[f"{policy}_reward"]) for row in rows]
    tokens = [int(row[f"{policy}_response_tokens"]) for row in rows]
    repetitions = [float(row[f"{policy}_repeated_4gram_fraction"]) for row in rows]
    return {
        "mean_reward": mean(rewards),
        "median_reward": median(rewards),
        "mean_tokens": mean(tokens),
        "median_tokens": median(tokens),
        "cap_hits": sum(bool(row[f"{policy}_cap_hit"]) for row in rows),
        "cap_hit_rate": mean(bool(row[f"{policy}_cap_hit"]) for row in rows),
        "heavy_repetition": sum(value > 0.25 for value in repetitions),
        "heavy_repetition_rate": mean(value > 0.25 for value in repetitions),
        "severe_repetition": sum(value > 0.50 for value in repetitions),
        "severe_repetition_rate": mean(value > 0.50 for value in repetitions),
        "sensitive_term_hits": sum(
            bool(row[f"{policy}_sensitive_terms"]) for row in rows
        ),
    }


def _delta_stats(rows, column):
    values = [float(row[column]) for row in rows]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    return {
        "mean": mean(values),
        "median": median(values),
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(values) - len(wins) - len(losses),
        "mean_win": mean(wins) if wins else 0.0,
        "mean_loss": mean(losses) if losses else 0.0,
        "sum_wins": sum(wins),
        "sum_losses": sum(losses),
    }


def _comparison_rows(current, baseline):
    current_by_idx = {int(row["idx"]): row for row in current}
    baseline_by_idx = {int(row["idx"]): row for row in baseline}
    if current_by_idx.keys() != baseline_by_idx.keys():
        raise ValueError(
            "Current and baseline evaluations do not contain the same example indices."
        )

    comparisons = []
    for policy in POLICIES:
        changed = 0
        old_uncapped_changed = 0
        old_capped_changed = 0
        reward_changes = []
        token_changes = []
        for idx, current_row in current_by_idx.items():
            baseline_row = baseline_by_idx[idx]
            response_changed = (
                current_row[f"{policy}_response"] != baseline_row[f"{policy}_response"]
            )
            changed += response_changed
            if bool(baseline_row[f"{policy}_cap_hit"]):
                old_capped_changed += response_changed
            else:
                old_uncapped_changed += response_changed
            reward_changes.append(
                float(current_row[f"{policy}_reward"])
                - float(baseline_row[f"{policy}_reward"])
            )
            token_changes.append(
                int(current_row[f"{policy}_response_tokens"])
                - int(baseline_row[f"{policy}_response_tokens"])
            )
        comparisons.append(
            {
                "policy": policy,
                "baseline_cap_hits": sum(
                    bool(row[f"{policy}_cap_hit"]) for row in baseline
                ),
                "current_cap_hits": sum(
                    bool(row[f"{policy}_cap_hit"]) for row in current
                ),
                "changed_responses": changed,
                "baseline_uncapped_changed": old_uncapped_changed,
                "baseline_capped_changed": old_capped_changed,
                "mean_reward_change": mean(reward_changes),
                "median_reward_change": median(reward_changes),
                "mean_token_change": mean(token_changes),
                "median_token_change": median(token_changes),
            }
        )
    return comparisons


def _candidate_fields():
    ppo_base_delta = f"delta_{PPO}_minus_{BASE}"
    ppo_sft_delta = f"delta_{PPO}_minus_{SFT}"
    fields = [
        "idx",
        "domain",
        "language",
        "winner",
        "reward_rank",
        "user_prompt",
        f"delta_{SFT}_minus_{BASE}",
        ppo_base_delta,
        ppo_sft_delta,
    ]
    for policy in POLICIES:
        fields.extend(
            [
                f"{policy}_reward",
                f"{policy}_response_tokens",
                f"{policy}_cap_hit",
                f"{policy}_repeated_4gram_fraction",
                f"{policy}_max_4gram_count",
                f"{policy}_sensitive_terms",
            ]
        )
    return fields


def _markdown_table(rows, columns, limit=20):
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(columns))) + "|",
    ]
    for row in rows[:limit]:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|").replace("\n", " ")[:160])
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    rows,
    output_path,
    comparisons,
    qualified,
    losses,
    repetition_risks,
    reward_mismatches,
):
    policy_stats = {policy: _policy_stats(rows, policy) for policy in POLICIES}
    ppo_base_delta = f"delta_{PPO}_minus_{BASE}"
    ppo_sft_delta = f"delta_{PPO}_minus_{SFT}"
    ppo_base = _delta_stats(rows, ppo_base_delta)
    ppo_sft = _delta_stats(rows, ppo_sft_delta)

    lines = [
        "# Automated qualitative audit",
        "",
        "This report scans every policy-suite row. Repetition and lexical checks are diagnostics, not substitutes for manual review.",
        "",
        "## Current evaluation",
        "",
        "| policy | mean reward | median tokens | cap-hit rate | repeated 4-grams > 25% | repeated 4-grams > 50% | sensitive-term hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        stats = policy_stats[policy]
        lines.append(
            f"| {policy} | {stats['mean_reward']:.4f} | {stats['median_tokens']:.1f} | "
            f"{_percent(float(stats['cap_hit_rate']))} | "
            f"{stats['heavy_repetition']} ({_percent(float(stats['heavy_repetition_rate']))}) | "
            f"{stats['severe_repetition']} ({_percent(float(stats['severe_repetition_rate']))}) | "
            f"{stats['sensitive_term_hits']} |"
        )

    lines.extend(
        [
            "",
            "## PPO reward margins",
            "",
            "| comparison | mean delta | median delta | wins | losses | ties | mean winning margin | mean losing margin |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| PPO - Base | {ppo_base['mean']:.4f} | {ppo_base['median']:.4f} | {ppo_base['wins']} | "
            f"{ppo_base['losses']} | {ppo_base['ties']} | {ppo_base['mean_win']:.4f} | {ppo_base['mean_loss']:.4f} |",
            f"| PPO - SFT | {ppo_sft['mean']:.4f} | {ppo_sft['median']:.4f} | {ppo_sft['wins']} | "
            f"{ppo_sft['losses']} | {ppo_sft['ties']} | {ppo_sft['mean_win']:.4f} | {ppo_sft['mean_loss']:.4f} |",
        ]
    )

    if comparisons:
        lines.extend(
            [
                "",
                "## Baseline comparison",
                "",
                "A changed batch size or evaluator implementation can alter greedy generations. Treat this as a run-to-run comparison unless every generation setting is controlled.",
                "",
                "| policy | old cap hits | new cap hits | changed responses | old uncapped responses that changed | mean reward change | mean token change |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in comparisons:
            lines.append(
                f"| {row['policy']} | {row['baseline_cap_hits']} | {row['current_cap_hits']} | "
                f"{row['changed_responses']} | {row['baseline_uncapped_changed']} | "
                f"{row['mean_reward_change']:.4f} | {row['mean_token_change']:.2f} |"
            )

    candidate_columns = [
        ("idx", "idx"),
        ("domain", "domain"),
        ("language", "language"),
        (ppo_base_delta, "PPO-Base"),
        (ppo_sft_delta, "PPO-SFT"),
        (f"{PPO}_reward", "PPO reward"),
        (f"{PPO}_response_tokens", "tokens"),
        (f"{PPO}_repeated_4gram_fraction", "repeat-4"),
    ]
    sections = [
        ("Qualified PPO candidates", qualified),
        ("Strong PPO losses", losses),
        ("Highest PPO repetition risk", repetition_risks),
        ("High-reward repetition mismatches", reward_mismatches),
    ]
    for title, candidates in sections:
        lines.extend(["", f"## {title}", ""])
        lines.extend(_markdown_table(candidates, candidate_columns))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_selected_examples(
    rows,
    selection_file,
    output_path,
):
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    rows_by_idx = {int(row["idx"]): row for row in rows}
    lines = [
        f"# {selection.get('title', 'Selected qualitative examples')}",
        "",
        "The reward scores below come from the learned reward model and should be compared with the response text, not treated as human judgments.",
        "",
    ]
    for item in selection.get("selections", []):
        idx = int(item["idx"])
        if idx not in rows_by_idx:
            raise ValueError(f"Selection index {idx} is not present in the evaluation.")
        row = rows_by_idx[idx]
        lines.extend(
            [
                f"## idx {idx}: {item.get('category', 'manual review')}",
                "",
                f"**Domain:** `{row.get('domain')}`  ",
                f"**Language:** `{row.get('language')}`  ",
                f"**Reward winner:** `{row.get('winner')}`  ",
                f"**Manual note:** {item.get('note', '')}",
                "",
                "### Prompt",
                "",
                str(row.get("user_prompt", "")),
                "",
            ]
        )
        for policy in POLICIES:
            lines.extend(
                [
                    f"### {policy}",
                    "",
                    f"Reward: `{float(row[f'{policy}_reward']):.4f}`; "
                    f"tokens: `{int(row[f'{policy}_response_tokens'])}`; "
                    f"cap hit: `{bool(row[f'{policy}_cap_hit'])}`; "
                    f"repeated 4-gram fraction: `{float(row[f'{policy}_repeated_4gram_fraction']):.4f}`",
                    "",
                    str(row.get(f"{policy}_response", "")),
                    "",
                ]
            )
        lines.extend(["---", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    global BASE, SFT, PPO, POLICIES

    parser = argparse.ArgumentParser(
        description="Audit policy-suite outputs and export qualitative curation candidates."
    )
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--base-label", default=BASE)
    parser.add_argument("--sft-label", default=SFT)
    parser.add_argument("--ppo-label", default=PPO)
    args = parser.parse_args()

    BASE, SFT, PPO = args.base_label, args.sft_label, args.ppo_label
    POLICIES = (BASE, SFT, PPO)

    samples_path = args.eval_dir / "policy_suite_samples.jsonl"
    rows = _load_jsonl(samples_path)
    _enrich(rows)
    ppo_base_delta = f"delta_{PPO}_minus_{BASE}"
    ppo_sft_delta = f"delta_{PPO}_minus_{SFT}"

    qualified = sorted(
        [
            row
            for row in rows
            if row.get("winner") == PPO
            and float(row[ppo_base_delta]) > 2.0
            and float(row[ppo_sft_delta]) > 1.0
            and not bool(row[f"{PPO}_cap_hit"])
            and float(row[f"{PPO}_repeated_4gram_fraction"]) < 0.15
        ],
        key=lambda row: float(row[ppo_base_delta]),
        reverse=True,
    )
    losses = sorted(
        [row for row in rows if float(row[ppo_base_delta]) < -5.0],
        key=lambda row: float(row[ppo_base_delta]),
    )
    repetition_risks = sorted(
        [row for row in rows if float(row[f"{PPO}_repeated_4gram_fraction"]) > 0.25],
        key=lambda row: (
            float(row[f"{PPO}_repeated_4gram_fraction"]),
            int(row[f"{PPO}_max_4gram_count"]),
        ),
        reverse=True,
    )
    reward_mismatches = sorted(
        [
            row
            for row in rows
            if float(row[f"{PPO}_reward"]) > 2.0
            and (
                float(row[f"{PPO}_repeated_4gram_fraction"]) > 0.25
                or int(row[f"{PPO}_max_4gram_count"]) >= 5
            )
        ],
        key=lambda row: float(row[f"{PPO}_reward"]),
        reverse=True,
    )

    fields = _candidate_fields()
    _write_csv(
        qualified, args.eval_dir / "curation_qualified_ppo_candidates.csv", fields
    )
    _write_csv(losses, args.eval_dir / "curation_strong_ppo_losses.csv", fields)
    _write_csv(
        repetition_risks, args.eval_dir / "curation_ppo_repetition_risks.csv", fields
    )
    _write_csv(
        reward_mismatches,
        args.eval_dir / "curation_reward_model_mismatches.csv",
        fields,
    )

    comparisons = None
    if args.baseline_dir is not None:
        baseline = _load_jsonl(args.baseline_dir / "policy_suite_samples.jsonl")
        _enrich(baseline)
        comparisons = _comparison_rows(rows, baseline)
        _write_csv(
            comparisons,
            args.eval_dir / "evaluation_comparison_with_baseline.csv",
            list(comparisons[0].keys()),
        )

    _write_report(
        rows,
        args.eval_dir / "qualitative_audit_auto.md",
        comparisons,
        qualified,
        losses,
        repetition_risks,
        reward_mismatches,
    )
    if args.selection_file is not None:
        _write_selected_examples(
            rows,
            args.selection_file,
            args.eval_dir / "selected_qualitative_examples.md",
        )
    print(f"Audited {len(rows)} examples in {args.eval_dir}")
    print(f"Qualified PPO candidates: {len(qualified)}")
    print(f"Strong PPO losses: {len(losses)}")
    print(f"High-reward repetition mismatches: {len(reward_mismatches)}")


if __name__ == "__main__":
    main()
