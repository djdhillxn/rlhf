# RLHF Curation Guide

The final TRL evaluation contains 2,017 validation prompts and three generated responses per prompt: Base, SFT, and PPO. Curation begins with deterministic filters, then uses human review to decide what the examples actually demonstrate. Reward-model margin alone is not a quality label.

Final evaluation directory:

```text
rlhf_runs/checkpoints_ckpt100_full/
```

Portfolio export:

```text
rlhf_runs/portfolio_curated_policy_comparisons_ckpt100.json
```

That JSON contains 50 curated examples, balanced as 25 positive and 25 negative cases. It is the artifact intended for the interactive portfolio response explorer.

## Reproduce The Audit

Run the automated scan against the final evaluation directory:

```bash
python scripts/rlhf_audit_policy_suite.py \
  --eval-dir rlhf_runs/checkpoints_ckpt100_full \
  --base-label base \
  --sft-label sft_trl \
  --ppo-label ppo_exact_ckpt100
```

The audit writes candidate tables for low-repetition PPO wins, strong PPO losses, repetition risks, and high-reward reward-model mismatches:

| Artifact | Purpose |
|---|---|
| `qualitative_audit_auto.md` | aggregate repetition, margin, and candidate-pool report |
| `curation_qualified_ppo_candidates.csv` | low-repetition PPO wins over both Base and SFT |
| `curation_strong_ppo_losses.csv` | examples where PPO loses strongly |
| `curation_ppo_repetition_risks.csv` | PPO responses with high repeated 4-gram rates |
| `curation_reward_model_mismatches.csv` | high-reward responses with repetition or other warning signs |
| `policy_suite_samples.jsonl` / `.csv` | full prompts and full generated responses |

The final deep-curation pass adds:

| Artifact | Purpose |
|---|---|
| `rlhf_runs/ckpt100_deep_curation.ipynb` | notebook for browsing aggregate tables and individual examples |
| `rlhf_runs/ckpt100_deep_curation_report.md` | compact human-readable curation summary |
| `rlhf_runs/ckpt100_judged_examples.csv` | deterministic judge-style labels for all 2,017 rows |
| `rlhf_runs/ckpt100_deep_curation_pools.json` | candidate pools used to choose portfolio examples |
| `rlhf_runs/portfolio_curated_policy_comparisons_ckpt100.json` | clean 50-example portfolio payload |

## What The Labels Mean

The notebook's `judge_label` values are deterministic diagnostic labels, not manual judgments. They are produced from reward deltas, cap-hit flags, EOS flags, response length, repeated 4-gram rates, and domain/language metadata. They are useful for triage; they do not replace reading the responses.

Examples:

- `likely_genuine_ppo_win`: PPO beats Base and SFT, reaches EOS, avoids the cap, and stays below the repetition threshold.
- `modest_clean_ppo_win`: PPO wins by a smaller margin without obvious stopping problems.
- `reward_model_false_positive_risk`: PPO receives a high reward while repetition or other diagnostics suggest the score may be misleading.
- `severe_repetition_failure`: PPO response is dominated by repeated word-level 4-grams.
- `strong_ppo_regression`: PPO loses sharply to Base or SFT.

## Review Categories

A defensible public report should include the whole spectrum:

| Category | Why it matters |
|---|---|
| Clean PPO wins | Shows where RLHF changed behavior in a useful direction. |
| Modest or near-tie wins | Shows incremental gains without overselling them. |
| Strong PPO losses | Shows capability regressions and domain weaknesses. |
| Repetition failures | Tests stopping and reward hacking. |
| Reward-model mismatches | Shows where scalar reward disagrees with visible quality. |
| Code/STEM failures | Highlights correctness risks that a generic preference reward may miss. |
| Multilingual failures | Tests language consistency and semantic drift. |

The final selected set includes 8 likely genuine PPO wins, 16 modest clean wins, 9 reward-model false-positive risks, 9 severe repetition failures, 7 strong PPO regressions, and 1 manual-review edge case. It deliberately avoids presenting only the most flattering examples.

## Publication Standard

Before presenting an example as an improvement, check instruction following, correctness, completeness, relevance, repetition, safety, and whether the reward advantage reflects substance rather than verbosity or formatting. Technical examples should be executed or independently verified. Scientific claims and citations should be checked against reliable sources.

The appropriate conclusion from the final run is balanced: PPO now edges Base under the learned reward model, but it is longer, more often capped, and more repetitive. Curation should make that mixed result understandable rather than hide the failure modes.
