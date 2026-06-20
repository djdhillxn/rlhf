# RLHF Qualitative Audit

Aggregate reward-model win rates are not enough to evaluate this project. The final TRL PPO policy beats the base model by a narrow learned-reward margin, but it also produces longer responses, more cap hits, and more repetition. This audit records both sides.

Final evaluation directory:

```text
rlhf_runs/checkpoints_ckpt100_full/
```

Final curation artifacts:

- `rlhf_runs/ckpt100_deep_curation_report.md`
- `rlhf_runs/ckpt100_deep_curation.ipynb`
- `rlhf_runs/portfolio_curated_policy_comparisons_ckpt100.json`

## Audit Method

Every Base, SFT, and PPO response was checked for response length, cap hits, empty outputs, reward margins, EOS behavior, repeated word-level 4-grams, and a small lexical list of sensitive terms. Candidate tables were produced for low-repetition PPO wins, strong PPO losses, high repetition risk, and high-reward mismatch risk.

The repetition metric is intentionally simple. A technical answer can legitimately repeat terminology, while a pathological loop may evade a word-level metric by changing punctuation or one token at a time. The lexical scan is also not a safety classifier. These checks narrow the review surface; they do not replace human judgment.

## Full-Suite Results

| Policy | Wins | Win rate | Mean reward | Median tokens | Cap-hit rate |
|---|---:|---:|---:|---:|---:|
| Base | 718 | 35.60% | 0.0803 | 331 | 8.82% |
| SFT | 508 | 25.19% | 0.0652 | 371 | 13.39% |
| PPO | 775 | 38.42% | 0.7300 | 520 | 27.42% |
| Tie | 16 | 0.79% | - | - | - |

Pairwise learned-reward results:

| Comparison | PPO/right wins | Other/left wins | Ties | Right win rate | Mean right-left delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 837 | 1158 | 22 | 41.50% | -0.0151 |
| Base vs PPO | 1027 | 981 | 9 | 50.92% | +0.6497 |
| SFT vs PPO | 1164 | 840 | 13 | 57.71% | +0.6648 |

Domain-level PPO results against Base:

| Domain | PPO wins | Base wins | Ties | PPO win rate |
|---|---:|---:|---:|---:|
| Code | 188 | 250 | 0 | 42.92% |
| General | 529 | 399 | 3 | 56.82% |
| STEM | 118 | 126 | 1 | 48.16% |
| Multilingual | 192 | 206 | 5 | 47.64% |

## Repetition And Stopping

| Policy | Repeated 4-grams >25% | Repeated 4-grams >50% | Sensitive-term hits |
|---|---:|---:|---:|
| Base | 204 (10.11%) | 61 (3.02%) | 2 |
| SFT | 405 (20.08%) | 171 (8.48%) | 5 |
| PPO | 643 (31.88%) | 319 (15.82%) | 3 |

The final PPO policy is the strongest by learned reward, but it is also the most prone to long, capped, or repetitive continuations. This is the main caveat in the final result. The reward model sometimes gives high scores to visibly repetitive answers, so reward-model wins must be inspected before being described as qualitative wins.

## Candidate Pools

| Pool | Count | Interpretation |
|---|---:|---|
| Qualified PPO candidates | 8 | PPO beats both baselines, avoids cap/EOS issues, and stays below the repetition threshold. |
| Strong PPO losses | 108 | PPO loses sharply to at least one baseline. |
| PPO repetition risks | 643 | PPO exceeds the 25% repeated 4-gram threshold. |
| Reward-model mismatch risks | 313 | High reward co-occurs with repetition or related warning signs. |

The 50-example portfolio export is intentionally balanced: 25 positive examples and 25 negative examples. The positive side includes clean and modest PPO wins; the negative side includes reward-model false positives, severe repetition, code failures, multilingual failures, and strong regressions.

## Interpretation

This run is the most usable PPO result from the project. It shows that the TRL pipeline, N+ implementation details, reward centering, EOS handling, and PPO training can produce a policy that narrowly beats the base model under the learned judge. It also shows why learned reward is not enough. PPO's higher reward comes with longer responses and substantially more repetition, and several of the largest margins are not trustworthy without manual review.

The correct conclusion is therefore not "PPO solved the task." The correct conclusion is that the full RLHF system is working, the aligned policy has measurable local wins, and the audit exposes exactly where the reward model and stopping behavior need improvement.
