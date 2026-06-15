# RLHF Curation Guide

The primary evaluation contains 2017 prompts and three generated responses per prompt. Curation should therefore begin with reproducible filters, then use manual review to decide what the examples actually demonstrate. Reward margin alone is not a quality label: several extreme scores in this run belong to repetitive or factually broken responses.

Primary evaluation:

```text
outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/
```

## Reproduce the audit

Run:

```bash
python scripts/rlhf_audit_policy_suite.py \
  --eval-dir outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024 \
  --baseline-dir outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400 \
  --selection-file configs/rlhf/qwen25_05b_helpsteer3_eval1024_curation.json
```

This scans every Base, SFT, and PPO response and writes candidate tables for low-repetition PPO wins, strong PPO losses, repetition risks, reward-model mismatches, and the comparison with the 512-token baseline.

The reviewed manifest generates:

[`selected_qualitative_examples.md`](../outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/selected_qualitative_examples.md)

That document contains the full prompt and all three policy responses, so readers can inspect the evidence without running a notebook.

## Review categories

A defensible report should include several kinds of evidence. Qualified improvements show prompts where PPO or SFT is more useful, direct, or supportive, while preserving caveats about factuality. Base wins reveal capability regression. Repetition and irrelevant continuations test stopping behavior. Factual failures cover fabricated citations, invalid code or equations, and unsupported technical claims. Reward-model mismatches show where numerical preference and visible response quality disagree.

The current reviewed set includes:

| Indices | Purpose |
|---|---|
| 2, 1210, 1158, 0, 418, 1511 | qualified local improvements and near ties |
| 1970, 1303, 176, 1835 | factual, safety, relevance, and stopping failures |
| 346, 1281, 1579, 656, 1548 | high-confidence reward-model mismatches |
| 86 | severe repetition that the reward model correctly penalizes |

The interpretation of these examples is recorded in [`rlhf_qualitative_audit.md`](rlhf_qualitative_audit.md).

## Interactive notebooks

`notebooks/analyzing_full_eval_results.ipynb` provides aggregate tables and candidate views. `notebooks/rlhf_full_eval_and_curation.ipynb` provides an interactive browser and Markdown export. Both now point to the 1024-token evaluation directory and use the updated reviewed indices.

Notebook outputs were cleared after the source paths changed, so rerunning the cells in Colab will populate them entirely from the 1024-token suite.

## Publication standard

Before presenting an example as an improvement, check instruction following, correctness, completeness, relevance, repetition, safety, and whether the reward advantage reflects substance rather than verbosity or formatting. Technical examples should be executed or independently verified. Scientific claims and citations should be checked against reliable sources.

The appropriate conclusion from this run is balanced: PPO produces some useful local changes, but Base remains stronger overall and the longer evaluation reveals significant stopping and reward-model failures. Curation should make that mixed result easier to understand, not hide it.
