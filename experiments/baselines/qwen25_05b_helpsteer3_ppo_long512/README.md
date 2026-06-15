# Long-512 PPO Baseline

This directory freezes the resolved configurations and outcome of the final
pre-restructure run. It is the reference point for future RLHF experiments, not
a claim that PPO improved the base model overall.

## Lineage

| Stage | Setting |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| SFT | HelpSteer3, 4096 total tokens, 2 epochs, LoRA rank 16 |
| Reward model | HelpSteer3 pairs, 4096 total tokens, 2 epochs |
| PPO | 3072 prompt tokens, 512 response tokens, 400 requested updates |
| Primary evaluation | 2017 validation prompts, 1024 response tokens |

The SFT run completed 4030 optimizer steps. The final reward model reached
71.62% accuracy on 1917 validation pairs. PPO produced 397 logged updates and
ended with a KL coefficient of 0.14.

## PPO Ingredients

Advantages were normalized over valid response tokens; scalar reward whitening
was not implemented. Rewards were clipped to `[-5, 5]`, a `0.002` length
penalty was used, the missing-EOS penalty was zero, and responses shorter than
32 tokens received a `0.5` penalty. Group-relative reward normalization was
disabled. Generation used a repetition penalty of `1.05` and blocked repeated
4-grams.

Prompt batches were left-padded for generation. The tokenizer EOS token served
as the pad token when no pad token was defined, while explicit response-length
masks prevented padded positions from entering PPO losses.

## Outcome

At the primary 1024-token evaluation, PPO won 38.92% of comparisons with Base
and had a mean reward delta of `-0.2137`. Against SFT, PPO won 44.22% with a
small positive mean reward delta of `+0.0343`. Its cap-hit rate was 11.60%,
compared with roughly 30% in the archived 512-token evaluation.

The run demonstrated a complete and stable pipeline, but not aggregate policy
improvement. Repetition, weak stopping behavior, reward-model mismatches, and a
poorly learned value function remain the central baseline limitations.

Machine-readable details are in [`record.yaml`](record.yaml). The four
`*_config_resolved.yaml` files are copied from the completed output directories
and should not be edited.
