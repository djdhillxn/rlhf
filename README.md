# RLHF Post-Training of Qwen2.5-0.5B on Heterogeneous, Long-Form HelpSteer3 Data

> Can a complete RLHF pipeline remain measurable and inspectable when the policy and compute budget are small, but the preference task is broad and often long-form?

This repository follows that question through supervised fine-tuning, pairwise reward modeling, guarded PPO, and a full Base/SFT/PPO evaluation. The model is **Qwen2.5-0.5B-Instruct** and the preference source is **NVIDIA HelpSteer3**. HelpSteer3 contributes 40,476 records across general, code, STEM, and multilingual tasks, including long conversations, code, and multi-step explanations.

A sequence-length audit shaped the experiment. A 1,024-token total budget would truncate **38.47%** of SFT examples and **40.82%** of reward-model pairs. At 4,096 tokens, those rates fall to **0.83%** and **1.00%**. The final policy stage then trains and evaluates responses up to **768 new tokens**, making stopping behavior and reward reliability part of the method rather than afterthoughts.

The final result is deliberately mixed. The guarded PPO run completes **12,032 on-policy rollouts and 188 updates** without empty-output or numerical collapse, and its stopping/repetition profile remains close enough to the baselines for meaningful qualitative comparison. It does **not** beat the already instruction-tuned Base model under the learned reward proxy: PPO wins **40.51%** of Base comparisons and **48.29%** of SFT comparisons on all 2,017 validation prompts. The project is therefore presented as an end-to-end RLHF systems, stabilization, and evaluation study—not as a claim that PPO universally improves Qwen.

## Pipeline

1. **Supervised fine-tuning:** response-only LoRA SFT on the human-preferred HelpSteer3 response.
2. **Reward modeling:** an SFT-initialized scalar model trained with Bradley–Terry pairwise ranking loss.
3. **Guarded PPO:** a trainable SFT policy, frozen SFT reference, RM-initialized critic, KL-regularized token-level PPO, and reward/stopping safeguards.
4. **Policy-suite evaluation:** one shared table of Base, SFT, and PPO generations for all 2,017 validation prompts, followed by reward scoring, stopping/repetition diagnostics, and qualitative curation.

## Final Training Profile

### Data and model

| Item | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Preference dataset | NVIDIA HelpSteer3 |
| Filtered SFT/RM training pairs | 36,264 |
| Reward-model validation pairs | 1,917 |
| Policy-suite validation prompts | 2,017 |
| SFT/RM maximum total length | 4,096 tokens |
| PPO/evaluation prompt limit | 3,072 tokens |
| PPO/evaluation response limit | 768 new tokens |

### Supervised fine-tuning

The preferred response is trained with prompt tokens masked from the loss. The merged SFT model is both a comparison policy and the initialization/reference point for later stages.

| Setting / metric | Value |
|---|---:|
| Backend | TRL `SFTTrainer` |
| LoRA rank / alpha | 16 / 32 |
| Epochs | 1 |
| Effective batch size | 32 |
| Learning rate | `5e-6` |
| Train / validation loss | 1.0556 / 1.1127 |
| Validation mean token accuracy | 72.02% |

### Reward model

The reward model starts from the merged SFT backbone and trains a scalar head so the chosen response scores above the rejected response.

| Metric | Value |
|---|---:|
| Backend | TRL `RewardTrainer` |
| LoRA rank / alpha | 32 / 64 |
| Total epochs | 2 |
| Effective batch size | 64 |
| Validation pairs | 1,917 |
| Pairwise accuracy | 65.62% |
| Validation loss | 0.6166 |
| Mean chosen-minus-rejected margin | 0.3578 |

Domain accuracy is 70.23% on code, 62.91% on general, 59.26% on STEM, and 71.82% on multilingual examples. The scalar output is a learned preference proxy, not a universal quality score; pairwise margins and direct response inspection matter more than the sign of one reward.

### Guarded PPO

The final run keeps 768-token rollouts and four PPO epochs per batch, but adds safeguards motivated by earlier reward-hacking and stopping failures:

- **domain-balanced rollouts:** every batch of 64 contains 16 code, 16 general, 16 STEM, and 16 multilingual prompts;
- **calibrated reward bounds:** terminal scores are clipped to the 0.5th and 99.5th percentiles of 4,096 stratified reward-training pairs (`-2.7507` to `1.5018`);
- **EOS-aware scoring:** a response missing EOS receives the calibrated lower-bound reward rather than an arbitrary small penalty;
- **repetition shaping:** a smooth penalty begins only above the preferred-response 95th-percentile repeated-token 4-gram fraction (`0.3084`);
- **conservative policy movement:** KL coefficient `0.10`, clipped policy/value objectives, a frozen SFT reference, and an RM-initialized critic;
- **exact resume:** every 25 updates, policy, value model, optimizer, scheduler, Trainer state, RNG state, rollout position, and guardrail fingerprints are checkpointed together.

| Setting / final metric | Value |
|---|---:|
| Completed rollouts / updates | 12,032 / 188 |
| Approximately unique rollout prompts | 12,007 |
| PPO rollout batch / epochs | 64 / 4 |
| Learning rate | `3e-6`, linear decay |
| Final objective KL to SFT reference | 0.7305 |
| Final objective KL per response token | 0.00199 |
| Final old-to-new approximate KL | 0.000537 |
| Final clip fraction | 0.00571 |
| Final value loss | 0.1076 |
| Final rollout EOS / cap rate | 82.81% / 17.19% |
| Empty-response rate | 0% |

#### Long-response memory path

Stock experimental TRL retained a dense `64 × 768 × vocabulary` generation-score tensor—about **27.77 GiB** in float32—before the PPO update. The final path instead generates token IDs with a rollout-only KV cache, requests no generation-score history, and recomputes sampled-token behavior log-probabilities in bounded chunks from response positions only. It also drops prompt columns masked for every row, shares the reference backbone by disabling the policy adapter, and backpropagates policy and value losses through separate graphs before one optimizer step. These are execution changes, not changes to rewards, GAE, clipping, domain balance, or effective batch size. Peak allocated memory in the completed run was **37.64 GiB**.

## Full Policy-Suite Evaluation

Base, SFT, and PPO each generate one sampled response for every validation prompt using the same tokenizer, 3,072-token prompt limit, 768-token response limit, temperature 0.7, and top-p 1.0. The same two-epoch reward model scores every prompt-response pair. Evaluation is resumable per policy and all pairwise comparisons derive from one shared response table.

### Overall behavior and three-way winners

Behavioral diagnostics are shown before reward/win statistics because reward alone is not an adequate account of policy quality.

| Policy | Mean tokens | Median tokens | Cap-hit rate | Repeated 4-grams >25% | Repeated 4-grams >50% | Mean reward | Three-way wins | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 373.5 | 333 | 16.21% | 165 (8.18%) | 43 (2.13%) | 0.2229 | 845 | 41.89% |
| SFT | 392.5 | 364 | 18.34% | 374 (18.54%) | 143 (7.09%) | 0.2037 | 585 | 29.00% |
| PPO | 384.1 | 355 | 16.31% | 326 (16.16%) | 105 (5.21%) | 0.1753 | 563 | 27.91% |
| Tie | — | — | — | — | — | — | 24 | 1.19% |

### Pairwise reward-model comparisons

| Comparison | Left wins | Right wins | Ties | Right win rate | Mean right-minus-left reward delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 1,145 | 849 | 23 | 42.09% | -0.0192 |
| Base vs PPO | 1,179 | 817 | 21 | 40.51% | -0.0476 |
| SFT vs PPO | 1,018 | 974 | 25 | 48.29% | -0.0284 |

PPO is closest to SFT and wins the SFT comparison on the STEM subset, but it does not beat Base in aggregate or within any of the four domains. The mean differences are small; the win counts still favor Base. This is a mixed alignment outcome rather than a proxy-judge victory.

### Qualitative audit

The automated audit uses reward margins, token counts, EOS/cap behavior, repeated word-level 4-grams, and sensitive-term checks to create review queues. It identifies 9 qualified PPO candidates, 120 strong PPO-loss candidates, 326 repetition-risk rows, and 155 high-reward repetition mismatches. These queues overlap and are navigation aids, not human or LLM-as-judge verdicts.

Manual inspection finds both useful local changes and real failures. Positive cases include clearer structure, better coverage of multi-part requests, and more compact responses. Negative cases include incorrect code, confused factual reasoning, prompt restatement, fabricated details, and occasional extreme loops that still receive high reward. The response explorer exposes the full prompt, Base and PPO text, learned rewards, token counts, EOS/cap state, and repetition diagnostics so aggregate metrics can be checked against actual generations.

## Interpretation

The guarded run answers a narrower and more defensible question than “did PPO beat Qwen?” It shows that long-form PPO can be executed stably on a small instruction model while explicitly controlling reward scale, EOS behavior, repetition, domain balance, KL drift, memory pressure, and interrupted notebook runs. The policy changes behavior and produces useful examples, but the 65.62%-accurate reward model remains an imperfect training signal and an imperfect judge.

Low update KL, small clip fractions, finite value loss, high EOS rate, and zero empty responses establish that the optimizer behaved as intended. They do not establish that the scalar reward captured every human preference. The final conclusion therefore combines three forms of evidence: training diagnostics, behavioral metrics, and direct response review.

Historical custom loops, the earlier higher-reward but more exploitable PPO checkpoint, memory failures, token-budget experiments, and the implemented-but-unexecuted DPO extension are preserved in the [technical companion](docs/rlhf_technical_companion.pdf). They are intentionally absent from this headline result.

## Reports and Artifacts

1. [`docs/rlhf_project_report.pdf`](docs/rlhf_project_report.pdf) — concise academic account of the final SFT, reward-model, guarded-PPO, and evaluation results.
2. [`docs/rlhf_technical_companion.pdf`](docs/rlhf_technical_companion.pdf) — full implementation record, experimental progression, earlier PPO results, GPU/memory redesign, operational contracts, artifact map, DPO design, and future-work agenda.
3. [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md) — setup, commands, repository structure, configuration conventions, checkpoint/resume behavior, and run-location guide.
4. [`notebooks/rlhf_ppo_guarded_colab_pipeline.ipynb`](notebooks/rlhf_ppo_guarded_colab_pipeline.ipynb) — executed guarded PPO workflow.
5. [`notebooks/rlhf_full_eval_and_curation.ipynb`](notebooks/rlhf_full_eval_and_curation.ipynb) — full-suite analysis and response curation.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

python -m scripts.rlhf_trl_prepare_data \
  --config configs/trl/qwen25_05b_helpsteer3_sft.yaml

python -m scripts.rlhf_trl_train_sft \
  --config configs/trl/qwen25_05b_helpsteer3_sft.yaml

python -m scripts.rlhf_trl_train_reward_model \
  --config configs/trl/qwen25_05b_helpsteer3_reward.yaml

python -m scripts.rlhf_trl_train_ppo \
  --config configs/trl/qwen25_05b_helpsteer3_ppo.yaml

python -m scripts.rlhf_evaluate_policy_suite \
  --config configs/trl/qwen25_05b_helpsteer3_eval_suite.yaml
```

For checkpoint conventions, Colab/Drive synchronization, configuration overrides, DPO commands, and the full directory map, use the [repository guide](docs/REPOSITORY_GUIDE.md).
