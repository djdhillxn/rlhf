# RLHF Post-Training of Qwen2.5-0.5B with HelpSteer3 through SFT, Reward Modeling, and PPO

<!-- This project adapts the trust-region idea behind PPO from classical reinforcement learning to language-model post-training. It grew out of my earlier [TRPO, NPG, and PPO](https://github.com/djdhillxn/trpo) work on MuJoCo and Atari, then asks the same question in an RLHF setting: -->

<!-- Can a small intruction model be supervised, judged, and PPO-aligned in a way that is measurable, reproducible, and honest about failure modes? -->

> Can a complete RLHF pipeline remain measurable and inspectable when the policy and compute budget are small, but the preference task is broad and often long-form?

The tension is not deliberate, it arose from my compute availabilities, but also my curiosity. Qwen2.5-0.5B-Instruct keeps the experiment feasible on rented notebook hardware, while HelpSteer3 supplies 40,476 preference records across general, code, STEM, and multilingual tasks, including code, multi-step explanations, long conversations, and multilingual responses. A length diagnostic showed that a 1,024-token total training budget would truncate 38.47% of SFT examples and 40.82% of reward-model pairs; the final 4,096-token budget reduced those rates to 0.83% and 1.00%. Sequence handling and reward reliability are therefore part of the research problem, not background implementation details.

The completed pipeline uses **Qwen2.5-0.5B-Instruct**, **NVIDIA HelpSteer3**, **Hugging Face TRL**, and LoRA adapters for three training stages plus a final evaluation stage:

1. supervised fine-tuning (SFT) on preferred HelpSteer3 responses;
2. reward-model training on chosen/rejected preference pairs;
3. token-level PPO with a frozen SFT reference and a learned reward model;
4. full policy-suite evaluation of Base, SFT, and PPO responses on the same validation prompts.

A controlled DPO extension is now implemented from the same frozen SFT
checkpoint. It is intentionally described as the next experiment until its
training and full-suite evaluation artifacts exist.

The final PPO policy does not make the 0.5B model universally better. It does produce the strongest run from this project: under the learned reward model, PPO wins **50.92%** of pairwise comparisons against the base instruction model and **57.71%** against the SFT policy on the 2,017-prompt evaluation. The same audit also shows a real cost: PPO is longer, hits the generation cap more often, and has the highest repetition rate. The result is therefore a useful RLHF case study rather than a blanket claim of model superiority.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

The completed PPO workflow is in
[`notebooks/rlhf_trl_colab_pipeline.ipynb`](notebooks/rlhf_trl_colab_pipeline.ipynb).
The restartable DPO workflow is in
[`notebooks/rlhf_dpo_colab_pipeline.ipynb`](notebooks/rlhf_dpo_colab_pipeline.ipynb).
The executed final-run notebook and lightweight exported artifacts are stored
locally under `rlhf_runs/` and `rlhf_runs_lightweight_export/` for analysis.

## Active Pipeline

The active training path uses Hugging Face TRL for SFT, reward modeling, PPO,
and DPO. The repository still owns HelpSteer3 preprocessing, chat formatting,
manifests, evaluation, repetition diagnostics, and qualitative curation.

```bash
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

Every command accepts repeated `--set dotted.path=value` overrides. The final Colab run used overrides for Google Drive/local-SSD paths and PPO hyperparameters; the important behavioral settings are summarized below and preserved in the executed notebook.

## Final TRL Run

### Data And Model

| Item | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Model size | about 0.5B parameters |
| Preference dataset | HelpSteer3 |
| Filtered training rows | 36,264 |
| Filtered validation preference rows | 1,917 |
| Policy-suite validation prompts | 2,017 |
| SFT/RM max total length | 4096 tokens |
| Max prompt length | 3072 tokens |
| Evaluation generation cap | 1024 new tokens |

The Qwen tokenizer chat template is used throughout:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
```

### SFT

The SFT policy trains on the preferred HelpSteer3 response with loss masked off on prompt tokens. It is both a comparison policy and the initialization/reference point for PPO.

| Setting / metric | Value |
|---|---:|
| Backend | TRL `SFTTrainer` |
| LoRA rank / alpha | 16 / 32 |
| Epochs | 1 |
| Effective batch size | 32 |
| Learning rate | `5e-6` |
| Max total length | 4096 |
| Train loss | 1.0556 |
| Eval loss | 1.1127 |
| Eval mean token accuracy | 72.02% |
| Output | `rlhf_runs_lightweight_export/.../full/sft/` |

Objective:

```text
L_SFT(theta) = - sum_t log pi_theta(y_t | x, y_<t)
```

### Reward Model

The reward model starts from the merged SFT model, adds a scalar head, and trains on HelpSteer3 chosen/rejected pairs using the Bradley-Terry logistic ranking loss.

```text
L_RM(phi) = - log sigmoid(r_phi(chosen) - r_phi(rejected))
```

The final reward model was trained for one epoch, then resumed for a second epoch from the saved checkpoint. It uses the N+ implementation detail of initializing from SFT, controlled scalar-head initialization, reward-centering regularization, and a persisted reward offset used by PPO/evaluation.

| Metric | Value |
|---|---:|
| Backend | TRL `RewardTrainer` |
| LoRA rank / alpha | 32 / 64 |
| Total epochs | 2 |
| Effective batch size | 64 |
| Learning rate | `5e-6` |
| Validation preference rows | 1,917 |
| Audit accuracy | 65.62% |
| Eval loss | 0.6166 |
| Mean reward margin | 0.3578 |
| Reward offset | -0.1985 |

Domain accuracy:

| Domain | Accuracy | Count |
|---|---:|---:|
| Code | 70.23% | 430 |
| General | 62.91% | 914 |
| STEM | 59.26% | 243 |
| Multilingual | 71.82% | 330 |

This reward model is useful, but it is not a human judge. Its scalar output is a learned proxy. Margins and rankings matter more than raw sign, and qualitative review remains necessary.

### PPO Alignment

PPO starts from the merged SFT policy, keeps a frozen SFT reference, scores sampled responses with the reward model, and uses a reward-model-initialized value model. The final run followed the highest-impact N+ implementation details: zero dropout, behavior log-probabilities matched to generation temperature, fixed-length generation with EOS handling, an invalid reward for missing EOS, Adam epsilon `1e-5`, reward whitening, an RM-initialized critic, and KL anchoring to the SFT reference.

The executed Colab overrides are the source of truth for the final PPO settings:

| Setting / metric | Value |
|---|---:|
| Backend | TRL experimental PPO trainer |
| Planned episodes | 12,000 |
| Evaluated episodes | 6,400 |
| Optimizer steps evaluated | 100 |
| PPO rollout response length | 768 new tokens |
| PPO epochs per rollout batch | 4 |
| KL coefficient | 0.07 |
| Temperature | 0.7 |
| Missing-EOS reward | -1.0 |
| Reward whitening | enabled |
| Learning rate | `3e-6` |
| Batch / accumulation | 2 per device / 32 accumulation |
| Average objective KL | 1.8278 |
| Final objective KL | 2.1648 |
| Average EOS count | 44.57 / 64 rollout samples |
| Final EOS count | 38 / 64 rollout samples |
| Average reward-model score during PPO | -0.5898 |

The run was intentionally stopped and evaluated after 100 optimizer steps because it had become stable enough to inspect, and longer continuation would have increased cost without guaranteeing better qualitative behavior. Continuing this same training segment with multi-metric checkpoint selection is future work, not part of the final reported result.

### DPO Extension

The DPO experiment changes only the preference-optimization method. It reuses
the frozen one-epoch SFT policy and the same non-tied HelpSteer3 pairs, tokenizer,
4096-token preparation budget, LoRA targets, and final evaluation prompts. The
primary configuration uses standard sigmoid DPO with beta `0.1`, one epoch,
effective batch size 32, LoRA rank 16, zero dropout, and precomputed reference
log probabilities:

```bash
python -m scripts.rlhf_trl_prepare_data \
  --config configs/trl/qwen25_05b_helpsteer3_dpo.yaml

python -m scripts.rlhf_trl_train_dpo \
  --config configs/trl/qwen25_05b_helpsteer3_dpo.yaml
```

HelpSteer3's preference strengths are retained as metadata for stratified
analysis. The main run does not assume that ordinal levels 1, 2, and 3 are
linearly spaced utility weights. A separately named `linear_replication`
sensitivity run is available, but it should not be conflated with the primary
standard-DPO result.

## Policy-Suite Evaluation

The final evaluator generates Base, SFT, and PPO responses for the same 2,017 HelpSteer3 validation prompts, scores every prompt-response pair with the same reward model, and derives all pairwise comparisons from that single table.

Final evaluation settings:

| Setting | Value |
|---|---:|
| Prompt budget | 3072 tokens |
| Generation budget | 1024 new tokens |
| Decoding | sampled, temperature 0.7, top-p 1.0 |
| Eval batch size | 256 |
| Reward model | TRL reward model after two epochs |
| Evaluation output | `rlhf_runs/checkpoints_ckpt100_full/` |

### Overall Winner Counts

| Policy | Wins | Win rate | Mean reward | Median response tokens | Cap-hit rate | Empty rate |
|---|---:|---:|---:|---:|---:|---:|
| Base | 718 | 35.60% | 0.0803 | 331 | 8.82% | 0.00% |
| SFT | 508 | 25.19% | 0.0652 | 371 | 13.39% | 0.00% |
| PPO | 775 | 38.42% | 0.7300 | 520 | 27.42% | 0.00% |
| Tie | 16 | 0.79% | - | - | - | - |

### Pairwise Comparisons

| Comparison | Left wins | Right wins | Ties | Right win rate | Mean right-left reward delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 1158 | 837 | 22 | 41.50% | -0.0151 |
| Base vs PPO | 981 | 1027 | 9 | 50.92% | +0.6497 |
| SFT vs PPO | 840 | 1164 | 13 | 57.71% | +0.6648 |

PPO is strongest in the general-prompt subset and weaker on code, STEM, and multilingual prompts:

| Domain | PPO wins vs Base | Base wins | Ties | PPO win rate |
|---|---:|---:|---:|---:|
| Code | 188 | 250 | 0 | 42.92% |
| General | 529 | 399 | 3 | 56.82% |
| STEM | 118 | 126 | 1 | 48.16% |
| Multilingual | 192 | 206 | 5 | 47.64% |

### Qualitative Audit

The stronger reward-model result does not remove the need for inspection. PPO responses are longer and more likely to reach the cap. They also repeat more often.

| Policy | Cap-hit rate | Repeated 4-grams >25% | Repeated 4-grams >50% |
|---|---:|---:|---:|
| Base | 8.82% | 204 (10.11%) | 61 (3.02%) |
| SFT | 13.39% | 405 (20.08%) | 171 (8.48%) |
| PPO | 27.42% | 643 (31.88%) | 319 (15.82%) |

The audit found:

- 8 likely genuine PPO wins and 354 modest clean wins under deterministic triage;
- 64 strong PPO regressions, 288 severe repetition failures, and 151 reward-model false-positive risks;
- full prompts and responses for all 2,017 validation examples;
- a 100-example first-pass subset balanced as 50 positive and 50 negative cases.

The complete response-explorer artifact is in [`rlhf_runs/portfolio_full_policy_comparisons_final_trl.json`](rlhf_runs/portfolio_full_policy_comparisons_final_trl.json), with the balanced subset recorded in [`rlhf_runs/portfolio_curated_100_manifest_final_trl.json`](rlhf_runs/portfolio_curated_100_manifest_final_trl.json).

## Interpretation

This run is the first one in the project where PPO edges the base model under the learned reward model on the full validation suite. It is also plainly not a solved alignment result. The policy often writes longer answers, and the reward model still over-rewards some repetitive or bloated responses. The project conclusion is therefore balanced:

- the TRL RLHF pipeline works end to end on real HelpSteer3 data;
- PPO can change behavior and produce useful local improvements;
- the learned reward model is strong enough to train with but not reliable enough to trust blindly;
- qualitative auditing is part of the result, not an optional afterthought;
- future improvement should focus on reward-model reliability, hard negatives, stopping behavior, and controlled checkpoint selection.

Older custom-training results are preserved in the technical companion and in the machine-readable records under [`experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/`](experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/). Those runs were essential for debugging long context, evaluation caps, checkpoint loading, and repetition diagnostics, but the final result reported here is the TRL run above.

The complete pre-cleanup implementation, including all custom training entry points, is frozen at Git commit `6233219d2ee34517bc4203e0bb986f7cb27bf5d1`. The current source tree intentionally keeps the TRL training path and the repository-owned policy-suite evaluation/audit infrastructure.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/rlhf/` | shared data/configuration utilities, TRL trainers, and policy-suite evaluation support |
| `scripts/` | command-line training, evaluation, audit, and comparison entry points |
| `configs/trl/` | active TRL SFT, reward-model, PPO, DPO, and evaluation configs |
| `configs/rlhf/` | historical custom-loop configs |
| `docs/` | concise project report, complete technical companion, and retained Colab training record |
| `experiments/baselines/` | frozen pre-TRL baseline records |
| `rlhf_runs/` | local final-run summaries, curation notebooks, and portfolio export artifacts |
| `rlhf_runs_lightweight_export/` | lightweight copy of Colab logs/configs without model weights |

## Recommended Reading Order

1. [`docs/rlhf_project_report.pdf`](docs/rlhf_project_report.pdf): concise two-column academic account of the method, final results, interpretation, and controlled DPO extension.
2. [`docs/rlhf_technical_companion.pdf`](docs/rlhf_technical_companion.pdf): trainer design and implementation details, chronological experiment log, Colab and checkpoint operations, qualitative and reward-mismatch audit, artifact map, DPO execution record, and future-work agenda.

The root README is the repository's only maintained Markdown documentation. Detailed prose lives in the two LaTeX report sources so that experimental history and current conclusions cannot drift across parallel documents.
