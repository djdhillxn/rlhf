# RLHF Post-Training with Qwen2.5, HelpSteer3, and TRL PPO

This project adapts the trust-region idea behind PPO from classical reinforcement learning to language-model post-training. It grew out of my earlier [TRPO, NPG, and PPO](https://github.com/djdhillxn/trpo) work on MuJoCo and Atari, then asks the same question in an RLHF setting:

> Can a small instruction model be supervised, judged, and PPO-aligned in a way that is measurable, reproducible, and honest about failure modes?

The final pipeline uses **Qwen2.5-0.5B-Instruct**, **NVIDIA HelpSteer3**, **Hugging Face TRL**, and LoRA adapters for three training stages plus a final evaluation stage:

1. supervised fine-tuning (SFT) on preferred HelpSteer3 responses;
2. reward-model training on chosen/rejected preference pairs;
3. token-level PPO with a frozen SFT reference and a learned reward model;
4. full policy-suite evaluation of Base, SFT, and PPO responses on the same validation prompts.

The final PPO policy does not make the 0.5B model universally better. It does produce the strongest run from this project: under the learned reward model, PPO wins **50.92%** of pairwise comparisons against the base instruction model and **57.71%** against the SFT policy on the 2,017-prompt evaluation. The same audit also shows a real cost: PPO is longer, hits the generation cap more often, and has the highest repetition rate. The result is therefore a useful RLHF case study rather than a blanket claim of model superiority.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

The Colab orchestration notebook is [`notebooks/rlhf_trl_colab_pipeline.ipynb`](notebooks/rlhf_trl_colab_pipeline.ipynb). The executed final-run notebook and lightweight exported artifacts are stored locally under `rlhf_runs/` and `rlhf_runs_lightweight_export/` for analysis.

## Active Pipeline

The active training path uses Hugging Face TRL for SFT, reward modeling, and PPO. The repository still owns the HelpSteer3 preprocessing, chat formatting, manifests, evaluation suite, repetition diagnostics, and qualitative curation.

```bash
python scripts/rlhf_trl_prepare_data.py \
  --config configs/trl/qwen25_05b_helpsteer3_sft.yaml

python scripts/rlhf_trl_train_sft.py \
  --config configs/trl/qwen25_05b_helpsteer3_sft.yaml

python scripts/rlhf_trl_train_reward_model.py \
  --config configs/trl/qwen25_05b_helpsteer3_reward.yaml

python scripts/rlhf_trl_train_ppo.py \
  --config configs/trl/qwen25_05b_helpsteer3_ppo.yaml

python scripts/rlhf_evaluate_policy_suite.py \
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

- 8 low-repetition PPO candidates that beat both Base and SFT under the reward model;
- 108 strong PPO losses;
- 313 high-reward repetition or reward-model mismatch risks;
- 50 curated portfolio examples, balanced as 25 positive and 25 negative cases.

The curated response explorer data is in [`rlhf_runs/portfolio_curated_policy_comparisons_ckpt100.json`](rlhf_runs/portfolio_curated_policy_comparisons_ckpt100.json). The deep curation report is [`rlhf_runs/ckpt100_deep_curation_report.md`](rlhf_runs/ckpt100_deep_curation_report.md).

## Interpretation

This run is the first one in the project where PPO edges the base model under the learned reward model on the full validation suite. It is also plainly not a solved alignment result. The policy often writes longer answers, and the reward model still over-rewards some repetitive or bloated responses. The project conclusion is therefore balanced:

- the TRL RLHF pipeline works end to end on real HelpSteer3 data;
- PPO can change behavior and produce useful local improvements;
- the learned reward model is strong enough to train with but not reliable enough to trust blindly;
- qualitative auditing is part of the result, not an optional afterthought;
- future improvement should focus on reward-model reliability, hard negatives, stopping behavior, and controlled checkpoint selection.

Older custom-training results are preserved in [`docs/rlhf_experiments.md`](docs/rlhf_experiments.md) and [`experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/`](experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/). Those runs were essential for debugging long context, evaluation caps, checkpoint loading, and repetition diagnostics, but the final result reported here is the TRL run above.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/rlhf/` | data preparation, TRL wrappers, reward/evaluation utilities, and legacy custom components |
| `scripts/` | command-line training, evaluation, audit, and comparison entry points |
| `configs/trl/` | active TRL SFT, reward-model, PPO, and evaluation configs |
| `configs/rlhf/` | historical custom-loop configs |
| `docs/` | experiment history, TRL migration notes, audit notes, and future work |
| `experiments/baselines/` | frozen pre-TRL baseline records |
| `rlhf_runs/` | local final-run summaries, curation notebooks, and portfolio export artifacts |
| `rlhf_runs_lightweight_export/` | lightweight copy of Colab logs/configs without model weights |

## Recommended Reading Order

1. [`docs/trl_migration.md`](docs/trl_migration.md): active TRL design, N+ implementation coverage, and final run settings.
2. [`docs/rlhf_experiments.md`](docs/rlhf_experiments.md): chronological experiment log from short-context debugging through the final TRL run.
3. [`docs/rlhf_qualitative_audit.md`](docs/rlhf_qualitative_audit.md): evidence for clean wins, strong losses, repetition, and reward-model mismatch.
4. [`docs/rlhf_curation_guide.md`](docs/rlhf_curation_guide.md): how to reproduce curation and export portfolio examples.
5. [`docs/rlhf_future_work.md`](docs/rlhf_future_work.md): the research program that follows from the final limitations.
6. [`docs/experiment_tracking.md`](docs/experiment_tracking.md): run manifests, exact configs, and reproducibility contract.
