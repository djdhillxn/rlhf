# Hugging Face TRL Migration

## Purpose

The active training path uses Hugging Face TRL 1.6.0 for supervised
fine-tuning, reward modeling, and PPO. The previous custom implementation is
preserved as a frozen historical baseline at Git commit
`6cbf214fcf1b91c7b756e303e533c2c86d2eba89`.

This migration does not discard the project. HelpSteer3 parsing, chat
formatting, truncation policy, experiment manifests, evaluation, repetition
diagnostics, reward-margin analysis, and qualitative curation remain
repository-owned. TRL now owns the trainer loops, distributed preparation,
mixed precision, gradient accumulation, checkpoint serialization, PPO ratio
calculation, KL penalties, clipped policy/value objectives, and advantage
estimation.

## Stage architecture

The data-preparation command builds three tokenized datasets from the same
filtered HelpSteer3 rows:

- SFT records contain `input_ids` and a `completion_mask`, so loss is computed
  only on the preferred assistant response.
- Reward records contain paired `chosen_ids` and `rejected_ids` with a shared
  prompt prefix.
- PPO records contain deduplicated, left-padded prompt token IDs.

Responses are never truncated to make room for a prompt. The preprocessor
first removes the oldest non-system turns, then left-truncates prompt tokens
only as a final fallback. Every SFT and reward completion ends in EOS. PAD and
EOS are distinct token IDs.

SFT starts from `Qwen/Qwen2.5-0.5B-Instruct` and trains a LoRA adapter with
response-only loss. The merged SFT model becomes both the reward-model
initialization and PPO reference policy.

The reward model starts from the merged SFT weights. Its scalar head is
initialized with standard deviation `1 / sqrt(hidden_size + 1)` and zero bias,
matching the N+ reference implementation. TRL applies the Bradley-Terry
pairwise objective and an explicit reward-centering regularizer. After
training, the repository measures the mean score on preferred SFT
demonstrations and stores it in `reward_center.json`; PPO and evaluation
subtract that fixed offset.

PPO starts a fresh policy LoRA adapter from the merged SFT policy. The frozen
reference is the same SFT model with the adapter disabled. Separate reward and
value models are both initialized from the trained reward model, so the critic
does not begin as an unrelated random value head.

## Implementation-detail coverage

The active path incorporates the highest-priority lessons from *The N+
Implementation Details of RLHF with PPO*:

| Detail | Active behavior |
|---|---|
| Disable dropout during PPO | TRL disables dropout in policy, reference, reward, and value models; LoRA dropout is required to be zero. |
| Match stored behavior probabilities | TRL stores generation logits and applies the same temperature during rollout and PPO ratio recomputation. Before trainer construction, the wrapper also neutralizes Qwen's model-card repetition and sampling heuristics so they cannot silently alter sampled probabilities. |
| Preserve complete responses and EOS | Repository preprocessing reserves completion space, appends EOS, and truncates prompt history first. |
| Initialize RM from SFT | Reward training loads the merged SFT model. |
| Initialize the scalar head deliberately | N+ head initialization is applied and recorded. |
| Center reward scale | TRL center regularization plus a persisted demonstration-score offset are used. |
| Initialize critic from RM | PPO loads the trained RM weights as the value model. |
| Use a fixed SFT reference | The PPO adapter is disabled for reference log-probabilities. |
| Penalize missing EOS | PPO uses EOS stopping and a configurable missing-EOS penalty. |
| Normalize advantages | TRL performs masked advantage whitening. |
| Keep reward whitening explicit | It is disabled by default and can be enabled as a controlled ablation. |

TRL is not a guarantee against reward hacking, weak judges, poor data, or bad
hyperparameters. The reward audit and human-readable response review remain
required.

## Commands

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

python scripts/rlhf_audit_policy_suite.py \
  --eval-dir outputs/trl/qwen25_05b_helpsteer3_eval1024_v1 \
  --base-label base \
  --sft-label sft_trl \
  --ppo-label ppo_trl
```

Every command accepts repeated `--set dotted.path=value` overrides. Resolved
configs and manifests are written into each output directory.

Before a Colab stage, `scripts/rlhf_trl_doctor.py` can validate the exact
interpreter, package versions, CUDA visibility, prepared dataset paths, and
local/Drive write permissions. The Colab notebook runs this automatically
before SFT and streams child-process output so the original traceback is not
hidden behind a generic `CalledProcessError`.

The TRL evaluation suite loads the tokenizer saved with SFT, including its
distinct PAD token. The Base model is resized once for that token before
generation. This keeps policy padding and reward-model final-token pooling
consistent across Base, SFT, and PPO. Neutral decoding controls are passed
explicitly so checkpoint-specific generation defaults cannot make the
comparison asymmetric.

## Colab and checkpoints

Use local Colab storage for active training and copy checkpoints to Drive with
`train.checkpoint_sync_dir` and final artifacts with `train.final_sync_dir`.
Local SSD access is faster than training directly against mounted Drive. SFT
and reward training support exact Transformers checkpoint resume through
`train.resume_from_checkpoint`.

TRL 1.6.0's experimental PPO trainer writes checkpoints but does not implement
an exact `resume_from_checkpoint` path. The wrapper rejects that option instead
of pretending it is safe. For long PPO experiments, run deliberate segments:
finish one segment, use its merged policy and saved value model as the next
segment's initialization, retain the original SFT reference, and record the
parent run. This is continuation training, not exact optimizer/dataloader
resume.

## Recommended first run

First run the notebook's `smoke` profile. It uses small dataset slices, short
responses, and a few optimizer steps to verify tokenizer growth, dataset
columns, adapter merging, reward centering, PPO rollout, evaluation, and audit.

After that passes, run the checked-in pilot:

- SFT: one epoch, effective batch 16, 4096 total tokens.
- Reward model: one epoch, effective batch 32, 4096 total tokens.
- PPO: 2,048 episodes, rollout batch 16, four PPO epochs, 512 response tokens.
- Evaluation: first inspect a 128-prompt subset, then run the full 2,017-prompt
  1024-token suite only after PPO diagnostics look healthy.

The pilot is intentionally smaller than a final campaign. Continue only if
reward accuracy/calibration are credible and PPO logs show finite losses,
bounded KL, non-collapsing entropy, reasonable EOS rates, and no sharp growth
in repetition.

## Versioning note

TRL PPO is experimental. `requirements-rlhf.txt` pins TRL 1.6.0 and the
matching PyTorch 2.6.0, torchvision 0.21.0, and torchaudio 2.6.0 family. TRL's
FSDP imports require this newer PyTorch generation, and keeping the three
PyTorch packages aligned avoids a common Colab binary mismatch. Core
dependencies are otherwise bounded. An upgrade should be treated as an
experiment change:
review upstream PPO source, run the smoke profile, and record a new baseline
before launching a costly run.
