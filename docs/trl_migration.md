# Hugging Face TRL Migration

## Purpose

The active training path uses Hugging Face TRL for supervised fine-tuning,
reward modeling, and PPO. The previous custom implementation is
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
| Disable dropout during PPO | The model loader explicitly sets PyTorch dropout modules to `p=0.0`; LoRA dropout is required to be zero; reward training also passes `disable_dropout=True`. |
| Match stored behavior probabilities | TRL stores generation logits and applies the same temperature during rollout and PPO ratio recomputation. Before trainer construction, the wrapper also neutralizes Qwen's model-card repetition and sampling heuristics so they cannot silently alter sampled probabilities. |
| Preserve complete responses and EOS | Repository preprocessing reserves completion space, appends EOS, and truncates prompt history first. |
| Initialize RM from SFT | Reward training loads the merged SFT model. |
| Initialize the scalar head deliberately | N+ head initialization is applied and recorded. |
| Center reward scale | TRL center regularization plus a persisted demonstration-score offset are used. |
| Initialize critic from RM | PPO loads the trained RM weights as the value model. |
| Use a fixed SFT reference | The PPO adapter is disabled for reference log-probabilities. |
| Penalize missing EOS | PPO uses fixed-length generation with EOS handling and a configurable invalid reward for missing EOS; the final run used `-1.0`. |
| Normalize advantages | TRL performs masked advantage whitening. |
| Keep reward whitening explicit | The setting is explicit in config/overrides and was enabled in the final PPO run. |

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
  --eval-dir rlhf_runs/checkpoints_ckpt100_full \
  --base-label base \
  --sft-label sft_trl \
  --ppo-label ppo_exact_ckpt100
```

Every command accepts repeated `--set dotted.path=value` overrides. Resolved
configs and manifests are written into each output directory.

Before a Colab stage, `scripts/rlhf_trl_doctor.py` can validate the exact
interpreter, package versions, CUDA visibility, prepared dataset paths, and
local/Drive write permissions. The Colab notebook runs this automatically
before SFT, reward modeling, and PPO, and streams child-process output so the
original traceback is not hidden behind a generic `CalledProcessError`.

The TRL evaluation suite loads the tokenizer saved with SFT, including its
distinct PAD token. The Base model is resized once for that token before
generation. This keeps policy padding and reward-model final-token pooling
consistent across Base, SFT, and PPO. Neutral decoding controls are passed
explicitly so checkpoint-specific generation defaults cannot make the
comparison asymmetric.

Evaluation overrides may address policy lists with either
`policies.1.checkpoint_dir=...` or
`policies[1].checkpoint_dir=...`. The policy labeled `base` should retain a
null checkpoint so it loads `model.name`; pointing it at the SFT checkpoint
would make the Base and SFT columns identical.

## Colab and checkpoints

The Colab notebook assumes the repository has already been cloned, typically
under Google Drive, and that `REPO_DIR` points at that clone. It no longer
clones the repository itself. Keeping the code checkout on Drive is fine
because Python files, configs, and notebooks are small relative to model
training.

Use local Colab storage for active training outputs and copy checkpoints to
Drive with `train.checkpoint_sync_dir` and final artifacts with
`train.final_sync_dir`. Local SSD access is faster than writing model shards,
datasets, logs, and temporary Trainer state directly into mounted Drive. It
also keeps the Git repository small, avoids accidental commits of large model
files, and makes it easier to download or share the source tree without
pulling checkpoints. SFT and reward training support exact Transformers
checkpoint resume through `train.resume_from_checkpoint`.

The current experimental TRL PPO trainer path writes checkpoints, but this
wrapper does not expose an exact `resume_from_checkpoint` path. It rejects that
option instead of pretending it is safe. For long PPO experiments, run
deliberate segments: finish one segment, use its merged policy and saved value
model as the next segment's initialization, retain the original SFT reference,
and record the parent run. This is continuation training, not exact
optimizer/dataloader resume.

## Final full run

The final reported run used the Colab full profile plus explicit overrides recorded in `rlhf_runs/rlhf_trl_colab_pipeline_final.ipynb`.

- SFT: one epoch, effective batch 32, 4096 total tokens, LoRA rank 16.
- Reward model: two total epochs, effective batch 64, 4096 total tokens, LoRA rank 32, resumed from the first reward-model checkpoint.
- PPO: configured for 12,000 episodes and evaluated after 6,400 episodes / 100 optimizer steps; 768-token PPO rollouts; four PPO epochs; KL coefficient 0.07; reward whitening enabled; missing-EOS reward `-1.0`; Adam epsilon `1e-5`.
- Evaluation: full 2,017-prompt policy suite with a 1024-token generation cap and a 3072-token prompt budget.

The final policy-suite evaluation reports PPO winning 50.92% of pairwise comparisons against Base and 57.71% against SFT under the learned reward model. This is the best PPO result in the project, but it still requires caveats: PPO has a 27.42% cap-hit rate and 31.88% of PPO responses exceed the 25% repeated 4-gram threshold. The final documentation therefore treats reward score as a diagnostic signal, not as a human-quality label.

The output directory name still contains `r512` from an earlier planned setting. The executed notebook overrides `ppo.response_length` to 768; documentation should use the executed override rather than the stale directory label.

## Versioning note

TRL PPO is experimental. `requirements.txt` intentionally installs the
current available package stack rather than pinning exact versions. The doctor
script therefore validates imports and the specific TRL trainer APIs used by
this repository before expensive stages begin. A dependency upgrade should
still be treated as an experiment change: run the smoke profile and record the
resolved package versions before launching a costly run.
