# RLHF Repository Guide

This guide contains the operational and structural details intentionally kept out of the root README. The root README tells the project story and reports the final guarded-PPO result; this document explains how the repository is organized and how the workflows are executed and resumed.

## Installation

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

The project is designed around CUDA notebook runtimes, bfloat16, TF32, local Colab SSD for active work, and explicit synchronization of durable artifacts to Google Drive.

## Active Training Commands

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

Every command supports repeated dotted configuration overrides:

```bash
python -m scripts.rlhf_trl_train_ppo \
  --config configs/trl/qwen25_05b_helpsteer3_ppo.yaml \
  --set train.output_dir=outputs/trl/ppo_experiment \
  --set train.target_update=25
```

Resolved configurations and manifests are written into the run directory and should be treated as the source of truth for executed experiments.

## Canonical Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/rlhf_trl_colab_pipeline.ipynb` | SFT, reward-model, and earlier TRL PPO pipeline |
| `notebooks/rlhf_ppo_guarded_colab_pipeline.ipynb` | final guarded, exactly resumable 188-update PPO workflow |
| `notebooks/rlhf_full_eval_and_curation.ipynb` | full 2,017-prompt policy-suite evaluation and qualitative curation |
| `notebooks/analyzing_full_eval_results.ipynb` | aggregate and domain-level analysis of suite outputs |
| `notebooks/rlhf_dpo_colab_pipeline.ipynb` | implemented DPO extension; retained as a future controlled comparison |

## Repository Structure

| Path | Purpose |
|---|---|
| `src/rlhf/` | data contracts, formatting, TRL trainers, guarded PPO patches, evaluation, metrics, and experiment utilities |
| `scripts/` | command-line entry points for preparation, training, evaluation, auditing, synchronization, and diagnostics |
| `configs/trl/` | canonical TRL SFT, reward-model, guarded PPO, DPO, and evaluation configurations |
| `configs/rlhf/` | historical custom-loop configurations retained for provenance |
| `notebooks/` | Colab execution, evaluation, and curation workflows |
| `docs/` | main report, technical companion, this guide, and retained training notebook |
| `research/` | papers used to ground the implementation and experimental design |
| `tests/` | data-contract, configuration, reward-calibration, resume, and PPO-guardrail tests |
| `rlhf_runs/` | lightweight synchronized run artifacts, evaluation tables, plots, manifests, and summaries |
| `outputs/` | local working outputs and smoke-test artifacts |

Large model weights and optimizer states may be excluded from lightweight repository copies. Run summaries, resolved configurations, manifests, diagnostics, and evaluation outputs are retained so reported results remain auditable.

## Data Preparation Contract

HelpSteer3 rows are converted into three stage-specific datasets:

- **SFT:** prompt plus preferred completion, with loss applied only to completion tokens;
- **reward modeling:** prompt plus chosen/rejected completions;
- **PPO:** deduplicated prompts with domain metadata for balanced rollout planning.

The Qwen chat template is used across stages. PAD and EOS remain distinct. For SFT and reward modeling, complete EOS-terminated responses are preserved; old prompt turns are removed before any final prompt-side token fallback. Tied or invalid preference rows are excluded from preference training.

## Guarded PPO Operational Contract

The canonical guarded PPO configuration is `configs/trl/qwen25_05b_helpsteer3_ppo.yaml`. Its major contracts are:

- 12,000 configured episodes, rounded to 12,032 actual rollouts / 188 updates at batch size 64;
- 768 generated response tokens;
- exactly 16 prompts from each of code, general, STEM, and multilingual per update;
- frozen SFT reference and RM-initialized critic;
- fixed-length generation followed by EOS truncation;
- calibrated terminal reward bounds and EOS replacement from a 4,096-pair stratified sample;
- smooth repeated-token 4-gram shaping above the preferred-response 95th percentile;
- KL coefficient 0.10, four PPO epochs, clipped policy/value objectives, and advantage whitening;
- reward whitening disabled so calibrated terminal values retain their intended scale.

### Memory execution path

The guarded PPO wrapper modifies execution, not the objective:

1. generation returns token IDs without full-vocabulary score history;
2. behavior log-probabilities are recomputed in bounded chunks from response-position logits only;
3. prompt columns masked for every row in a processing chunk are removed;
4. the reference uses the policy backbone with the LoRA adapter disabled;
5. policy and value losses use separate backward graphs whose gradients accumulate before one optimizer step;
6. generation and log-probability chunk candidates fall back to smaller values after restoring RNG state, preserving sampled trajectories.

The run writes phase-level allocated/reserved VRAM and throughput diagnostics for rollout generation, rollout scoring, policy backward, value backward, and optimizer stepping.

## Checkpoints and Exact Resume

SFT and reward modeling use standard Transformers/TRL checkpoint resume. Guarded PPO adds a stricter verified-checkpoint layer. A checkpoint is considered resumable only after all required state has been written and an `exact_resume_complete.json` marker is present.

A healthy PPO checkpoint contains:

- LoRA policy adapter;
- complete value model;
- optimizer and scheduler state;
- Trainer state;
- Python, NumPy, CPU, and CUDA RNG state;
- rollout-plan and data-position metadata;
- reward-guardrail fingerprint;
- checkpoint-health report.

Incomplete newer checkpoint directories are ignored in favor of the latest verified checkpoint. The configuration and rollout/guardrail fingerprints prevent silently resuming under changed semantics.

## Resumable Evaluation

Policy-suite evaluation writes one JSONL shard per policy as generation proceeds. Rerunning the same evaluation skips completed prompt-policy records and resumes the incomplete policy. A fingerprint binds cached outputs to policy checkpoint and generation settings.

The output directory contains:

- `partial_policy_outputs/*.jsonl` — interrupt-safe policy shards;
- `policy_suite_samples.csv/jsonl/xlsx` — the shared Base/SFT/PPO response table;
- `policy_suite_summary.md/json` — overall and pairwise results;
- `policy_suite_pairwise_summary.csv` — comparison-ready metrics;
- `qualitative_audit_auto.md` and `curation_*.csv` — audit queues;
- `plots/` — reward, length, winner, and domain visualizations;
- `config_resolved.yaml` and `experiment_manifest.json` — run provenance.

Use a new output directory whenever checkpoint identity, prompt ordering, seed, response budget, or decoding settings change.

## DPO Extension

A controlled DPO path is implemented but is not part of the final reported result. It reuses the frozen one-epoch SFT checkpoint, non-tied HelpSteer3 pairs, Qwen tokenizer, 4,096-token preparation budget, LoRA targets, and final evaluation prompts.

```bash
python -m scripts.rlhf_trl_prepare_data \
  --config configs/trl/qwen25_05b_helpsteer3_dpo.yaml

python -m scripts.rlhf_trl_train_dpo \
  --config configs/trl/qwen25_05b_helpsteer3_dpo.yaml
```

The primary design uses standard sigmoid DPO with beta `0.1`, one epoch, effective batch size 32, LoRA rank 16, zero dropout, and precomputed reference log-probabilities. HelpSteer3 preference strength is retained as metadata; the primary run does not assume the ordinal levels are linearly spaced utility weights. A separately named `linear_replication` configuration exists only as a sensitivity experiment.

## Historical Code and Experiment Record

The project initially used custom SFT, reward-model, and token-level PPO loops. Those implementations exposed truncation, EOS collapse, checkpoint-loading errors, KL drift, and reward exploitation before the trainer loops were migrated to TRL.

Two revision boundaries preserve this history:

- custom implementation snapshot: `6cbf214fcf1b91c7b756e303e533c2c86d2eba89`;
- complete pre-pruning repository: `6233219d2ee34517bc4203e0bb986f7cb27bf5d1`.

The detailed chronological record and metrics from superseded runs live in `docs/rlhf_technical_companion.tex` / `.pdf`; the root README intentionally reports only the final guarded run.

## Recommended Reading Order

1. Root `README.md` for the question, final pipeline, and final result.
2. `docs/rlhf_project_report.pdf` for the concise academic account.
3. `docs/rlhf_technical_companion.pdf` for implementation decisions, historical experiments, failure analysis, and artifact governance.
4. `notebooks/rlhf_full_eval_and_curation.ipynb` for direct exploration of the final evaluation table.
5. `src/rlhf/trl_train_ppo.py` and `tests/test_trl_ppo_guardrails.py` for the guarded PPO implementation and invariants.
