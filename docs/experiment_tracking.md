# Experiment Tracking

The project uses a small filesystem-based tracking contract. It keeps the useful
parts of larger tracking systems without adding a server or third-party runtime
dependency.

Every new training or evaluation output directory contains:

| File | Purpose |
|---|---|
| `config_resolved.yaml` | exact configuration consumed by the run |
| `experiment_manifest.json` | identity, intent, config hash, command, source state, environment, status, and summary |
| `run_metadata.json` | backward-compatible hardware and software metadata for training runs |
| `*_metrics.jsonl` / `*.csv` | step-level measurements |
| `run_summary.json` or evaluation summary | final aggregate results |
| checkpoints, samples, and plots | stage-specific artifacts |

An interrupted process leaves its manifest in `running` state. Re-running in
the same directory appends an attempt and records whether the resolved
configuration changed. Successful completion updates the status and embeds the
available summary.

## Declaring Intent

Each YAML config may include metadata that does not change training behavior:

```yaml
experiment:
  id: ppo-reward-whitening-v1
  name: PPO reward-whitening ablation
  group: qwen25-05b-helpsteer3
  tags: [ppo, reward-whitening, ablation]
  hypothesis: Reward whitening stabilizes scale without weakening KL control.
  parent_runs:
    - qwen25-05b-helpsteer3-ppo-long512-baseline
```

Algorithmic ingredients still belong in their functional sections, such as
`generation`, `reward_shaping`, `kl`, and `ppo`. Because the complete resolved
config is stored and hashed, future switches such as reward whitening, EOS
penalties, padding policy, or a different advantage estimator become explicit
and comparable as soon as they are added to the implementation and config.

## Comparing Runs

Compare two run directories, manifests, or YAML files:

```bash
python scripts/rlhf_compare_runs.py \
  outputs/rlhf/baseline_run \
  outputs/rlhf/new_run
```

The command prints only changed behavioral parameters and hides the
non-behavioral `experiment.*` metadata block by default. Use
`--include-metadata`, `--format json`, or `--output comparison.md` when needed.

The frozen pre-restructure baseline lives at
[`experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/`](../experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/).
Its source implementation is Git commit
`6cbf214fcf1b91c7b756e303e533c2c86d2eba89`. Active experiments use the
checked-in `configs/trl/` configurations and record `trl_backend: true` in
their manifests.

## Design Basis

The `src/` package layout follows the PyPA recommendation that importable code
be separated from the repository root. The manifest follows the common
run/parameters/metrics/artifacts model documented by MLflow and Weights &
Biases, implemented locally here. PPO-specific fields remain in the resolved
config, including the KL, clipping, GAE, value-loss, and reward-normalization
choices also exposed by TRL's PPO configuration.

- [PyPA: `src` layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Weights & Biases configuration](https://docs.wandb.ai/models/track/config)
- [TRL PPO Trainer](https://huggingface.co/docs/trl/ppo_trainer)
