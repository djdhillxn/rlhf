# Historical A100 40GB RLHF run profile

This note preserves the settings used during an early short-context stage of the project. It is not the configuration used for the final 4096-token SFT/reward-model runs, 768-token PPO rollouts, or 1024-token policy-suite evaluation. Use the repository [`README.md`](../README.md) for the current workflow.

The historical profile targeted Google Colab A100 40GB with Qwen2.5-0.5B-Instruct and favored speed over the earlier low-memory debug defaults.

## Reward model

Use the default config:

```bash
python3 scripts/rlhf_train_reward_model.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_reward.yaml
```

Important settings:

- full BF16 backbone (`load_in_4bit: false`) because a 0.5B model fits comfortably on A100 and is faster than bitsandbytes 4-bit training;
- `batch_size: 8`, `gradient_accumulation_steps: 2`, effective batch size 16;
- full HelpSteer3 preference data after filtering ties/malformed pairs;
- token/log throughput and CUDA memory are written to `train_metrics.jsonl`/CSV;
- CSV files and plots refresh every `artifact_every` optimizer steps.

If peak GPU memory stays below roughly 28-30 GB and the run is stable, try `batch_size: 12` with `gradient_accumulation_steps: 2`.
If memory exceeds roughly 36 GB or the run OOMs, use `batch_size: 6`.

## PPO

Use:

```bash
python3 scripts/rlhf_train_ppo.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_ppo.yaml
```

Important settings:

- policy/reference/reward models are full BF16;
- rollout batch size is 16;
- PPO minibatch size is 4;
- max prompt length is 768 and max generation length is 160;
- checkpoints are saved every 50 updates;
- CSV metrics refresh every 25 updates.

If peak GPU memory is comfortably below 30-32 GB, increase `rollout_batch_size` to 24 first.
Only increase `minibatch_size` to 6 or 8 after confirming that the PPO update step is not near OOM.

## Evaluation

The early workflow used:

```bash
python3 scripts/rlhf_evaluate_before_after.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_eval.yaml
```

It evaluated 200 validation prompts and wrote before/after artifacts. The completed project instead uses:

```bash
python3 scripts/rlhf_evaluate_policy_suite.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_eval_suite.yaml
```

The current suite compares Base, SFT, and PPO on all 2017 validation prompts. Results and caveats are documented in [`rlhf_experiments.md`](rlhf_experiments.md).
