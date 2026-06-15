# RLHF Post-Training with Qwen2.5, HelpSteer3, and Token-Level PPO

This project grew from earlier work on [TRPO, NPG, and PPO](https://github.com/djdhillxn/trpo) across MuJoCo and Atari. That study of conservative policy improvement led naturally to the same question in language-model post-training:

> Can the same PPO trust-region idea be adapted from environment rollouts to language-model post-training?

The implementation trains an RLHF pipeline around **Qwen2.5-0.5B-Instruct** and **NVIDIA HelpSteer3**:

1. supervised fine-tuning (SFT) on preferred HelpSteer3 responses,
2. reward-model training from HelpSteer3 chosen/rejected preference pairs,
3. KL-controlled token-level PPO on sampled LLM responses,
4. policy-suite evaluation comparing Base, SFT, and PPO responses on the same prompts.

This student-scale run does not beat a modern instruction-tuned model. It does provide an end-to-end, debuggable RLHF pipeline with documented failure modes, long-context data handling, reward-model diagnostics, PPO checkpoints, resumable evaluation, and qualitative example curation.

## From policy optimization to RLHF

The earlier policy-optimization project examined how vanilla policy gradients can move a policy too far, while NPG, TRPO, and PPO measure or constrain policy movement in policy space. This RLHF project keeps that same idea but changes the setting:

| Policy-optimization project | RLHF project |
|---|---|
| MuJoCo / Atari state | chat prompt tokens |
| action | generated token |
| rollout trajectory | prompt + generated response |
| environment reward | learned scalar reward model |
| old-policy KL / trust region | KL to frozen SFT reference model |
| PPO update on action log-probs | PPO update on response-token log-probs |

The implementation therefore keeps a dedicated `src/rlhf/` package instead of directly reusing the Gym-oriented PPO code. The math is still PPO-Clip with a KL anchor, but the tensors, masking, sampling, and scoring pipeline are different.

## Setup

```bash
python3 -m pip install -r requirements-rlhf.txt
python3 -m pip install -e .
```

## Repository structure and experiment records

The importable implementation uses a standard `src/` layout:

| Path | Purpose |
|---|---|
| `src/rlhf/` | SFT, reward-model, PPO, rollout, evaluation, and tracking code |
| `scripts/` | stable command-line entry points |
| `configs/rlhf/` | editable experiment configurations |
| `outputs/rlhf/` | checkpoints, metrics, summaries, samples, and plots |
| `experiments/baselines/` | frozen resolved configs and baseline records |
| `docs/` | experiment history, audits, future work, and operating notes |

Each new run writes `experiment_manifest.json` beside its resolved config and
metrics. The manifest records the experiment identity and hypothesis, exact
config hash, invocation, Git state, environment, lifecycle status, and final
summary. Existing artifact names remain unchanged for notebook and script
compatibility.

To inspect the exact ingredient changes between two runs:

```bash
python scripts/rlhf_compare_runs.py \
  outputs/rlhf/first_run \
  outputs/rlhf/second_run
```

See [`experiment_tracking.md`](docs/experiment_tracking.md) for the compact
tracking contract. The completed 4096/512/1024 run is frozen under
[`experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/`](experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/).

## Model and data

### Base model

We use `Qwen/Qwen2.5-0.5B-Instruct`, a 0.49B parameter instruction-tuned model. The Qwen2.5-0.5B-Instruct model card lists a **32,768-token context length** and **8,192-token generation length**. The model itself can therefore support far longer generations than we used, but full-length RLHF training is much more expensive than inference.

### Dataset

We use HelpSteer3 preference data. Each training example contains a conversation context, two candidate responses, domain/language metadata, and a preference score. The final run used the full train/validation splits after filtering invalid or tied preference rows.

### Chat formatting

HelpSteer3 stores messages in a chat-style format. Before training/evaluation, we render those messages with the Qwen tokenizer chat template:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
```

Qwen chat/instruct models expect this format, which is produced through `tokenizer.apply_chat_template(...)`.

## Final long-context configuration

Earlier experiments used short output budgets such as 128 new tokens. Those runs were useful for debugging, but they clipped many responses and were not suitable for qualitative examples. We therefore ran a final long-context version.

| Stage | Final setting |
|---|---:|
| SFT max sequence length | 4096 total tokens |
| Reward-model max sequence length | 4096 total tokens |
| PPO max prompt length | 3072 prompt tokens |
| PPO max generated response length | 512 new tokens |
| Evaluation max prompt length | 3072 prompt tokens |
| Evaluation max generated response length | 1024 new tokens |

The evaluation budget is larger than the PPO rollout budget: inference is not required to use the same response cap as training. A 3072-token prompt plus 1024 generated tokens remains within the 4096-token sequence length used for SFT and reward-model training.

RLHF training is substantially more expensive than inference because it must keep the policy, frozen reference, reward model, value head, token log-probabilities, masks, and rollout tensors in memory. The 512-token PPO response cap therefore controls training cost, while the 1024-token evaluation budget tests behavior over longer continuations. The reward model's 4096-token limit is a **total prompt-plus-response budget**, not permission to score a 4096-token response after an already long prompt.

## Why 4096 mattered

A token-length diagnostic showed that the earlier 1024-token SFT/RM cap was too small for HelpSteer3:

| Limit | Train SFT truncation | Train RM truncation | Validation SFT truncation | Validation RM truncation |
|---:|---:|---:|---:|---:|
| 1024 | 38.47% | 40.82% | 36.78% | 39.49% |
| 2048 | 15.48% | 16.47% | 13.51% | 14.87% |
| 3072 | 5.28% | 5.83% | 4.69% | 5.32% |
| 4096 | 0.83% | 1.00% | 0.68% | 0.89% |

At 4096 tokens, the training stages retain substantially more of each example.

## Training stages

### 1. SFT policy

```bash
python scripts/rlhf_train_sft_policy.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_sft.yaml
```

Final SFT configuration:

- base model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA rank: 16
- max length: 4096
- epochs: 2
- batch size: 6
- gradient accumulation: 3
- learning rate: `5e-6`
- output: `outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/`

The SFT stage teaches the policy to imitate the preferred HelpSteer3 response. It is the supervised alignment baseline and also the starting/reference policy for PPO. Given a rendered prompt `x` and preferred response `y`, the causal LM is trained on the concatenated sequence:

```text
[prompt tokens][assistant response tokens]
```

The prompt tokens provide context, but the loss is masked there and computed only on assistant response tokens:

```text
L_SFT(theta) = - sum_t log pi_theta(y_t | x, y_<t)
```

The 4096-token limit matters because HelpSteer3 contains both long prompts and long preferred responses. As the truncation table above shows, the 1024-token configuration truncated about 38% of chosen SFT sequences, while the final configuration truncated less than 1%.

### 2. Reward model

```bash
python scripts/rlhf_train_reward_model.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_reward.yaml
```

The reward model takes a complete prompt-response pair and returns a scalar:

```text
r_phi(prompt, response) -> real number
```

It is trained on HelpSteer3 chosen/rejected pairs with a Bradley-Terry logistic ranking loss:

```text
L_RM(phi) = - log sigmoid(r_phi(chosen) - r_phi(rejected))
```

The objective rewards a positive margin between the chosen and rejected responses. The final reward model was trained in two one-epoch runs: first from Qwen, then resumed from the best checkpoint for a second epoch.

Final reward-model result:

| Metric | Value |
|---|---:|
| validation pairs | 1917 |
| validation accuracy | 71.62% |
| validation loss | 0.9734 |
| average reward margin | 0.9094 |
| code accuracy | 74.88% |
| general accuracy | 71.01% |
| STEM accuracy | 63.37% |
| multilingual accuracy | 75.15% |

The model is useful as a PPO training signal, but it is not a perfect judge. Reward-model-based win rates are proxy metrics rather than ground-truth human preferences.

The scalar output is not calibrated to an external human score, so its sign is not intrinsically meaningful. Adding a constant to every reward would leave all pairwise preferences unchanged; a response scored `-3.5` can still be preferred to one scored `-5.0` for the same prompt. The final histograms are roughly bell-shaped because most prompt-response pairs fall within the model's usual scoring range while unusually preferred or dispreferred examples form the tails. The important quantities are margins and rankings. PPO training clips rewards for optimization stability, whereas the evaluation plots report raw reward-model outputs.

### 3. PPO post-training

```bash
python scripts/rlhf_train_ppo.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_ppo.yaml
```

In language-model PPO, each generated token is an action and the complete assistant response is a rollout. For prompt `x`, the current policy samples:

```text
y ~ pi_theta(. | x)
```

The reward model scores the response, while a KL penalty discourages movement away from the frozen SFT reference:

```text
R_total = R_model - beta * KL(pi_theta || pi_ref)
```

Only response-token log-probabilities participate in the policy-gradient loss. Prompt tokens are context rather than sampled actions. For each response token, PPO compares the updated policy with the policy that generated the rollout:

```text
ratio_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
```

It then optimizes the clipped surrogate:

```text
L_clip(theta) = E[min(ratio_t A_t, clip(ratio_t, 1-eps, 1+eps) A_t)]
```

Clipping limits how much reward-model incentive can be extracted from the same sampled token batch. This is the language-model analogue of PPO's trust-region motivation.

Final PPO configuration:

- initial policy: `outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/checkpoint_final`
- frozen reference: same SFT checkpoint
- reward model: `outputs/rlhf/qwen25_05b_helpsteer3_reward_4096_epoch2/checkpoint_best`
- max prompt length: 3072
- max new tokens: 512
- LoRA rank: 16
- total updates: 400 requested; 397 completed
- PPO epochs per rollout batch: 1
- learning rate: `3e-7`
- clip range: 0.06
- KL coefficient: initialized at 0.18, with minimum 0.14 and maximum 3.0
- output: `outputs/rlhf/qwen25_05b_helpsteer3_ppo_4096_epoch2_long512/`

The frozen SFT reference is essential: without that anchor, reward optimization can drift into empty answers, repetition, multilingual drift, or high-reward nonsense. These settings were conservative enough to avoid collapse, but that conservatism may also have limited improvement. Empty-rate stayed at zero, response lengths remained long, and the checkpoint loaded correctly in the final suite evaluation. However, PPO did not outperform the Base or SFT policies overall.

## Primary 1024-token policy-suite evaluation

Instead of running three separate pairwise evaluations, the final evaluator generates Base, SFT, and PPO responses once per prompt, scores all three with the same reward model, and derives all pairwise comparisons from the same table.

```bash
python scripts/rlhf_evaluate_policy_suite.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_eval_suite.yaml
```

Final evaluation:

- split: HelpSteer3 validation
- examples: 2017
- prompt budget: 3072 tokens
- generation budget: 1024 tokens
- policies: Base Qwen, SFT-4096, PPO-4096-epoch2-update400
- output: `outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/`

The earlier 512-token evaluation is preserved in the chronological [`rlhf_experiments.md`](docs/rlhf_experiments.md) log. The 1024-token suite is the primary result because it reduces cap hits from roughly 30% to 8-12% and gives a better view of complete response behavior. A larger inference budget is not automatically better, however: longer continuations also expose hallucination, irrelevance, repetition, and failure to stop.

The 512-token and 1024-token suites should not be presented as a controlled one-variable ablation because evaluation batch size also changed. The final 3072-token prompt plus 1024-token response remains within the reward model's 4096-token total sequence budget; a 4096-token response following a long prompt would not.

### Overall three-way winner counts

| Policy | Wins | Win rate | Mean reward | Median response tokens | Cap-hit rate | Empty rate |
|---|---:|---:|---:|---:|---:|---:|
| Base | 978 | 48.49% | -3.3634 | 334 | 8.08% | 0.00% |
| SFT-4096 | 475 | 23.55% | -3.6114 | 360 | 10.16% | 0.00% |
| PPO-4096 | 467 | 23.15% | -3.5771 | 363 | 11.60% | 0.00% |
| Tie | 97 | 4.81% | — | — | — | — |

### Pairwise comparisons

| Comparison | Left wins | Right wins | Ties | Right win rate | Mean right-left reward delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 1215 | 763 | 39 | 37.83% | -0.2480 |
| Base vs PPO | 1190 | 785 | 42 | 38.92% | -0.2137 |
| SFT vs PPO | 963 | 892 | 162 | 44.22% | +0.0343 |

PPO's reward margins are asymmetric. Its 785 wins over Base average `+1.6210`, while its 1190 losses average `-1.4315`. Against SFT, PPO's winning margins also exceed its losing margins on average, producing the slightly positive aggregate delta despite fewer wins. This does not overturn the win-rate result: Base remains the strongest policy under the learned reward model.

The qualitative audit adds an important limitation. At 1024 tokens, more than 25% of word-level 4-grams are repeated in 7.49% of Base, 13.78% of SFT, and 16.11% of PPO responses. Several high-reward PPO outputs are visibly broken loops, fabricated citations, or irrelevant continuations. The reward model also occasionally assigns very low scores to comparatively useful responses. See [`rlhf_qualitative_audit.md`](docs/rlhf_qualitative_audit.md) for the evidence and full selected responses.

## Why PPO did not dominate

Several constraints plausibly explain why training remained stable without producing a clear aggregate improvement:

1. **The Base model is already instruction-tuned.** Qwen2.5-0.5B-Instruct is a post-trained assistant rather than an unaligned pretrained language model.
2. **The reward model is imperfect.** Its 71.62% validation accuracy provides a useful signal but is not equivalent to a human judge.
3. **The policy is small.** A 0.5B model has limited capacity to improve preference behavior while preserving broad capability.
4. **PPO is deliberately conservative.** Strong KL anchoring and a low learning rate reduce collapse risk but also constrain improvement.
5. **PPO induces distribution shift.** The reward model learns from chosen/rejected dataset responses, while PPO optimizes newly generated responses that may fall outside that training distribution.
6. **Long generations increase variance.** A 1024-token evaluation allows useful detail but also more reward hacking, noisy continuation, hallucination, and irrelevance than the 512-token PPO rollout horizon.
7. **Stopping behavior remains weak.** PPO exceeds the 25% repeated word-level 4-gram threshold on 16.11% of primary-evaluation responses, versus 7.49% for Base.
8. **The reward model has blind spots.** Some high-scoring PPO responses are repetition loops or prompt restatements, while some comparatively relevant responses receive very low scores.

Together, these factors are consistent with a successful systems experiment and stable optimization, but not with a claim of general policy improvement.

## Scope and findings

The implementation includes:

- an end-to-end RLHF pipeline;
- long-context SFT and reward-model training;
- pairwise reward modeling with domain-level diagnostics;
- token-level PPO with KL anchoring to a frozen SFT reference;
- observed RLHF failure modes and fixes: gibberish drift, vulgar output, EOS/blank collapse, wrong checkpoint loading, and non-resumable long evaluation;
- full-validation policy-suite evaluation;
- curation tooling to inspect both wins and failures.

The results do **not** show that a 0.5B PPO adapter beats Qwen2.5-Instruct at scale. The reward model provides a usable training signal and PPO remains stable, but Base wins most comparisons and the longer audit exposes repetition and judge failures. The implementation, reproducible diagnostics, and concrete failure analysis are the main outcomes.

## Main artifacts

| Artifact | Purpose |
|---|---|
| `configs/rlhf/qwen25_05b_helpsteer3_sft.yaml` | final SFT configuration |
| `configs/rlhf/qwen25_05b_helpsteer3_reward.yaml` | final reward-model configuration |
| `configs/rlhf/qwen25_05b_helpsteer3_ppo.yaml` | final PPO configuration |
| `configs/rlhf/qwen25_05b_helpsteer3_eval_suite.yaml` | final policy-suite evaluation configuration |
| `experiments/baselines/qwen25_05b_helpsteer3_ppo_long512/` | frozen pre-restructure baseline and exact resolved configs |
| `outputs/rlhf/length_diagnostics/` | token-length and truncation study |
| `outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/` | SFT checkpoint and metrics |
| `outputs/rlhf/qwen25_05b_helpsteer3_reward_4096_epoch2/` | final reward model |
| `outputs/rlhf/qwen25_05b_helpsteer3_ppo_4096_epoch2_long512/` | final PPO checkpoint |
| `outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/` | primary 1024-token Base/SFT/PPO evaluation |
| `outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400/` | archived 512-token evaluation baseline |
| `scripts/rlhf_audit_policy_suite.py` | repetition, reward-margin, and curation audit |
| `configs/rlhf/qwen25_05b_helpsteer3_eval1024_curation.json` | reviewed-example manifest |

## Recommended reading order

- [`rlhf_experiments.md`](docs/rlhf_experiments.md): experiment timeline, failed runs, archived 512-token baseline, and primary 1024-token evaluation.
- [`rlhf_qualitative_audit.md`](docs/rlhf_qualitative_audit.md): manual analysis of useful responses, failures, and reward-model mismatches.
- [`rlhf_curation_guide.md`](docs/rlhf_curation_guide.md): how to reproduce and extend the qualitative review.
- [`rlhf_future_work.md`](docs/rlhf_future_work.md): a prioritized research program based on the observed limitations.
- [`experiment_tracking.md`](docs/experiment_tracking.md): run manifests, experiment intent, and exact config comparison.
- `notebooks/analyzing_full_eval_results.ipynb`: summary analysis notebook for the final policy-suite outputs.
- `notebooks/rlhf_full_eval_and_curation.ipynb`: interactive example browser and curation notebook.
