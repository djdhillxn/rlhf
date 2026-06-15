# RLHF Qualitative Audit

Aggregate win rates are not sufficient to evaluate this run. The reward model is imperfect, long generations can repeat or drift, and a large reward margin can correspond either to a real improvement or to a scoring failure. This audit combines a scan of all 2017 evaluation rows with manual inspection of selected extremes.

The primary data comes from:

```text
outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/
```

The full selected responses are available in:

[`selected_qualitative_examples.md`](../outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/selected_qualitative_examples.md)

## Audit method

Every Base, SFT, and PPO response was checked for response length, cap hits, reward margins, repeated word-level 4-grams, and a small lexical list of sensitive terms. Candidate tables were then produced for low-repetition PPO wins, strong PPO losses, severe repetition, and cases where a high reward co-occurs with repetition.

The automated scan is reproducible with:

```bash
python scripts/rlhf_audit_policy_suite.py \
  --eval-dir outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024 \
  --baseline-dir outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400 \
  --selection-file configs/rlhf/qwen25_05b_helpsteer3_eval1024_curation.json
```

The repetition metric is intentionally simple. A technical answer can legitimately repeat terminology, while a pathological loop may evade a word-level metric by changing punctuation or one token at a time. The lexical scan is also not a safety classifier. These checks narrow the review surface; they do not replace human judgment.

## Full-suite findings

| Policy | Cap-hit rate | Responses with >25% repeated 4-grams | Responses with >50% repeated 4-grams |
|---|---:|---:|---:|
| Base | 8.08% | 151 (7.49%) | 60 (2.97%) |
| SFT | 10.16% | 278 (13.78%) | 134 (6.64%) |
| PPO | 11.60% | 325 (16.11%) | 156 (7.73%) |

The 1024-token budget solves much of the truncation problem, but it does not solve stopping behavior. SFT and PPO both repeat more often than Base, and PPO has the highest measured repetition rate. This is consistent with a policy that can produce useful longer answers but has not learned a reliable stopping criterion for every prompt.

The reward model catches some extreme loops. Index 86 repeats essentially one phrase for the entire PPO response and receives a very low reward. It misses other cases. Thirty-six PPO responses combine a reward above `2.0` with either more than 25% repeated 4-grams or a repeated 4-gram appearing at least five times. Several of the largest PPO reward margins belong to this group.

The small sensitive-term scan found two PPO responses containing a listed term, both in prompts or discussions involving suicide. It did not find the earlier vulgar/debug vocabulary that appeared during unstable preliminary runs. That is encouraging but limited: absence from a short term list is not evidence of broad safety.

## Qualified improvements

These examples show local improvements, not a globally superior policy.

### Index 2: PostgreSQL tutorial

PPO receives a large reward advantage over Base and SFT and organizes the answer from introductory SQL through more advanced database concepts. Base includes a plainly false account of PostgreSQL's origin. PPO is more coherent, but it reaches 1024 tokens and should still be fact-checked before publication.

### Index 1210: ambiguous distress prompt

Base gives a bare refusal. SFT and PPO interpret the prompt as possible emotional distress and respond supportively. PPO offers somewhat more actionable guidance without becoming graphic or punitive. This is one of the cleaner examples of alignment changing response style in a useful direction.

### Index 1158: conversational engagement

For the prompt “I love crows,” Base begins with a generic statement about lacking emotions. SFT and PPO engage with the topic directly. PPO is warmer and more expansive, although several biological details are overstated. The improvement is conversational rather than factual.

### Index 0: JSON to React

All three policies provide usable React examples. Base receives the highest reward, while PPO modestly beats SFT. This is a useful near-tie because it shows that the policies can differ in structure and explanation without yielding a decisive alignment result.

### Index 418: constrained scene writing

PPO includes more of the named characters and requested setup than the alternatives. It still confuses family roles and scene logic. The example demonstrates improved constraint coverage alongside the capacity limits of the 0.5B model.

### Index 1511: data-cleaning tutorial

PPO covers several common data-cleaning operations with Python snippets and receives a strong reward. Some method descriptions and code choices remain imprecise. It is a better candidate for discussing breadth and formatting than for claiming technical correctness.

## Factual and behavioral failures

### Index 1970: fabricated scientific literature

The prompt asks for studies questioning whether character displacement causes speciation. PPO invents article titles, authors, dates, and an elephant case study. This is a serious factual failure, especially because the answer presents the citations confidently.

### Index 1303: incorrect chemistry

PPO proposes invalid reaction equations and unsafe procedural steps for producing sodium titanate. The response confuses formulas, solubility, products, and acids. This example shows why reward-model evaluation alone is inadequate for scientific or safety-sensitive answers.

### Index 176: dialogue loop

PPO begins a plausible continuation and then repeats the same exchange until it reaches the generation cap. The reward model penalizes it, but the behavior confirms that a 1024-token allowance can expose failure modes hidden by a shorter cap.

### Index 1835: irrelevant continuation

The user asks for a short title related to “sky exchange with number.” PPO instead produces a long, repetitive Pixel 8 passage. This is both an instruction-following failure and a stopping failure.

## Reward-model mismatches

### Index 346: highest PPO reward, visibly broken response

PPO receives the maximum PPO reward in the suite, `10.7546`, and beats Base by `16.6331`. The response nevertheless enters a long repeated-phrase loop. This is the clearest evidence that reward magnitude cannot be used as a qualitative quality score without inspection.

### Index 1281: question repetition rewarded as an answer

For a Russian literary-analysis prompt, PPO repeatedly restates the question rather than explaining the symbolism of the moon. It receives a reward of `8.6657` and a PPO-minus-Base margin of `13.7555`.

### Index 1579: promises instead of completion

The German prompt asks for a dating profile using a specified formula. PPO repeatedly says that it will create the profile but never delivers the requested text. The reward remains strongly positive.

### Index 656: near-total repetition with positive reward

The Vietnamese response repeats one claim about AI speech rights for almost the entire generation. It receives a positive reward and beats both alternatives, another direct reward-model miss.

### Index 1548: the reward model prefers the worse response

Base fabricates a very long meeting-attendee list and repeats names until it reaches the cap. PPO produces a shorter and more relevant meeting summary, yet Base receives `11.0233` while PPO receives `-2.8518`. Reward-model errors therefore occur in both directions: it can reward broken PPO responses and reject comparatively useful ones.

## Interpretation

The latest evaluation is more informative than the 512-token run because only 8–12% of responses hit the cap instead of roughly 30%. It reveals that PPO can produce locally better answers and slightly larger winning margins, but it also reveals a higher rate of repetition and several severe reward-model mismatches.

The defensible conclusion is not that PPO improved the model overall. The conclusion is that the pipeline trains stably, changes behavior measurably, and creates both improvements and failures that can be diagnosed. The main technical bottleneck is now evaluation and reward quality rather than the ability to run PPO end to end.

## Generated artifacts

The audit script writes:

- `qualitative_audit_auto.md`
- `selected_qualitative_examples.md`
- `curation_qualified_ppo_candidates.csv`
- `curation_strong_ppo_losses.csv`
- `curation_ppo_repetition_risks.csv`
- `curation_reward_model_mismatches.csv`
- `evaluation_comparison_with_baseline.csv`

The interactive notebooks use the same 1024-token evaluation directory:

- `notebooks/analyzing_full_eval_results.ipynb`
- `notebooks/rlhf_full_eval_and_curation.ipynb`
