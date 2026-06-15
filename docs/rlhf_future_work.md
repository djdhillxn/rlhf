# RLHF Future Work

The current system establishes that the complete training and evaluation path works: long-context supervised fine-tuning, pairwise reward modeling, KL-controlled token-level PPO, resumable policy-suite generation, and qualitative auditing all run on real HelpSteer3 data. The negative and mixed results are therefore useful design evidence. They identify the parts of the alignment stack that now limit performance and suggest a sequence of experiments that is more informative than simply increasing the number of PPO updates.

## Establish a controlled evaluation protocol

The first priority is to make evaluation differences attributable to one intervention at a time. The 512-token and 1024-token suites also changed batch size, and most generated responses changed even when the earlier response had not reached its cap. Future evaluations should hold the model checkpoint, software environment, precision, batch size, prompt order, decoding configuration, and evaluator implementation fixed while varying only one parameter. A token-budget study should run 256, 512, 768, and 1024 new-token limits under the same conditions and record whether each longer response is an exact continuation of the shorter response.

Greedy decoding is useful for reproducibility, but it describes only one trajectory from each policy. A fuller evaluation should include repeated sampled generations with fixed seeds. This would separate policy quality from the luck of a single continuation and reveal whether PPO changes the variance of responses as well as their mean reward. Confidence intervals should be reported for overall and domain-level win rates, preferably through paired bootstrap resampling because every policy answers the same prompts.

Human review should become a first-class evaluation signal. A practical design would sample prompts by domain, reward margin, response length, and repetition risk, then ask reviewers to compare Base, SFT, and PPO responses without seeing policy labels or reward scores. Review dimensions should include correctness, relevance, completeness, clarity, harmlessness, and unnecessary verbosity. Inter-rater agreement and adjudication rates would indicate which prompts are genuinely ambiguous. Even a few hundred carefully reviewed comparisons would be more valuable than treating all reward-model decisions as ground truth.

Length should be controlled explicitly. Longer responses have more opportunities to provide detail, but also more opportunities to hallucinate, repeat, or accumulate reward-model features. Evaluations should report reward per response-length bucket, win rates among responses of comparable length, and the relationship between reward delta and token delta. A judge that systematically prefers formatting or verbosity can then be distinguished from one that recognizes substantive improvements.

## Improve reward-model reliability

The reward model is the central bottleneck exposed by the qualitative audit. It reaches 71.62% validation pairwise accuracy, but several extreme scores contradict obvious response quality. The next reward-model dataset should include hard negatives drawn from the policy itself: repeated loops, prompt restatements, fabricated citations, irrelevant continuations, malformed code, and confident scientific errors. These examples are more representative of PPO's optimization distribution than random rejected HelpSteer3 answers.

Training should become multi-objective rather than relying on one scalar preference target to capture every desirable property. One approach is to add auxiliary heads or separate judges for relevance, factuality, repetition, safety, and style, then combine their outputs with transparent weights. Another is to retain a general preference reward while applying deterministic penalties for measurable defects such as repeated n-grams, missing EOS, empty responses, language mismatch, or failure to answer a direct question. Deterministic checks should not replace learned preference, but they can prevent the optimizer from exploiting known blind spots.

Reward uncertainty should also be modeled. Ensembles trained from different seeds or data splits can identify examples on which the judge is unstable. PPO rewards could be discounted when ensemble disagreement is high, while evaluation reports could separate high-confidence and low-confidence comparisons. Calibration studies should measure whether a reward margin of two points actually corresponds to a more reliable preference than a margin of one point. The current examples show that large margins are not automatically trustworthy.

The reward dataset should be rebalanced across domains and languages. STEM accuracy is materially lower than code, general, and multilingual accuracy, and the qualitative audit finds fabricated technical content that a generic preference model does not reliably reject. Domain-specific hard negatives, verified code execution, citation checks, and scientifically reviewed pairs would improve the signal. Multilingual evaluation should include native-speaker review because repetition and semantic drift are difficult to diagnose from English-centric lexical metrics.

Long-response training deserves its own curriculum. The reward model was trained with a 4096-token total sequence limit, but that does not guarantee consistent judgment at every position or length. Training batches could deliberately balance short, medium, and long responses, including pairs where the first 300 tokens are similar but one response later degrades. Such examples would teach the model that a strong opening does not excuse a repetitive or incorrect ending.

## Strengthen supervised fine-tuning

SFT establishes both the PPO initialization and the frozen reference, so weaknesses in SFT constrain every later stage. A future SFT dataset should be filtered for factuality, coherent stopping, and response relevance rather than using preferred labels alone. Duplicate or near-duplicate answers should be removed, and examples with broken markup, repeated text, or unsupported citations should be excluded or repaired. A smaller clean dataset may provide a better reference policy than a larger noisy one.

Data composition should match intended use. The current validation set spans code, general, STEM, and multilingual prompts, but performance differs substantially by domain. Stratified sampling and per-domain loss monitoring would prevent high-volume categories from dominating training. For code, executable examples and unit-tested answers are preferable to preference labels based only on prose quality. For STEM, verified derivations and citation-grounded responses are necessary. Multilingual examples should preserve language consistency and include culturally natural answers rather than translations alone.

Sequence-length curriculum is another promising direction. Training could begin with shorter, high-quality examples and gradually introduce longer contexts and responses. This would let the model learn instruction following and stopping before it must manage long-range coherence. Packing strategies should preserve response-only loss masks and avoid combining unrelated examples in ways that create confusing attention patterns.

The project should also compare SFT checkpoints across epochs instead of assuming the final checkpoint is best. Validation perplexity on preferred responses is insufficient by itself; checkpoint selection should include response-level evaluation, repetition rate, factuality checks, and downstream preference win rate. The 1024-token suite suggests that later behavior may degrade after an initially useful answer, a property that token-level loss does not measure directly.

## Redesign PPO around observed failures

Training PPO with 512-token rollouts was a reasonable compute and stability choice, but the 1024-token evaluation shows that behavior after token 512 matters. A direct 1024-token PPO run is worth testing only as a controlled experiment, not as an assumed improvement. It should begin from the same SFT checkpoint, use the same prompt set, and be compared with the 512-token PPO policy at matched update-token budgets. Otherwise, twice-longer rollouts also change the amount of optimization data and the effective training horizon.

A curriculum may be more stable than beginning with 1024-token rollouts. PPO could start at 256 or 512 tokens, then increase the response allowance after reward, KL, EOS rate, and repetition metrics stabilize. The curriculum should include explicit stopping objectives so that the policy is not merely given more room to continue. Missing-EOS penalties, length-aware terminal rewards, and penalties for repeated n-grams can be introduced gradually and monitored for unintended short-answer collapse.

The reward should distinguish the useful prefix from the degraded suffix. A single terminal score assigns the same sequence-level outcome across all response tokens, even when quality changes late in the generation. Process rewards, segment-level scoring, or periodic reward-model evaluations could provide denser information. At minimum, the final response can be scored both in full and at intermediate truncation points. If the 256-token prefix scores well but the 1024-token answer scores poorly, the optimizer should learn that continuing was harmful.

KL control can also become more adaptive. The current coefficient protects against global drift, but different prompts and response positions may require different constraints. Per-token KL diagnostics, target-KL controllers, and domain-specific monitoring could reveal whether repetition begins after the policy moves away from the reference late in a response. A stronger late-token KL penalty or a reference-based stopping criterion may preserve useful early improvements while preventing long repetitive tails.

Advantage estimation and reward normalization should be revisited. Prompt difficulty strongly affects raw reward, so group-relative comparisons among multiple responses to the same prompt can provide a cleaner learning signal than normalizing across unrelated prompts. Generating several candidates per prompt and centering rewards within each group would train the policy to prefer better continuations for the same context. RLOO- or GRPO-style baselines may therefore be useful comparisons to the current value-head PPO implementation.

Checkpoint selection should use a multi-metric stopping rule. Reward alone is unsafe because the judge can reward repetition. Candidate checkpoints should be compared on reward-model win rate, KL, EOS rate, empty rate, repetition, length, human preference, and domain-specific correctness. Early stopping should trigger when reward improves while qualitative safety metrics deteriorate.

## Compare alternative preference objectives

PPO is valuable here because it directly connects the project to trust-region policy optimization, but it should not be the only alignment baseline. Direct Preference Optimization can train from the same chosen/rejected pairs without online rollouts or a learned value function. IPO, KTO, ORPO, and related objectives offer different robustness and data requirements. A controlled comparison would clarify whether the mixed result comes from the reward model, online RL optimization, model capacity, or the preference data itself.

Online methods remain relevant when the policy produces errors that are absent from the original preference dataset. An iterative design could alternate between generation, human or high-quality judge labeling, reward-model updates, and policy updates. This would turn the current static HelpSteer3 setup into an active-learning loop focused on the policy's actual failure distribution.

## Improve decoding and stopping

Inference settings should be evaluated as part of the policy, not treated as an afterthought. The current suite uses greedy decoding with no repetition penalty and no no-repeat n-gram constraint. This cleanly exposes model behavior, but deployment-oriented evaluation should compare modest repetition penalties, no-repeat constraints, temperature and top-p sampling, and task-aware stopping. These controls can reduce visible degeneration even before retraining, although they may also suppress legitimate repeated syntax in code or structured text.

Dynamic token budgets are preferable to one universal cap. Short classification, title-generation, and extraction prompts rarely need 1024 tokens, while code explanations and long-form synthesis may. A length predictor or prompt-class rule could assign response budgets by task. The evaluator should report both quality and compute, since a policy that needs twice as many tokens for the same usefulness is not necessarily better.

Stopping quality should be measured directly through EOS rate, cap-hit rate, late-response repetition, and the reward change between a response and its shorter prefixes. This would make it possible to distinguish models that produce complete long answers from models that fail to stop.

## Add factuality, execution, and safety evaluators

General preference scoring is insufficient for domains where correctness can be checked. Code responses should be parsed, linted, and executed in isolated tests when possible. SQL can be run against temporary schemas, shell snippets can be statically inspected, and Python examples can be unit tested. These checks turn subjective-looking code evaluation into measurable behavior.

Scientific responses need equation, unit, and citation verification. The chemistry and evolutionary-biology failures in the current audit are not minor stylistic problems; they can mislead users. Retrieval-augmented evaluation, citation existence checks, and specialist model judges should be used for high-risk STEM prompts. When the model lacks evidence, abstention or uncertainty should be rewarded over confident fabrication.

Safety evaluation should include structured test sets rather than only lexical screening. The project should measure self-harm handling, medical advice, harassment, sexual content, privacy, and dangerous procedural assistance. False refusals must be tracked alongside unsafe compliance, because alignment that refuses benign prompts is also a quality failure.

## Scale model and systems capacity deliberately

The 0.5B model makes the project inexpensive and debuggable, but its capacity is a real limitation. Repeating the pipeline at 1.5B or 3B scale would test whether the same reward and PPO design becomes more effective when the policy has stronger base capabilities. The comparison should keep data and evaluation fixed so that improvements can be attributed to scale.

Larger models require systems changes. Distributed or sharded training, activation checkpointing, Flash Attention, efficient sequence packing, and quantized frozen reference or reward models can reduce memory pressure. Rollout generation can be separated from optimization and accelerated with inference-oriented runtimes. Asynchronous generation would keep the training GPU occupied while new response batches are prepared.

Scaling the reward model independently may be more valuable than scaling the policy first. A stronger judge can improve both PPO training and evaluation, whereas a larger policy optimized against the same flawed reward may exploit it more effectively. Experiments should therefore compare policy scale and reward-model scale as separate axes.

## Reproducibility and artifact governance

Every evaluation should preserve its resolved configuration, environment versions, checkpoint hashes, prompt-order hash, generation signature, and source revision. The new resume fingerprint is a start because it prevents stale cached responses from being silently reused under changed settings. The same principle should extend to final summaries and curation artifacts.

Generated reports should be reproducible from lightweight commands. The audit script and curation manifest provide a model: raw response tables remain in `outputs/`, while selected examples, metrics, and interpretation are linked from the documentation. Future runs should use versioned directories rather than overwrite prior evidence.

## A practical next sequence

The highest-value next experiment is not a larger PPO run. It is a controlled re-evaluation at 512 and 1024 tokens with identical batch size and software, followed by blinded human review of a stratified sample. That establishes whether the longer budget itself helps and gives a calibration set for the reward model.

The second step is to retrain the reward model with hard negatives from the current audit, especially repeated loops, prompt restatements, irrelevant continuations, fabricated citations, and unsafe scientific answers. The revised judge should be validated specifically on those cases before it is used for another policy update.

The third step is to compare a conservative PPO rerun, a length-curriculum PPO run, and a direct preference baseline from the same SFT checkpoint. Each should be selected using a multi-metric evaluation that includes human preference and repetition, not reward alone. This sequence targets the observed bottlenecks and would produce substantially stronger evidence than simply increasing training time.
