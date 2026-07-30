from types import SimpleNamespace

import torch

from rlhf.trl_train_ppo import (
    _RewardDiagnostics,
    _patch_trl_memory_efficient_rollout,
    build_balanced_rollout_plan,
    latest_exact_ppo_checkpoint,
    memory_efficient_batch_generation,
    repeated_ngram_fraction,
    shape_terminal_reward,
)


def _selective_log_softmax(logits, token_ids):
    return (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, token_ids.unsqueeze(-1))
        .squeeze(-1)
    )


class _DummyPolicy(torch.nn.Module):
    def __init__(self, vocabulary_size=11, fail_above=None):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.fail_above = fail_above
        self.generate_calls = []
        self.forward_calls = []

    def generate(self, input_ids, attention_mask, generation_config):
        response = torch.randint(
            1,
            self.vocabulary_size,
            (input_ids.shape[0], int(generation_config.max_new_tokens)),
            device=input_ids.device,
        )
        self.generate_calls.append(
            {
                "batch_size": input_ids.shape[0],
                "training": self.training,
                "use_cache": generation_config.use_cache,
                "output_scores": generation_config.output_scores,
                "return_dict_in_generate": generation_config.return_dict_in_generate,
            }
        )
        if self.fail_above is not None and input_ids.shape[0] > self.fail_above:
            raise torch.OutOfMemoryError("synthetic CUDA OOM")
        return torch.cat((input_ids, response), dim=1)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        return_dict,
        output_hidden_states,
        use_cache,
        logits_to_keep,
    ):
        self.forward_calls.append(
            {
                "batch_size": input_ids.shape[0],
                "logits_to_keep": logits_to_keep,
                "use_cache": use_cache,
            }
        )
        retained = input_ids[:, -int(logits_to_keep) :].float()
        vocabulary = torch.arange(self.vocabulary_size, device=input_ids.device).view(
            1, 1, -1
        )
        logits = -(vocabulary - retained.unsqueeze(-1)).square()
        return SimpleNamespace(logits=logits)


def _generation_config():
    return SimpleNamespace(
        max_new_tokens=3,
        temperature=0.7,
        eos_token_id=9,
        forced_eos_token_id=9,
        use_cache=False,
        output_scores=True,
        return_dict_in_generate=True,
    )


def test_balanced_rollout_plan_is_deterministic_and_balanced_per_update():
    domains = ["code", "general", "stem", "multilingual"]
    values = [domain for domain in domains for _ in range(20)]
    plan, report = build_balanced_rollout_plan(
        values,
        total_updates=3,
        batch_size=8,
        seed=839,
        domains=domains,
    )
    repeated_plan, repeated_report = build_balanced_rollout_plan(
        values,
        total_updates=3,
        batch_size=8,
        seed=839,
        domains=domains,
    )

    assert plan == repeated_plan
    assert report["plan_sha256"] == repeated_report["plan_sha256"]
    for update in range(3):
        batch = plan[update * 8 : (update + 1) * 8]
        batch_domains = [values[index] for index in batch]
        assert {domain: batch_domains.count(domain) for domain in domains} == {
            domain: 2 for domain in domains
        }


def test_balanced_plan_segment_slice_matches_uninterrupted_plan():
    domains = ["code", "general", "stem", "multilingual"]
    values = [domain for domain in domains for _ in range(12)]
    plan, _ = build_balanced_rollout_plan(
        values,
        total_updates=5,
        batch_size=8,
        seed=41,
        domains=domains,
    )
    assert plan[2 * 8 : 5 * 8] == plan[16:]


def test_repetition_fraction_and_smooth_penalty():
    tokens = [1, 2, 3, 4, 1, 2, 3, 4]
    repetition = repeated_ngram_fraction(tokens, ngram_size=4)
    assert repetition > 0.0

    shaped, penalty, clipped_low, clipped_high = shape_terminal_reward(
        1.5,
        has_eos=True,
        repetition_fraction=0.6,
        lower_bound=-2.0,
        upper_bound=2.0,
        repetition_threshold=0.2,
        repetition_penalty_scale=1.0,
    )
    assert -2.0 <= shaped < 1.5
    assert penalty > 0.0
    assert not clipped_low
    assert not clipped_high


def test_missing_eos_replaces_reward_with_scale_aware_lower_bound():
    shaped, penalty, clipped_low, clipped_high = shape_terminal_reward(
        99.0,
        has_eos=False,
        repetition_fraction=1.0,
        lower_bound=-2.5,
        upper_bound=1.75,
        repetition_threshold=0.1,
        repetition_penalty_scale=1.0,
    )
    assert shaped == -2.5
    assert penalty == 0.0
    assert not clipped_low
    assert clipped_high


def test_latest_exact_checkpoint_ignores_newer_partial_save(tmp_path):
    exact = tmp_path / "checkpoint-25"
    exact.mkdir()
    (exact / "trainer_state.json").write_text("{}", encoding="utf-8")
    (exact / "exact_resume_complete.json").write_text("{}", encoding="utf-8")
    partial = tmp_path / "checkpoint-50"
    partial.mkdir()
    (partial / "trainer_state.json").write_text("{}", encoding="utf-8")

    assert latest_exact_ppo_checkpoint(tmp_path) == exact


def test_reward_diagnostics_include_domain_guardrail_metrics():
    diagnostics = _RewardDiagnostics(("code", "general"))
    for domain, clipped in (("code", True), ("general", False)):
        diagnostics.add(
            {
                "domain": domain,
                "raw_reward": 1.0,
                "shaped_reward": 0.5,
                "clipped_low": False,
                "clipped_high": clipped,
                "has_eos": True,
                "response_length": 20,
                "repetition_fraction": 0.1,
                "repetition_penalty": 0.0,
                "repetition_triggered": False,
            }
        )
    metrics = diagnostics.consume()
    assert metrics["domain/code/reward_clipped_high_fraction"] == 1.0
    assert metrics["domain/general/reward_clipped_high_fraction"] == 0.0
    assert metrics["rollout/eos_rate"] == 1.0


def test_memory_efficient_rollout_retains_only_selected_logprobs():
    torch.manual_seed(17)
    policy = _DummyPolicy()
    policy.train()
    queries = torch.tensor(
        [[0, 1, 2], [0, 3, 4], [5, 6, 7], [8, 9, 10]],
        dtype=torch.long,
    )
    query_responses, old_logprobs, profile = memory_efficient_batch_generation(
        policy,
        queries,
        pad_token_id=0,
        generation_config=_generation_config(),
        generation_batch_candidates=[2],
        logprob_batch_candidates=[2],
        selective_log_softmax=_selective_log_softmax,
    )

    assert query_responses.shape == (4, 6)
    assert old_logprobs.shape == (4, 3)
    assert old_logprobs.numel() == 12
    retained = query_responses[:, -4:].float()
    vocabulary = torch.arange(policy.vocabulary_size).view(1, 1, -1)
    expected_logits = -(vocabulary - retained.unsqueeze(-1)).square()
    expected = _selective_log_softmax(
        expected_logits[:, :-1] / 0.7,
        query_responses[:, 3:],
    )
    assert torch.allclose(old_logprobs, expected)
    assert profile["rollout/compact_logprob_elements"] == 12
    assert profile["rollout/generation_batch_size"] == 2
    assert profile["rollout/logprob_forward_batch_size"] == 2
    assert policy.training
    assert all(not row["training"] for row in policy.generate_calls)
    assert all(row["use_cache"] for row in policy.generate_calls)
    assert all(not row["output_scores"] for row in policy.generate_calls)
    assert all(not row["return_dict_in_generate"] for row in policy.generate_calls)
    assert all(row["logits_to_keep"] == 4 for row in policy.forward_calls)
    assert all(not row["use_cache"] for row in policy.forward_calls)


def test_generation_oom_fallback_restores_rng_before_smaller_chunk_retry():
    queries = torch.tensor(
        [[0, 1, 2], [0, 3, 4], [5, 6, 7], [8, 9, 10]],
        dtype=torch.long,
    )
    torch.manual_seed(29)
    fallback_policy = _DummyPolicy(fail_above=2)
    fallback_responses, fallback_logprobs, fallback_profile = (
        memory_efficient_batch_generation(
            fallback_policy,
            queries,
            pad_token_id=0,
            generation_config=_generation_config(),
            generation_batch_candidates=[4, 2],
            logprob_batch_candidates=[2],
            selective_log_softmax=_selective_log_softmax,
        )
    )

    torch.manual_seed(29)
    direct_policy = _DummyPolicy()
    direct_responses, direct_logprobs, _ = memory_efficient_batch_generation(
        direct_policy,
        queries,
        pad_token_id=0,
        generation_config=_generation_config(),
        generation_batch_candidates=[2],
        logprob_batch_candidates=[2],
        selective_log_softmax=_selective_log_softmax,
    )

    assert torch.equal(fallback_responses, direct_responses)
    assert torch.equal(fallback_logprobs, direct_logprobs)
    assert fallback_profile["rollout/generation_batch_size"] == 2
    assert fallback_profile["rollout/generation_oom_fallbacks"] == 1


def test_trl_selective_logsoftmax_patch_accepts_compact_rollout_tensor():
    module = SimpleNamespace(
        batch_generation=lambda *args, **kwargs: None,
        selective_log_softmax=_selective_log_softmax,
    )
    _patch_trl_memory_efficient_rollout(
        module,
        {
            "generation_batch_size_candidates": [2],
            "logprob_batch_size_candidates": [1],
            "enable_generation_cache": True,
            "require_logits_to_keep": True,
        },
    )
    compact = torch.randn(2, 3)
    token_ids = torch.ones((2, 3), dtype=torch.long)
    assert module.selective_log_softmax(compact, token_ids) is compact
