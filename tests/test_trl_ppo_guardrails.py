from rlhf.trl_train_ppo import (
    _RewardDiagnostics,
    build_balanced_rollout_plan,
    latest_exact_ppo_checkpoint,
    repeated_ngram_fraction,
    shape_terminal_reward,
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
