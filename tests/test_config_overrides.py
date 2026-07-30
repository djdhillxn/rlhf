import pytest

from rlhf.config import ConfigError, DotDict, apply_overrides


def policy_config():
    return DotDict(
        {
            "model": {"name": "base"},
            "policies": [
                {"label": "base", "checkpoint_dir": None},
                {"label": "sft", "checkpoint_dir": "default-sft"},
                {"label": "ppo", "checkpoint_dir": "default-ppo"},
            ],
        }
    )


def test_apply_overrides_supports_bracketed_list_indices():
    updated = apply_overrides(
        policy_config(),
        {
            "policies[1].checkpoint_dir": "/tmp/sft",
            "policies[2].checkpoint_dir": "/tmp/ppo",
        },
    )
    assert updated["policies"][1]["checkpoint_dir"] == "/tmp/sft"
    assert updated["policies"][2]["checkpoint_dir"] == "/tmp/ppo"


def test_apply_overrides_supports_dotted_list_indices():
    updated = apply_overrides(
        policy_config(),
        {
            "policies.1.label": "sft_trl",
            "policies.2.label": "ppo_trl",
        },
    )
    assert updated["policies"][1]["label"] == "sft_trl"
    assert updated["policies"][2]["label"] == "ppo_trl"


def test_apply_overrides_rejects_out_of_range_list_index():
    with pytest.raises(ConfigError, match="length 3"):
        apply_overrides(policy_config(), {"policies[3].label": "invalid"})
