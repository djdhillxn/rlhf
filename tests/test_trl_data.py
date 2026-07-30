from rlhf.trl_data import (
    build_dpo_records,
    build_reward_records,
    build_sft_records,
    fit_prompt_to_budget,
)
from rlhf.data import build_prompt_records


class FakeTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token_id = 0
    pad_token = "<pad>"
    chat_template = "fake"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 50 + 1 for char in str(text)]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return self(rendered)["input_ids"] if tokenize else rendered

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(str(token_id) for token_id in token_ids)


def example(score=-2, prompt="question", chosen="better", rejected="worse"):
    return {
        "context": [{"role": "user", "content": prompt}],
        "response1": chosen,
        "response2": rejected,
        "preference": score,
        "domain": "general",
        "language": "English",
    }


def test_sft_preserves_response_and_eos():
    tokenizer = FakeTokenizer()
    records, stats = build_sft_records([example()], tokenizer, max_length=64)
    assert stats["kept"] == 1
    assert records[0]["input_ids"][-1] == tokenizer.eos_token_id
    assert records[0]["completion_mask"][-1] == 1
    assert len(records[0]["input_ids"]) <= 64


def test_reward_uses_one_shared_prompt_and_preserves_both_eos_tokens():
    tokenizer = FakeTokenizer()
    records, stats = build_reward_records([example()], tokenizer, max_length=64)
    assert stats["kept"] == 1
    assert records[0]["chosen_ids"][-1] == tokenizer.eos_token_id
    assert records[0]["rejected_ids"][-1] == tokenizer.eos_token_id
    prompt_length = records[0]["prompt_length"]
    assert (
        records[0]["chosen_ids"][:prompt_length]
        == records[0]["rejected_ids"][:prompt_length]
    )


def test_dpo_keeps_complete_responses_and_strength_metadata():
    tokenizer = FakeTokenizer()
    row = example()
    row["individual_preference"] = [
        {"score": -1, "reasoning": ""},
        {"score": -2, "reasoning": ""},
    ]
    records, stats = build_dpo_records([row], tokenizer, max_length=64)
    assert stats["source_pairs"] == 1
    assert len(records) == 1
    assert records[0]["chosen"].endswith(str(tokenizer.eos_token_id))
    assert records[0]["rejected"].endswith(str(tokenizer.eos_token_id))
    assert records[0]["preference_strength"] == 2
    assert records[0]["annotator_direction_agreement"] == 1.0


def test_dpo_strength_replication_is_explicit_and_train_only_ready():
    tokenizer = FakeTokenizer()
    records, stats = build_dpo_records(
        [example(score=-3)],
        tokenizer,
        max_length=64,
        preference_strength_weighting="linear_replication",
    )
    assert stats["source_pairs"] == 1
    assert stats["kept"] == 3
    assert [row["replica_index"] for row in records] == [0, 1, 2]


def test_prompt_truncation_drops_old_turns_before_token_fallback():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest"},
    ]
    ids, metadata = fit_prompt_to_budget(tokenizer, messages, max_prompt_tokens=40)
    assert len(ids) <= 40
    assert metadata["prompt_truncated"]
    assert metadata["dropped_turns"] > 0


def test_policy_eval_prompt_selection_can_be_domain_stratified():
    tokenizer = FakeTokenizer()
    rows = []
    for domain in ("code", "general", "stem", "multilingual"):
        for index in range(5):
            rows.append(
                {
                    "context": [
                        {
                            "role": "user",
                            "content": f"{domain} question {index}",
                        }
                    ],
                    "domain": domain,
                    "language": "English",
                }
            )
    records = build_prompt_records(
        rows,
        tokenizer,
        max_samples=8,
        seed=839,
        stratify_by_domain=True,
    )
    assert len(records) == 8
    assert {
        domain: sum(row["domain"] == domain for row in records)
        for domain in ("code", "general", "stem", "multilingual")
    } == {
        "code": 2,
        "general": 2,
        "stem": 2,
        "multilingual": 2,
    }
