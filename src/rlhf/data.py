import json
import random
from pathlib import Path

from .formatting import (
    normalize_messages,
    render_prompt,
    render_prompt_with_response,
    strip_trailing_assistant,
)


PREFERENCE_SCORE_KEYS = (
    "preference_score",
    "overall_preference_score",
    "overall_preference",
    "preference",
    "score",
)
RESPONSE1_KEYS = (
    "response1",
    "response_1",
    "answer1",
    "answer_1",
    "output1",
    "output_1",
)
RESPONSE2_KEYS = (
    "response2",
    "response_2",
    "answer2",
    "answer_2",
    "output2",
    "output_2",
)
CONTEXT_KEYS = ("context", "messages", "conversation", "conversations", "prompt")


class PreferencePair:
    def __init__(
        self,
        prompt,
        chosen,
        rejected,
        chosen_text,
        rejected_text,
        margin,
        domain="unknown",
        language="unknown",
    ):
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected
        self.chosen_text = chosen_text
        self.rejected_text = rejected_text
        self.margin = margin
        self.domain = domain
        self.language = language


def _first_present(example, keys):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_helpsteer3_preference(split="train", *, streaming=False):
    """Load HelpSteer3 preference data with robust config fallbacks."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install RLHF extras first: pip install -r requirements-rlhf.txt"
        ) from exc

    errors = []
    for args in (("nvidia/HelpSteer3", "preference"), ("nvidia/HelpSteer3",)):
        try:
            return load_dataset(*args, split=split, streaming=streaming)
        except Exception as exc:  # pragma: no cover - depends on hub metadata
            errors.append(f"load_dataset{args!r}: {exc}")
    raise RuntimeError(
        "Could not load HelpSteer3 preference split. Tried:\n" + "\n".join(errors)
    )


def example_to_preference_pair(example, tokenizer):
    score = _to_float(_first_present(example, PREFERENCE_SCORE_KEYS))
    if score is None or score == 0.0:
        return None

    response1 = _first_present(example, RESPONSE1_KEYS)
    response2 = _first_present(example, RESPONSE2_KEYS)
    if not response1 or not response2:
        return None
    response1 = str(response1).strip()
    response2 = str(response2).strip()
    if not response1 or not response2:
        return None

    context = _first_present(example, CONTEXT_KEYS)
    messages = strip_trailing_assistant(normalize_messages(context))
    prompt = render_prompt(tokenizer, messages, add_generation_prompt=True)

    if score > 0:
        chosen, rejected = response2, response1
    else:
        chosen, rejected = response1, response2

    return PreferencePair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        chosen_text=render_prompt_with_response(tokenizer, messages, chosen),
        rejected_text=render_prompt_with_response(tokenizer, messages, rejected),
        margin=abs(float(score)),
        domain=str(example.get("domain", "unknown")),
        language=str(example.get("language", "unknown")),
    )


def build_preference_pairs(
    raw_dataset,
    tokenizer,
    *,
    max_samples=None,
    shuffle=False,
    seed=0,
):
    pairs = []
    for ex in raw_dataset:
        pair = example_to_preference_pair(dict(ex), tokenizer)
        if pair is not None:
            pairs.append(pair)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pairs)
    if max_samples is not None:
        pairs = pairs[: int(max_samples)]
    return pairs


def build_prompt_records(
    raw_dataset,
    tokenizer,
    *,
    max_samples=None,
    seed=0,
    shuffle=True,
):
    records = []
    for ex in raw_dataset:
        context = _first_present(dict(ex), CONTEXT_KEYS)
        messages = strip_trailing_assistant(normalize_messages(context))
        prompt = render_prompt(tokenizer, messages, add_generation_prompt=True)
        if prompt.strip():
            records.append(
                {
                    "prompt": prompt,
                    "domain": str(dict(ex).get("domain", "unknown")),
                    "language": str(dict(ex).get("language", "unknown")),
                }
            )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)
    if max_samples is not None:
        records = records[: int(max_samples)]
    return records


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def preference_pairs_to_dicts(pairs):
    return [pair.__dict__.copy() for pair in pairs]
