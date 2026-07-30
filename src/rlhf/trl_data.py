import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import (
    CONTEXT_KEYS,
    PREFERENCE_SCORE_KEYS,
    RESPONSE1_KEYS,
    RESPONSE2_KEYS,
    load_helpsteer3_preference,
)
from .formatting import normalize_messages, render_prompt, strip_trailing_assistant
from .trl_common import write_json


def _first_present(example, keys):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _preference_components(example):
    raw_score = _first_present(example, PREFERENCE_SCORE_KEYS)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    if score == 0:
        return None
    response1 = _first_present(example, RESPONSE1_KEYS)
    response2 = _first_present(example, RESPONSE2_KEYS)
    if response1 is None or response2 is None:
        return None
    response1 = str(response1).strip()
    response2 = str(response2).strip()
    if not response1 or not response2:
        return None
    context = _first_present(example, CONTEXT_KEYS)
    messages = strip_trailing_assistant(normalize_messages(context))
    if not messages:
        return None
    chosen, rejected = (response2, response1) if score > 0 else (response1, response2)
    return messages, chosen, rejected, abs(score)


def _encode_text(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _encode_completion(tokenizer, response):
    ids = _encode_text(tokenizer, response.strip())
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    if not ids or ids[-1] != eos_id:
        ids.append(int(eos_id))
    return ids


def _prompt_ids(tokenizer, messages):
    return _encode_text(
        tokenizer, render_prompt(tokenizer, messages, add_generation_prompt=True)
    )


def _drop_oldest_turn(messages):
    if len(messages) <= 1:
        return messages, False
    for idx in range(len(messages) - 1):
        if messages[idx].get("role") != "system":
            return messages[:idx] + messages[idx + 1 :], True
    return messages, False


def fit_prompt_to_budget(
    tokenizer,
    messages,
    max_prompt_tokens,
):
    """Drop old turns first, then left-truncate only as a final fallback."""
    if max_prompt_tokens <= 0:
        return [], {
            "prompt_truncated": True,
            "dropped_turns": len(messages),
            "token_fallback": True,
        }

    working = [dict(message) for message in messages]
    dropped_turns = 0
    ids = _prompt_ids(tokenizer, working)
    while len(ids) > max_prompt_tokens:
        working, removed = _drop_oldest_turn(working)
        if not removed:
            break
        dropped_turns += 1
        ids = _prompt_ids(tokenizer, working)

    token_fallback = len(ids) > max_prompt_tokens
    if token_fallback:
        ids = ids[-max_prompt_tokens:]
    return ids, {
        "prompt_truncated": bool(dropped_turns or token_fallback),
        "dropped_turns": dropped_turns,
        "token_fallback": token_fallback,
    }


def _example_id(messages, chosen, rejected):
    payload = json.dumps(
        [messages, chosen, rejected], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _metadata(
    example,
    messages,
    chosen,
    rejected,
    strength,
    fit,
):
    annotator_scores = []
    for vote in example.get("individual_preference") or []:
        if not isinstance(vote, dict):
            continue
        try:
            annotator_scores.append(float(vote.get("score")))
        except (TypeError, ValueError):
            continue
    raw_score = _first_present(example, PREFERENCE_SCORE_KEYS)
    try:
        direction = 1 if float(raw_score) > 0 else -1
    except (TypeError, ValueError):
        direction = 0
    directional_votes = [score for score in annotator_scores if score != 0]
    agreeing_votes = sum(
        1 for score in directional_votes if (1 if score > 0 else -1) == direction
    )
    return {
        "example_id": _example_id(messages, chosen, rejected),
        "domain": str(example.get("domain", "unknown")),
        "language": str(example.get("language", "unknown")),
        "preference_strength": float(strength),
        "annotator_scores": annotator_scores,
        "annotator_count": len(annotator_scores),
        "annotator_direction_agreement": (
            agreeing_votes / len(directional_votes) if directional_votes else None
        ),
        **fit,
    }


def build_sft_records(
    raw_dataset,
    tokenizer,
    *,
    max_length,
    max_samples=None,
):
    records = []
    stats = Counter()
    for raw in raw_dataset:
        example = dict(raw)
        parsed = _preference_components(example)
        if parsed is None:
            stats["invalid_or_tied"] += 1
            continue
        messages, chosen, rejected, strength = parsed
        completion_ids = _encode_completion(tokenizer, chosen)
        if len(completion_ids) >= max_length:
            stats["response_too_long"] += 1
            continue
        prompt_ids, fit = fit_prompt_to_budget(
            tokenizer, messages, max_length - len(completion_ids)
        )
        if not prompt_ids:
            stats["no_prompt_budget"] += 1
            continue
        input_ids = prompt_ids + completion_ids
        record = {
            "input_ids": input_ids,
            "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids),
            "sequence_length": len(input_ids),
            "prompt_length": len(prompt_ids),
            "response_length": len(completion_ids),
            **_metadata(example, messages, chosen, rejected, strength, fit),
        }
        records.append(record)
        stats["kept"] += 1
        stats["prompt_truncated"] += int(fit["prompt_truncated"])
        stats["token_fallback"] += int(fit["token_fallback"])
        if max_samples is not None and len(records) >= int(max_samples):
            break
    return records, stats


def build_reward_records(
    raw_dataset,
    tokenizer,
    *,
    max_length,
    max_samples=None,
):
    records = []
    stats = Counter()
    for raw in raw_dataset:
        example = dict(raw)
        parsed = _preference_components(example)
        if parsed is None:
            stats["invalid_or_tied"] += 1
            continue
        messages, chosen, rejected, strength = parsed
        chosen_completion = _encode_completion(tokenizer, chosen)
        rejected_completion = _encode_completion(tokenizer, rejected)
        max_response = max(len(chosen_completion), len(rejected_completion))
        if max_response >= max_length:
            stats["response_too_long"] += 1
            continue
        prompt_ids, fit = fit_prompt_to_budget(
            tokenizer, messages, max_length - max_response
        )
        if not prompt_ids:
            stats["no_prompt_budget"] += 1
            continue
        chosen_ids = prompt_ids + chosen_completion
        rejected_ids = prompt_ids + rejected_completion
        record = {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_length": len(chosen_ids),
            "rejected_length": len(rejected_ids),
            "prompt_length": len(prompt_ids),
            **_metadata(example, messages, chosen, rejected, strength, fit),
        }
        records.append(record)
        stats["kept"] += 1
        stats["prompt_truncated"] += int(fit["prompt_truncated"])
        stats["token_fallback"] += int(fit["token_fallback"])
        if max_samples is not None and len(records) >= int(max_samples):
            break
    return records, stats


def _decode_ids(tokenizer, token_ids):
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_dpo_records(
    raw_dataset,
    tokenizer,
    *,
    max_length,
    max_samples=None,
    preference_strength_weighting="none",
):
    """Build explicit prompt/chosen/rejected records without cutting completions.

    Standard DPO uses one row per non-tied pair. ``linear_replication`` is an
    optional sensitivity experiment that materializes strength 1/2/3 pairs
    one/two/three times. HelpSteer3 strengths are ordinal, so this option is
    intentionally not the default and is never applied to validation data.
    """
    mode = str(preference_strength_weighting).strip().lower()
    if mode not in {"none", "linear_replication"}:
        raise ValueError(
            "data.preference_strength_weighting must be 'none' or "
            "'linear_replication'"
        )

    records = []
    stats = Counter()
    for raw in raw_dataset:
        example = dict(raw)
        parsed = _preference_components(example)
        if parsed is None:
            stats["invalid_or_tied"] += 1
            continue
        messages, chosen, rejected, strength = parsed
        chosen_completion = _encode_completion(tokenizer, chosen)
        rejected_completion = _encode_completion(tokenizer, rejected)
        max_response = max(len(chosen_completion), len(rejected_completion))
        if max_response >= max_length:
            stats["response_too_long"] += 1
            continue
        prompt_ids, fit = fit_prompt_to_budget(
            tokenizer, messages, max_length - max_response
        )
        if not prompt_ids:
            stats["no_prompt_budget"] += 1
            continue

        metadata = _metadata(example, messages, chosen, rejected, strength, fit)
        base_record = {
            "prompt": _decode_ids(tokenizer, prompt_ids),
            "chosen": _decode_ids(tokenizer, chosen_completion),
            "rejected": _decode_ids(tokenizer, rejected_completion),
            "prompt_length": len(prompt_ids),
            "chosen_response_length": len(chosen_completion),
            "rejected_response_length": len(rejected_completion),
            "chosen_length": len(prompt_ids) + len(chosen_completion),
            "rejected_length": len(prompt_ids) + len(rejected_completion),
            **metadata,
        }
        repetitions = int(strength) if mode == "linear_replication" else 1
        for replica_index in range(repetitions):
            record = dict(base_record)
            record["replica_index"] = replica_index
            records.append(record)

        stats["source_pairs"] += 1
        stats["kept"] += repetitions
        stats[f"preference_strength_{int(strength)}"] += 1
        stats["prompt_truncated"] += int(fit["prompt_truncated"])
        stats["token_fallback"] += int(fit["token_fallback"])
        if max_samples is not None and stats["source_pairs"] >= int(max_samples):
            break
    return records, stats


def build_ppo_records(
    raw_dataset,
    tokenizer,
    *,
    max_prompt_length,
    max_samples=None,
):
    records = []
    stats = Counter()
    seen = set()
    for raw in raw_dataset:
        example = dict(raw)
        context = _first_present(example, CONTEXT_KEYS)
        messages = strip_trailing_assistant(normalize_messages(context))
        if not messages:
            stats["invalid_prompt"] += 1
            continue
        prompt_ids, fit = fit_prompt_to_budget(tokenizer, messages, max_prompt_length)
        if not prompt_ids:
            stats["invalid_prompt"] += 1
            continue
        digest = hashlib.sha256(bytes(str(prompt_ids), "utf-8")).hexdigest()[:20]
        if digest in seen:
            stats["duplicate_prompt"] += 1
            continue
        seen.add(digest)
        records.append(
            {
                "input_ids": prompt_ids,
                "prompt_length": len(prompt_ids),
                "example_id": digest,
                "domain": str(example.get("domain", "unknown")),
                "language": str(example.get("language", "unknown")),
                **fit,
            }
        )
        stats["kept"] += 1
        stats["prompt_truncated"] += int(fit["prompt_truncated"])
        stats["token_fallback"] += int(fit["token_fallback"])
        if max_samples is not None and len(records) >= int(max_samples):
            break
    return records, stats


def _length_summary(records):
    values = defaultdict(list)
    for row in records:
        for key, value in row.items():
            if key.endswith("_length") and isinstance(value, int):
                values[key].append(value)
    summary = {}
    for key, series in values.items():
        ordered = sorted(series)
        if not ordered:
            continue
        summary[key] = {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "max": ordered[-1],
        }
    return summary


def stage_dataset_path(cache_root, stage, split):
    return Path(cache_root) / stage / split


def load_stage_dataset(cache_root, stage, split):
    from datasets import load_from_disk

    path = stage_dataset_path(cache_root, stage, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared TRL dataset not found at {path}. Run scripts/rlhf_trl_prepare_data.py first."
        )
    return load_from_disk(str(path))


def prepare_helpsteer3_for_trl(cfg, tokenizer):
    from datasets import Dataset

    cache_root = Path(cfg.get("cache_dir", "outputs/trl/data/helpsteer3"))
    cache_root.mkdir(parents=True, exist_ok=True)
    report_path = cache_root / "preparation_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        report = {}
    report.update(
        {
            "schema_version": 2,
            "cache_dir": str(cache_root),
            "max_total_length": int(cfg.get("max_total_length", 4096)),
            "max_prompt_length": int(cfg.get("max_prompt_length", 3072)),
        }
    )
    report.setdefault("splits", {})
    stages = [str(stage) for stage in cfg.get("stages", ["sft", "reward", "ppo"])]
    invalid_stages = sorted(set(stages) - {"sft", "reward", "ppo", "dpo"})
    if invalid_stages:
        raise ValueError(f"Unsupported TRL data stages: {invalid_stages}")
    for split in cfg.get("splits", ["train", "validation"]):
        max_samples = cfg.get(f"max_{split}_samples")
        split_report = dict(report["splits"].get(str(split), {}))
        for stage in stages:
            stage_raw = load_helpsteer3_preference(str(split))
            if stage == "sft":
                records, stats = build_sft_records(
                    stage_raw,
                    tokenizer,
                    max_length=int(cfg.get("max_total_length", 4096)),
                    max_samples=max_samples,
                )
            elif stage == "reward":
                records, stats = build_reward_records(
                    stage_raw,
                    tokenizer,
                    max_length=int(cfg.get("max_total_length", 4096)),
                    max_samples=max_samples,
                )
            elif stage == "ppo":
                records, stats = build_ppo_records(
                    stage_raw,
                    tokenizer,
                    max_prompt_length=int(cfg.get("max_prompt_length", 3072)),
                    max_samples=max_samples,
                )
            else:
                weighting = (
                    cfg.get("preference_strength_weighting", "none")
                    if str(split) == str(cfg.get("train_split", "train"))
                    else "none"
                )
                records, stats = build_dpo_records(
                    stage_raw,
                    tokenizer,
                    max_length=int(cfg.get("max_total_length", 4096)),
                    max_samples=max_samples,
                    preference_strength_weighting=weighting,
                )
            path = stage_dataset_path(cache_root, stage, str(split))
            if path.exists():
                import shutil

                shutil.rmtree(path)
            Dataset.from_list(records).save_to_disk(str(path))
            split_report[stage] = {
                "path": str(path),
                "counts": dict(stats),
                "lengths": _length_summary(records),
            }
        report["splits"][str(split)] = split_report
    write_json(report, report_path)
    tokenizer.save_pretrained(cache_root / "tokenizer")
    return report
