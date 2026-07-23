import json
import math
from pathlib import Path

import torch
from torch import nn

from .trl_common import resize_embeddings_if_needed, resolve_dtype


def model_load_kwargs(cfg):
    kwargs = {
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
    }
    dtype = resolve_dtype(cfg.get("dtype", cfg.get("torch_dtype", "bfloat16")))
    if dtype != "auto":
        kwargs["dtype"] = dtype
    if cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = str(cfg["attn_implementation"])
    return kwargs


def disable_dropout_modules(model):
    """Make PPO log-probability ratios reproducible even if a backbone defines dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def load_causal_model(model_name_or_path, tokenizer, cfg):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, **model_load_kwargs(cfg)
    )
    resize_embeddings_if_needed(model, tokenizer)
    disable_dropout_modules(model)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model


def load_sequence_classification_model(model_name_or_path, tokenizer, cfg):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1,
        **model_load_kwargs(cfg),
    )
    resize_embeddings_if_needed(model, tokenizer)
    disable_dropout_modules(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model


def configure_ppo_sampling_distribution(model, *, temperature):
    """Remove model-card decoding heuristics that invalidate PPO behavior ratios."""
    generation_config = model.generation_config
    neutral_values = {
        "do_sample": True,
        "temperature": float(temperature),
        "top_k": 0,
        "top_p": 1.0,
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "sequence_bias": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "diversity_penalty": 0.0,
    }
    applied = {}
    for name, value in neutral_values.items():
        if hasattr(generation_config, name):
            setattr(generation_config, name, value)
            applied[name] = value
    return applied


def initialize_reward_head(model):
    """Apply the scalar-head initialization used by the N+ reference implementation."""
    score = getattr(model, "score", None)
    if not isinstance(score, nn.Linear):
        raise TypeError(f"Expected a linear `score` head, found {type(score).__name__}")
    hidden_size = int(score.in_features)
    std = 1.0 / math.sqrt(hidden_size + 1)
    nn.init.normal_(score.weight, mean=0.0, std=std)
    if score.bias is not None:
        nn.init.zeros_(score.bias)
    return {"hidden_size": hidden_size, "weight_std": std, "bias": 0.0}


class OffsetScore(nn.Module):
    """Subtract a fixed calibration offset while retaining the original score head."""

    def __init__(self, base, offset):
        super().__init__()
        self.base = base
        self.register_buffer(
            "offset", torch.tensor(float(offset), dtype=torch.float32), persistent=False
        )

    def forward(self, hidden_states):
        scores = self.base(hidden_states)
        return scores - self.offset.to(device=scores.device, dtype=scores.dtype)


def apply_reward_center(model, offset):
    score = getattr(model, "score", None)
    if score is None:
        raise AttributeError("Reward/value model has no `score` head.")
    if isinstance(score, OffsetScore):
        score.offset.fill_(float(offset))
    else:
        model.score = OffsetScore(score, float(offset))
    return model


def remove_reward_center(model):
    score = getattr(model, "score", None)
    if not isinstance(score, OffsetScore):
        return 0.0
    offset = float(score.offset.item())
    model.score = score.base
    return offset


def save_reward_center(offset, path, *, num_examples, raw_std):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reward_offset": float(offset),
                "reference": "HelpSteer3 preferred SFT demonstrations",
                "num_examples": int(num_examples),
                "raw_reward_std": float(raw_std),
                "interpretation": "centered_reward = raw_reward - reward_offset",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_reward_center(path):
    if not path:
        return 0.0
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(payload.get("reward_offset", 0.0))


def merge_peft_model(model, output_dir, tokenizer):
    from peft import PeftModel

    if not isinstance(model, PeftModel):
        raise TypeError(f"Expected a PEFT model to merge, found {type(model).__name__}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


@torch.inference_mode()
def score_tokenized_sequences(
    model,
    records,
    *,
    pad_token_id,
    device,
    batch_size=16,
):
    scores = []
    model.eval()
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        width = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), width), pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (len(rows), width), dtype=torch.long, device=device
        )
        for idx, row in enumerate(rows):
            length = len(row)
            input_ids[idx, :length] = torch.tensor(row, dtype=torch.long, device=device)
            attention_mask[idx, :length] = 1
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        scores.append(output.logits.squeeze(-1).float().cpu())
    return torch.cat(scores) if scores else torch.empty(0)
