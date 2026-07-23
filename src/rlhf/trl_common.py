import json
import os
import shutil
from pathlib import Path

import yaml

from .config import apply_overrides, load_config


def parse_cli_overrides(values):
    """Parse repeated KEY=VALUE arguments using YAML scalar semantics."""
    overrides = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Override must use KEY=VALUE syntax, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {raw!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def load_config_with_overrides(path, values=None):
    cfg = load_config(path)
    overrides = parse_cli_overrides(values)
    return apply_overrides(cfg, overrides) if overrides else cfg


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def resolve_dtype(name):
    import torch

    if name is None or str(name).lower() == "auto":
        return "auto"
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def ensure_distinct_pad_token(tokenizer, pad_token="<|pad|>"):
    """Guarantee a real padding token rather than aliasing EOS as padding."""
    eos_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == eos_id:
        vocab = tokenizer.get_vocab()
        if pad_token in vocab:
            tokenizer.pad_token = pad_token
        else:
            tokenizer.add_special_tokens({"pad_token": pad_token})
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer still has no pad token after setup.")
    if eos_id is not None and tokenizer.pad_token_id == eos_id:
        raise ValueError(
            "PAD and EOS must have different token IDs for the TRL pipeline."
        )
    return int(tokenizer.pad_token_id)


def load_tokenizer(
    model_name_or_path, *, trust_remote_code=False, padding_side="right"
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )
    ensure_distinct_pad_token(tokenizer)
    return tokenizer


def resize_embeddings_if_needed(model, tokenizer):
    embeddings = model.get_input_embeddings()
    if embeddings is not None and len(tokenizer) != embeddings.num_embeddings:
        model.resize_token_embeddings(len(tokenizer))


def build_lora_config(cfg, *, modules_to_save=None):
    from peft import LoraConfig

    return LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        bias=str(cfg.get("bias", "none")),
        task_type=str(cfg.get("task_type", "CAUSAL_LM")),
        target_modules=list(
            cfg.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        ),
        modules_to_save=modules_to_save,
    )


def trainer_report_to(value):
    if value in {None, "", "none", "None", False}:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def maybe_sync_tree(source, destination):
    if not destination:
        return
    source = Path(source)
    destination = Path(os.path.expanduser(str(destination)))
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def common_training_kwargs(cfg):
    """Translate the shared YAML training fields into Transformers arguments."""
    kwargs = {
        "output_dir": str(cfg["output_dir"]),
        "seed": int(cfg.get("seed", 839)),
        "data_seed": int(cfg.get("data_seed", cfg.get("seed", 839))),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(cfg.get("learning_rate", 3e-6)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "num_train_epochs": float(cfg.get("num_train_epochs", 1.0)),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.0)),
        "lr_scheduler_type": str(cfg.get("lr_scheduler_type", "cosine")),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "save_strategy": str(cfg.get("save_strategy", "steps")),
        "save_steps": int(cfg.get("save_steps", 100)),
        "save_total_limit": int(cfg.get("save_total_limit", 2)),
        "eval_strategy": str(cfg.get("eval_strategy", "steps")),
        "eval_steps": int(cfg.get("eval_steps", 100)),
        "bf16": bool(cfg.get("bf16", True)),
        "fp16": bool(cfg.get("fp16", False)),
        "tf32": bool(cfg.get("tf32", True)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 2)),
        "dataloader_pin_memory": bool(cfg.get("dataloader_pin_memory", True)),
        "remove_unused_columns": bool(cfg.get("remove_unused_columns", True)),
        "report_to": trainer_report_to(cfg.get("report_to")),
        "run_name": cfg.get("run_name"),
        "optim": str(cfg.get("optim", "adamw_torch_fused")),
    }
    if cfg.get("max_steps") is not None:
        kwargs["max_steps"] = int(cfg["max_steps"])
    if cfg.get("logging_first_step") is not None:
        kwargs["logging_first_step"] = bool(cfg["logging_first_step"])
    return kwargs


def _checkpoint_sync_callback(destination):
    from transformers import TrainerCallback

    class CheckpointSyncCallback(TrainerCallback):
        """Copy completed Trainer checkpoints to persistent storage after each save."""

        def __init__(self, destination):
            self.destination = Path(destination).expanduser() if destination else None

        def on_save(self, args, state, control, **kwargs):
            if self.destination is None or not state.is_world_process_zero:
                return control
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not checkpoint.exists():
                return control
            target = self.destination / checkpoint.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(checkpoint, target, dirs_exist_ok=True)
            (self.destination / "latest_checkpoint.txt").write_text(
                target.name + "\n", encoding="utf-8"
            )
            return control

    return CheckpointSyncCallback(destination)


def build_callbacks(cfg):
    destination = cfg.get("checkpoint_sync_dir")
    return [_checkpoint_sync_callback(destination)] if destination else []
