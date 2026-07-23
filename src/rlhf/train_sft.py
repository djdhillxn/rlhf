from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import load_config, save_config
from .data import (
    build_preference_pairs,
    load_helpsteer3_preference,
    preference_pairs_to_dicts,
    save_jsonl,
)
from .experiment import finalize_experiment, initialize_experiment
from .lm_policy import TokenPolicyWithValue
from .metrics import (
    append_jsonl,
    collect_run_metadata,
    jsonl_to_csv,
    read_jsonl,
    save_metric_plots,
    write_json,
)


class SFTDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "prompt": pair.prompt,
            "text": pair.chosen_text,
            "domain": pair.domain,
            "language": pair.language,
        }


class SFTCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, rows):
        encoded_rows = []
        max_len = 0
        for row in rows:
            full = self.tokenizer(
                row["text"],
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            prompt = self.tokenizer(
                row["prompt"],
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            prompt_len = min(len(prompt), len(full))
            if len(full) <= prompt_len:
                # If truncation removed the response entirely, keep the row but mask it out.
                labels = [-100] * len(full)
            else:
                labels = [-100] * prompt_len + full[prompt_len:]
            encoded_rows.append((full, labels))
            max_len = max(max_len, len(full))

        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for ids, labs in encoded_rows:
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(labs + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "domains": [r["domain"] for r in rows],
        }


def _device_from_cfg(cfg):
    name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _move_batch(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _cuda_memory():
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 4),
        "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 4),
        "cuda_max_memory_allocated_gb": round(
            torch.cuda.max_memory_allocated() / (1024**3), 4
        ),
    }


def _refresh_sft_artifacts(output_dir):
    jsonl_to_csv(output_dir / "train_metrics.jsonl", output_dir / "train_metrics.csv")
    rows = read_jsonl(output_dir / "train_metrics.jsonl")
    return save_metric_plots(
        rows,
        output_dir / "plots",
        x_key="step",
        y_keys=["loss", "learning_rate", "tokens_per_sec", "examples_per_sec"],
        prefix="sft_train",
    )


def run_sft_training(config_path, *, output_dir=None):
    cfg = load_config(config_path)
    output_dir = Path(
        output_dir
        or cfg.train.get("output_dir", "outputs/rlhf/qwen25_05b_helpsteer3_sft")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="rlhf_sft_policy",
        config_path=config_path,
        extra={"model_name": str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))},
    )

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=bool(cfg.model.get("trust_remote_code", False))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    seed = int(cfg.train.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    raw = load_helpsteer3_preference(str(cfg.data.get("train_split", "train")))
    pairs = build_preference_pairs(
        raw,
        tokenizer,
        max_samples=cfg.data.get("max_train_samples"),
        shuffle=True,
        seed=seed,
    )
    if not pairs:
        raise RuntimeError("No SFT training pairs were built from HelpSteer3.")
    save_jsonl(
        preference_pairs_to_dicts(pairs[: min(len(pairs), 1000)]),
        output_dir / "sft_pairs_preview.jsonl",
    )
    write_json(
        collect_run_metadata(
            run_type="rlhf_sft_policy",
            config_path=config_path,
            extra={"model_name": model_name, "num_train_pairs": len(pairs)},
        ),
        output_dir / "run_metadata.json",
    )

    device = _device_from_cfg(cfg.train)
    policy = TokenPolicyWithValue.from_model_name(
        model_name,
        torch_dtype=str(cfg.model.get("torch_dtype", "auto")),
        device_map=cfg.model.get("device_map"),
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
        load_in_8bit=bool(cfg.model.get("load_in_8bit", False)),
        lora=dict(cfg.get("lora", {})),
        gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if cfg.model.get("device_map") is None:
        policy.to(device)
    policy.train()

    collator = SFTCollator(tokenizer, max_length=int(cfg.data.get("max_length", 1024)))
    num_workers = int(cfg.train.get("num_workers", 0))
    pin_memory = bool(cfg.train.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = (
        bool(cfg.train.get("persistent_workers", num_workers > 0)) and num_workers > 0
    )
    loader = DataLoader(
        SFTDataset(pairs),
        batch_size=int(cfg.train.get("batch_size", 4)),
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    optimizer = torch.optim.AdamW(
        policy.trainable_parameters(),
        lr=float(cfg.train.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    epochs = int(cfg.train.get("epochs", 1))
    grad_accum = int(cfg.train.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(cfg.train.get("max_grad_norm", 1.0))
    total_steps = max(1, (len(loader) * epochs + grad_accum - 1) // grad_accum)
    warmup_steps = int(cfg.train.get("warmup_steps", max(10, int(0.03 * total_steps))))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    global_step = 0
    examples_seen = 0
    tokens_seen = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_examples = 0
    last_log_tokens = 0
    running_loss = 0.0
    running_batches = 0
    log_every = int(cfg.train.get("log_every", 10))
    artifact_every = int(cfg.train.get("artifact_every", 100))
    save_every_steps = int(cfg.train.get("save_every_steps", 0))

    optimizer.zero_grad(set_to_none=True)
    pending = 0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"sft epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch = _move_batch(batch, device)
            out = policy.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = out.loss
            (loss / grad_accum).backward()
            pending += 1
            running_loss += float(loss.item())
            running_batches += 1
            examples_seen += int(batch["input_ids"].size(0))
            tokens_seen += int(batch["attention_mask"].sum().item())

            if pending >= grad_accum:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        policy.trainable_parameters(), max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                global_step += 1

                if global_step % log_every == 0:
                    now = time.time()
                    dt = max(now - last_log_time, 1e-8)
                    record = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": running_loss / max(running_batches, 1),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_sec": now - start_time,
                        "examples_seen": examples_seen,
                        "tokens_seen": tokens_seen,
                        "examples_per_sec": (examples_seen - last_log_examples) / dt,
                        "tokens_per_sec": (tokens_seen - last_log_tokens) / dt,
                    }
                    record.update(_cuda_memory())
                    append_jsonl(record, output_dir / "train_metrics.jsonl")
                    pbar.set_postfix(
                        loss=f"{record['loss']:.4f}",
                        tok_s=f"{record['tokens_per_sec']:.0f}",
                    )
                    running_loss = 0.0
                    running_batches = 0
                    last_log_time = now
                    last_log_examples = examples_seen
                    last_log_tokens = tokens_seen
                if artifact_every > 0 and global_step % artifact_every == 0:
                    _refresh_sft_artifacts(output_dir)
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    policy.save_rlhf_pretrained(
                        output_dir / f"checkpoint_step_{global_step:06d}",
                        tokenizer=tokenizer,
                    )

    if pending > 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    policy.save_rlhf_pretrained(output_dir / "checkpoint_final", tokenizer=tokenizer)
    plot_paths = _refresh_sft_artifacts(output_dir)
    summary = {"total_steps": global_step, "plot_paths": plot_paths}
    write_json(summary, output_dir / "run_summary.json")
    finalize_experiment(output_dir, summary=summary)
    return output_dir
