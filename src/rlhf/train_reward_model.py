import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import load_config, save_config
from .data import build_preference_pairs, load_helpsteer3_preference, preference_pairs_to_dicts, save_jsonl
from .experiment import finalize_experiment, initialize_experiment
from .metrics import (
    append_jsonl,
    collect_run_metadata,
    jsonl_to_csv,
    read_jsonl,
    save_metric_plots,
    write_json,
)
from .reward_model import RewardModel


class PreferencePairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


class PreferenceCollator:
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, pairs):
        chosen_texts = [p.chosen_text for p in pairs]
        rejected_texts = [p.rejected_text for p in pairs]
        chosen = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "weights": torch.tensor([max(1.0, float(p.margin)) for p in pairs], dtype=torch.float32),
            "domains": [p.domain for p in pairs],
        }


def _device_from_cfg(cfg: dict[str, Any]) -> torch.device:
    name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 4),
        "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 4),
        "cuda_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 4),
    }


def _refresh_reward_artifacts(output_dir: Path) -> list[str]:
    jsonl_to_csv(output_dir / "train_metrics.jsonl", output_dir / "train_metrics.csv")
    jsonl_to_csv(output_dir / "eval_metrics.jsonl", output_dir / "eval_metrics.csv")
    train_rows = read_jsonl(output_dir / "train_metrics.jsonl")
    eval_rows = read_jsonl(output_dir / "eval_metrics.jsonl")
    plot_paths: list[str] = []
    plot_paths.extend(
        save_metric_plots(
            train_rows,
            output_dir / "plots",
            x_key="step",
            y_keys=["loss", "accuracy_batch", "reward_margin_batch", "examples_per_sec", "tokens_per_sec"],
            prefix="reward_train",
        )
    )
    plot_paths.extend(
        save_metric_plots(
            eval_rows,
            output_dir / "plots",
            x_key="step",
            y_keys=["loss", "accuracy", "avg_margin"],
            prefix="reward_eval",
        )
    )
    return plot_paths


def _load_step_offset_from_metrics(path: Path) -> int:
    """Best-effort global-step offset for resumed reward runs."""
    if path.is_dir():
        candidates = [
            path / ".." / "final_eval_metrics.json",
            path / ".." / "best_checkpoint.json",
            path / ".." / "eval_metrics.jsonl",
            path / ".." / "train_metrics.jsonl",
        ]
    else:
        candidates = [path]
    best = 0
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".jsonl":
                for row in read_jsonl(candidate):
                    best = max(best, int(row.get("step", 0)))
            else:
                import json

                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    best = max(best, int(data.get("step", 0)))
                    metrics = data.get("metrics") or data.get("final_metrics") or {}
                    if isinstance(metrics, dict):
                        best = max(best, int(metrics.get("step", 0)))
        except Exception:
            continue
    return best


def _maybe_clear_metric_artifacts(output_dir: Path, enabled: bool) -> None:
    """Remove metric files for clean resumed runs without deleting checkpoints."""
    if not enabled:
        return
    for name in [
        "train_metrics.jsonl",
        "eval_metrics.jsonl",
        "train_metrics.csv",
        "eval_metrics.csv",
        "run_summary.json",
        "final_eval_metrics.json",
        "best_checkpoint.json",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()
    plots = output_dir / "plots"
    if plots.exists():
        shutil.rmtree(plots)
    plots.mkdir(parents=True, exist_ok=True)


def evaluate_reward_model(model: RewardModel, loader: DataLoader, device: torch.device, *, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    total_margin = 0.0
    by_domain: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            batch = _move_batch(batch, device)
            chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            diff = chosen - rejected
            weights = batch["weights"].to(diff.device)
            loss = -(weights * F.logsigmoid(diff)).mean()
            total_loss += float(loss.item()) * diff.numel()
            total_correct += int((diff > 0).sum().item())
            total_margin += float(diff.sum().item())
            total += int(diff.numel())
            domains = batch.get("domains") or ["unknown"] * int(diff.numel())
            for domain, ok, margin in zip(domains, (diff > 0).detach().cpu().tolist(), diff.detach().cpu().tolist()):
                stats = by_domain.setdefault(str(domain), {"correct": 0.0, "total": 0.0, "margin_sum": 0.0})
                stats["correct"] += float(bool(ok))
                stats["total"] += 1.0
                stats["margin_sum"] += float(margin)
    result = {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "avg_margin": total_margin / max(total, 1),
        "num_pairs": total,
    }
    for domain, stats in by_domain.items():
        safe = max(stats["total"], 1.0)
        key = domain.replace("/", "_").replace(" ", "_")
        result[f"accuracy_domain_{key}"] = stats["correct"] / safe
        result[f"avg_margin_domain_{key}"] = stats["margin_sum"] / safe
    return result


def _optimizer_step(model: RewardModel, optimizer: torch.optim.Optimizer, max_grad_norm: float) -> None:
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def run_reward_training(config_path: str | Path, *, output_dir: str | Path | None = None) -> Path:
    cfg = load_config(config_path)
    output_dir = Path(output_dir or cfg.train.get("output_dir", "outputs/rlhf/qwen25_05b_helpsteer3_reward"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    _maybe_clear_metric_artifacts(output_dir, bool(cfg.train.get("clear_existing_metrics", False)))
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="rlhf_reward_model",
        config_path=config_path,
        extra={"model_name": str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))},
    )

    from transformers import AutoTokenizer

    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg.model.get("trust_remote_code", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    seed = int(cfg.train.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_raw = load_helpsteer3_preference(str(cfg.data.get("train_split", "train")))
    val_raw = load_helpsteer3_preference(str(cfg.data.get("eval_split", "validation")))
    train_pairs = build_preference_pairs(
        train_raw,
        tokenizer,
        max_samples=cfg.data.get("max_train_samples"),
        shuffle=True,
        seed=seed,
    )
    val_pairs = build_preference_pairs(
        val_raw,
        tokenizer,
        max_samples=cfg.data.get("max_eval_samples", 1000),
        shuffle=False,
        seed=seed,
    )
    if not train_pairs:
        raise RuntimeError("No reward-model training pairs were built from HelpSteer3.")
    if not val_pairs:
        raise RuntimeError("No reward-model validation pairs were built from HelpSteer3.")
    save_jsonl(preference_pairs_to_dicts(train_pairs[: min(len(train_pairs), 1000)]), output_dir / "train_pairs_preview.jsonl")
    save_jsonl(preference_pairs_to_dicts(val_pairs[: min(len(val_pairs), 1000)]), output_dir / "eval_pairs_preview.jsonl")
    write_json(
        collect_run_metadata(
            run_type="rlhf_reward_model",
            config_path=config_path,
            extra={
                "model_name": model_name,
                "num_train_pairs": len(train_pairs),
                "num_eval_pairs": len(val_pairs),
                "resume_from_checkpoint": cfg.train.get("resume_from_checkpoint"),
            },
        ),
        output_dir / "run_metadata.json",
    )

    device = _device_from_cfg(cfg.train)
    resume_checkpoint = cfg.train.get("resume_from_checkpoint")
    if resume_checkpoint:
        resume_checkpoint = Path(str(resume_checkpoint))
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Reward resume checkpoint not found: {resume_checkpoint}")
        model = RewardModel.load_rlhf_pretrained(
            resume_checkpoint,
            base_model_name=model_name,
            torch_dtype=str(cfg.model.get("torch_dtype", "auto")),
            device_map=cfg.model.get("device_map"),
            load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
            load_in_8bit=bool(cfg.model.get("load_in_8bit", False)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
            is_trainable=True,
            strict=True,
        )
        if bool(cfg.model.get("gradient_checkpointing", True)) and hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()
            if hasattr(model.backbone.config, "use_cache"):
                model.backbone.config.use_cache = False
    else:
        model = RewardModel.from_model_name(
            model_name,
            torch_dtype=str(cfg.model.get("torch_dtype", "auto")),
            device_map=cfg.model.get("device_map"),
            load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
            load_in_8bit=bool(cfg.model.get("load_in_8bit", False)),
            lora=dict(cfg.get("lora", {})),
            gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", True)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    if cfg.model.get("device_map") is None:
        model.to(device)

    collator = PreferenceCollator(tokenizer, max_length=int(cfg.data.get("max_length", 1024)))
    num_workers = int(cfg.train.get("num_workers", 0))
    pin_memory = bool(cfg.train.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(cfg.train.get("persistent_workers", num_workers > 0)) and num_workers > 0
    train_loader = DataLoader(
        PreferencePairDataset(train_pairs),
        batch_size=int(cfg.train.get("batch_size", 2)),
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        PreferencePairDataset(val_pairs),
        batch_size=int(cfg.train.get("eval_batch_size", cfg.train.get("batch_size", 2))),
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.train.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    grad_accum = int(cfg.train.get("gradient_accumulation_steps", 8))
    max_grad_norm = float(cfg.train.get("max_grad_norm", 1.0))
    num_epochs = int(cfg.train.get("epochs", 2))
    log_every = int(cfg.train.get("log_every", 10))
    eval_every = int(cfg.train.get("eval_every", 200))
    resume_step_offset = cfg.train.get("resume_step_offset")
    if resume_step_offset in {None, "auto", "AUTO"} and cfg.train.get("resume_from_checkpoint"):
        resume_step_offset = _load_step_offset_from_metrics(Path(str(cfg.train.get("resume_from_checkpoint"))))
    global_step = int(resume_step_offset or 0)
    examples_seen = 0
    tokens_seen = 0
    running_loss = 0.0
    running_batches = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_examples = 0
    last_log_tokens = 0
    artifact_every = int(cfg.train.get("artifact_every", 100))
    save_every_steps = int(cfg.train.get("save_every_steps", 0))
    best_accuracy = -1.0
    best_step = 0

    def maybe_save_best(metrics: dict[str, float]) -> None:
        nonlocal best_accuracy, best_step
        acc = float(metrics.get("accuracy", -1.0))
        step = int(metrics.get("step", global_step))
        if acc > best_accuracy:
            best_accuracy = acc
            best_step = step
            best_dir = output_dir / "checkpoint_best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            model.save_rlhf_pretrained(best_dir, tokenizer=tokenizer)
            write_json({"best_accuracy": best_accuracy, "best_step": best_step, "metrics": metrics}, output_dir / "best_checkpoint.json")

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"reward epoch {epoch + 1}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)
        pending_grads = 0
        for local_step, batch in enumerate(pbar, start=1):
            batch = _move_batch(batch, device)
            batch_examples = int(batch["chosen_input_ids"].size(0))
            batch_tokens = int(batch["chosen_attention_mask"].sum().item() + batch["rejected_attention_mask"].sum().item())
            examples_seen += batch_examples
            tokens_seen += batch_tokens
            chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            diff = chosen - rejected
            weights = batch["weights"].to(diff.device)
            loss = -(weights * F.logsigmoid(diff)).mean()
            (loss / grad_accum).backward()
            running_loss += float(loss.item())
            running_batches += 1
            pending_grads += 1

            if pending_grads >= grad_accum:
                _optimizer_step(model, optimizer, max_grad_norm)
                global_step += 1
                pending_grads = 0

                if global_step % log_every == 0:
                    now = time.time()
                    dt = max(now - last_log_time, 1e-8)
                    record = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": running_loss / max(running_batches, 1),
                        "chosen_reward": float(chosen.mean().item()),
                        "rejected_reward": float(rejected.mean().item()),
                        "reward_margin_batch": float(diff.mean().item()),
                        "accuracy_batch": float((diff > 0).float().mean().item()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_sec": now - start_time,
                        "examples_seen": examples_seen,
                        "tokens_seen": tokens_seen,
                        "examples_per_sec": (examples_seen - last_log_examples) / dt,
                        "tokens_per_sec": (tokens_seen - last_log_tokens) / dt,
                    }
                    record.update(_cuda_memory())
                    running_loss = 0.0
                    running_batches = 0
                    last_log_time = now
                    last_log_examples = examples_seen
                    last_log_tokens = tokens_seen
                    append_jsonl(record, output_dir / "train_metrics.jsonl")
                    pbar.set_postfix(loss=f"{record['loss']:.4f}", acc=f"{record['accuracy_batch']:.3f}", ex_s=f"{record['examples_per_sec']:.1f}")

                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    model.save_rlhf_pretrained(output_dir / f"checkpoint_step_{global_step:06d}", tokenizer=tokenizer)

                if artifact_every > 0 and global_step % artifact_every == 0:
                    _refresh_reward_artifacts(output_dir)

                if eval_every > 0 and global_step % eval_every == 0:
                    eval_metrics = evaluate_reward_model(
                        model,
                        val_loader,
                        device,
                        max_batches=cfg.train.get("eval_max_batches", 50),
                    )
                    eval_metrics.update({"step": global_step, "epoch": epoch + 1, "elapsed_sec": time.time() - start_time})
                    eval_metrics.update(_cuda_memory())
                    append_jsonl(eval_metrics, output_dir / "eval_metrics.jsonl")
                    maybe_save_best(eval_metrics)
                    _refresh_reward_artifacts(output_dir)
                    model.train()

        # Do not silently drop the last partial accumulation at epoch end.
        if pending_grads > 0:
            _optimizer_step(model, optimizer, max_grad_norm)
            global_step += 1
            pending_grads = 0

        epoch_eval = evaluate_reward_model(model, val_loader, device, max_batches=cfg.train.get("eval_max_batches", 50))
        epoch_eval.update({"step": global_step, "epoch": epoch + 1, "elapsed_sec": time.time() - start_time})
        epoch_eval.update(_cuda_memory())
        append_jsonl(epoch_eval, output_dir / "eval_metrics.jsonl")
        maybe_save_best(epoch_eval)
        model.save_rlhf_pretrained(output_dir / f"checkpoint_epoch_{epoch + 1:02d}", tokenizer=tokenizer)
        _refresh_reward_artifacts(output_dir)

    final_metrics = evaluate_reward_model(model, val_loader, device, max_batches=cfg.train.get("final_eval_max_batches"))
    final_metrics.update({"step": global_step, "elapsed_sec": time.time() - start_time})
    write_json(final_metrics, output_dir / "final_eval_metrics.json")
    final_metrics["step"] = global_step
    maybe_save_best(final_metrics)
    model.save_rlhf_pretrained(output_dir / "checkpoint_final", tokenizer=tokenizer)

    plot_paths = _refresh_reward_artifacts(output_dir)
    summary = {
        "final_metrics": final_metrics,
        "best_accuracy": best_accuracy,
        "best_step": best_step,
        "resume_from_checkpoint": cfg.train.get("resume_from_checkpoint"),
        "resume_step_offset": int(resume_step_offset or 0),
        "plot_paths": plot_paths,
    }
    write_json(summary, output_dir / "run_summary.json")
    finalize_experiment(output_dir, summary=summary)
    return output_dir
