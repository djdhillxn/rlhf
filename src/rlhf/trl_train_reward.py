import csv
import gc
from collections import defaultdict
from pathlib import Path

import torch

from .config import save_config
from .experiment import finalize_experiment, initialize_experiment
from .trl_common import (
    build_callbacks,
    build_lora_config,
    common_training_kwargs,
    load_tokenizer,
    maybe_sync_tree,
    write_json,
)
from .trl_data import load_stage_dataset
from .trl_models import (
    apply_reward_center,
    initialize_reward_head,
    load_sequence_classification_model,
    merge_peft_model,
    save_reward_center,
    score_tokenized_sequences,
)


def _group_accuracy(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(bool(row["correct"]))
    return {
        name: {"accuracy": sum(values) / len(values), "count": len(values)}
        for name, values in sorted(grouped.items())
    }


def _calibration(rows, bins=10):
    grouped = defaultdict(list)
    for row in rows:
        confidence = float(row["preference_probability"])
        index = min(bins - 1, max(0, int(confidence * bins)))
        grouped[index].append(row)
    output = []
    for index in range(bins):
        values = grouped.get(index, [])
        if not values:
            continue
        output.append(
            {
                "bin_lower": index / bins,
                "bin_upper": (index + 1) / bins,
                "mean_predicted_probability": sum(
                    float(row["preference_probability"]) for row in values
                )
                / len(values),
                "empirical_accuracy": sum(bool(row["correct"]) for row in values)
                / len(values),
                "count": len(values),
            }
        )
    return output


def _write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _audit_reward_model(
    model,
    dataset,
    *,
    tokenizer,
    device,
    batch_size,
    output_dir,
):
    chosen = score_tokenized_sequences(
        model,
        [list(row) for row in dataset["chosen_ids"]],
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        batch_size=batch_size,
    )
    rejected = score_tokenized_sequences(
        model,
        [list(row) for row in dataset["rejected_ids"]],
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        batch_size=batch_size,
    )
    rows = []
    for idx, (chosen_score, rejected_score) in enumerate(
        zip(chosen.tolist(), rejected.tolist())
    ):
        diff = float(chosen_score - rejected_score)
        rows.append(
            {
                "example_id": dataset[idx].get("example_id", str(idx)),
                "domain": dataset[idx].get("domain", "unknown"),
                "language": dataset[idx].get("language", "unknown"),
                "preference_strength": float(
                    dataset[idx].get("preference_strength", 0.0)
                ),
                "chosen_reward": float(chosen_score),
                "rejected_reward": float(rejected_score),
                "reward_difference": diff,
                "preference_probability": float(
                    torch.sigmoid(torch.tensor(diff)).item()
                ),
                "correct": diff > 0,
            }
        )
    calibration = _calibration(rows)
    _write_csv(rows, output_dir / "reward_validation_predictions.csv")
    _write_csv(calibration, output_dir / "reward_calibration.csv")
    return {
        "accuracy": sum(bool(row["correct"]) for row in rows) / max(len(rows), 1),
        "count": len(rows),
        "mean_margin": sum(float(row["reward_difference"]) for row in rows)
        / max(len(rows), 1),
        "by_domain": _group_accuracy(rows, "domain"),
        "by_language": _group_accuracy(rows, "language"),
        "by_preference_strength": _group_accuracy(rows, "preference_strength"),
        "calibration": calibration,
    }


def run_trl_reward(cfg, *, config_path=None):
    from trl import RewardConfig, RewardTrainer

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="trl_reward_model",
        config_path=config_path,
        extra={"trl_backend": True, "model_name": cfg["model"]["sft_model_path"]},
    )

    tokenizer = load_tokenizer(
        str(cfg["model"]["sft_model_path"]),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        padding_side="right",
    )
    train_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"], "reward", cfg["data"].get("train_split", "train")
    )
    eval_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"], "reward", cfg["data"].get("eval_split", "validation")
    )
    if cfg["data"].get("max_train_samples"):
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), int(cfg["data"]["max_train_samples"])))
        )
    if cfg["data"].get("max_eval_samples"):
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), int(cfg["data"]["max_eval_samples"])))
        )

    model = load_sequence_classification_model(
        str(cfg["model"]["sft_model_path"]), tokenizer, cfg["model"]
    )
    if bool(cfg["model"].get("initialize_reward_head", True)):
        initialization = initialize_reward_head(model)
    else:
        initialization = {
            "skipped": True,
            "reason": "model.initialize_reward_head=false",
        }
    write_json(initialization, output_dir / "reward_head_initialization.json")

    lora_cfg = dict(cfg.get("lora", {}))
    if float(lora_cfg.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("The TRL pipeline requires lora_dropout: 0.0.")
    lora_cfg["task_type"] = "SEQ_CLS"

    training_args = RewardConfig(
        **common_training_kwargs(cfg["train"]),
        max_length=None,
        disable_dropout=True,
        center_rewards_coefficient=float(
            cfg["train"].get("center_rewards_coefficient", 0.01)
        ),
    )
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(lora_cfg, modules_to_save=["score"]),
        callbacks=build_callbacks(cfg["train"]),
    )
    resume = cfg["train"].get("resume_from_checkpoint")
    train_result = trainer.train(resume_from_checkpoint=resume if resume else None)

    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    merged_dir = output_dir / "final_merged_model"
    merge_peft_model(unwrapped, merged_dir, tokenizer)

    del trainer, model, unwrapped
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model = load_sequence_classification_model(
        str(merged_dir), tokenizer, cfg["model"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged_model.to(device)

    center_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"],
        "sft",
        cfg["data"].get("center_split", cfg["data"].get("train_split", "train")),
    )
    max_center = cfg["data"].get("max_center_samples")
    if max_center:
        center_dataset = center_dataset.select(
            range(min(len(center_dataset), int(max_center)))
        )
    raw_reference_scores = score_tokenized_sequences(
        merged_model,
        [list(row) for row in center_dataset["input_ids"]],
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        batch_size=int(cfg["train"].get("audit_batch_size", 16)),
    )
    reward_offset = float(raw_reference_scores.mean().item())
    raw_std = float(raw_reference_scores.std(unbiased=False).item())
    center_path = output_dir / "reward_center.json"
    save_reward_center(
        reward_offset,
        center_path,
        num_examples=len(raw_reference_scores),
        raw_std=raw_std,
    )
    apply_reward_center(merged_model, reward_offset)

    audit = _audit_reward_model(
        merged_model,
        eval_dataset,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(cfg["train"].get("audit_batch_size", 16)),
        output_dir=output_dir,
    )
    write_json(audit, output_dir / "reward_audit.json")

    summary = {
        "backend": "trl",
        "stage": "reward_model",
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(merged_dir),
        "reward_center_path": str(center_path),
        "reward_offset": reward_offset,
        "metrics": dict(train_result.metrics),
        "audit": audit,
    }
    write_json(summary, output_dir / "run_summary.json")
    finalize_experiment(output_dir, summary=summary)
    maybe_sync_tree(output_dir, cfg["train"].get("final_sync_dir"))
    return output_dir
