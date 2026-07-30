import gc
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
    resolve_resume_checkpoint,
    write_json,
)
from .trl_data import load_stage_dataset
from .trl_models import load_causal_model, merge_peft_model


def run_trl_dpo(cfg, *, config_path=None):
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="trl_dpo",
        config_path=config_path,
        extra={"trl_backend": True, "model_name": cfg["model"]["sft_model_path"]},
    )

    tokenizer = load_tokenizer(
        str(cfg["model"]["sft_model_path"]),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        padding_side="right",
    )
    train_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"], "dpo", cfg["data"].get("train_split", "train")
    )
    eval_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"],
        "dpo",
        cfg["data"].get("eval_split", "validation"),
    )
    if cfg["data"].get("max_train_samples"):
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), int(cfg["data"]["max_train_samples"])))
        )
    if cfg["data"].get("max_eval_samples"):
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), int(cfg["data"]["max_eval_samples"])))
        )

    model = load_causal_model(
        str(cfg["model"]["sft_model_path"]), tokenizer, cfg["model"]
    )
    lora_cfg = dict(cfg.get("lora", {}))
    if float(lora_cfg.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("The DPO path requires lora_dropout: 0.0.")

    dpo_cfg = cfg.get("dpo", {})
    training_args = DPOConfig(
        **common_training_kwargs(cfg["train"]),
        beta=float(dpo_cfg.get("beta", 0.1)),
        loss_type=str(dpo_cfg.get("loss_type", "sigmoid")),
        label_smoothing=float(dpo_cfg.get("label_smoothing", 0.0)),
        max_length=dpo_cfg.get("max_length"),
        truncation_mode=str(dpo_cfg.get("truncation_mode", "keep_start")),
        disable_dropout=bool(dpo_cfg.get("disable_dropout", True)),
        dataset_num_proc=int(dpo_cfg.get("dataset_num_proc", 4)),
        precompute_ref_log_probs=bool(
            dpo_cfg.get("precompute_ref_log_probs", True)
        ),
        precompute_ref_batch_size=int(
            dpo_cfg.get(
                "precompute_ref_batch_size",
                cfg["train"].get("per_device_train_batch_size", 1),
            )
        ),
        padding_free=bool(dpo_cfg.get("padding_free", False)),
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(lora_cfg),
        callbacks=build_callbacks(cfg["train"]),
    )
    resume = resolve_resume_checkpoint(
        output_dir, cfg["train"].get("resume_from_checkpoint")
    )
    train_result = trainer.train(
        resume_from_checkpoint=str(resume) if resume is not None else None
    )
    eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    merged_dir = output_dir / "final_merged_model"
    merge_peft_model(unwrapped, merged_dir, tokenizer)

    summary = {
        "backend": "trl",
        "stage": "dpo",
        "objective": str(dpo_cfg.get("loss_type", "sigmoid")),
        "beta": float(dpo_cfg.get("beta", 0.1)),
        "preference_strength_weighting": str(
            cfg["data"].get("preference_strength_weighting", "none")
        ),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "resumed_from_checkpoint": str(resume) if resume is not None else None,
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(merged_dir),
        "train_metrics": dict(train_result.metrics),
        "eval_metrics": dict(eval_metrics),
    }
    write_json(summary, output_dir / "run_summary.json")
    finalize_experiment(output_dir, summary=summary)
    maybe_sync_tree(output_dir, cfg["train"].get("final_sync_dir"))

    del trainer, model, unwrapped
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir
