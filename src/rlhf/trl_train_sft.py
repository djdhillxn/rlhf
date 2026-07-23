from pathlib import Path

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
from .trl_models import load_causal_model, merge_peft_model


def run_trl_sft(cfg, *, config_path=None):
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="trl_sft",
        config_path=config_path,
        extra={"trl_backend": True, "model_name": cfg["model"]["name"]},
    )

    tokenizer = load_tokenizer(
        str(cfg["model"]["name"]),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        padding_side="right",
    )
    train_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"], "sft", cfg["data"].get("train_split", "train")
    )
    eval_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"], "sft", cfg["data"].get("eval_split", "validation")
    )
    if cfg["data"].get("max_train_samples"):
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), int(cfg["data"]["max_train_samples"])))
        )
    if cfg["data"].get("max_eval_samples"):
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), int(cfg["data"]["max_eval_samples"])))
        )

    model = load_causal_model(str(cfg["model"]["name"]), tokenizer, cfg["model"])
    lora_cfg = dict(cfg.get("lora", {}))
    if float(lora_cfg.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError(
            "The TRL pipeline requires lora_dropout: 0.0 for deterministic PPO initialization."
        )

    training_kwargs = common_training_kwargs(cfg["train"])
    training_args = SFTConfig(
        **training_kwargs,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=True,
        assistant_only_loss=False,
        packing=False,
        loss_type=str(cfg["train"].get("loss_type", "nll")),
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(lora_cfg),
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

    summary = {
        "backend": "trl",
        "stage": "sft",
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(merged_dir),
        "metrics": dict(train_result.metrics),
    }
    write_json(summary, output_dir / "run_summary.json")
    finalize_experiment(output_dir, summary=summary)
    maybe_sync_tree(output_dir, cfg["train"].get("final_sync_dir"))
    return output_dir
