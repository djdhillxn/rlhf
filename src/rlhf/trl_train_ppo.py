from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from .config import save_config
from .experiment import finalize_experiment, initialize_experiment
from .trl_callbacks import build_callbacks
from .trl_common import build_lora_config, load_tokenizer, maybe_sync_tree, trainer_report_to, write_json
from .trl_data import load_stage_dataset
from .trl_models import (
    apply_reward_center,
    configure_ppo_sampling_distribution,
    load_causal_model,
    load_reward_center,
    load_sequence_classification_model,
    merge_peft_model,
    remove_reward_center,
)


def run_trl_ppo(cfg: dict[str, Any], *, config_path: str | Path | None = None) -> Path:
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    if cfg["train"].get("resume_from_checkpoint"):
        raise ValueError(
            "TRL v1.6 PPO does not implement exact resume_from_checkpoint. "
            "Start a new segment with model.policy_model_path and model.value_model_path instead."
        )

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")
    initialize_experiment(
        output_dir,
        cfg,
        run_type="trl_ppo",
        config_path=config_path,
        extra={"trl_backend": True, "model_name": cfg["model"]["policy_model_path"]},
    )

    tokenizer = load_tokenizer(
        str(cfg["model"]["policy_model_path"]),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        padding_side="left",
    )
    policy = load_causal_model(str(cfg["model"]["policy_model_path"]), tokenizer, cfg["model"])
    sampling_distribution = configure_ppo_sampling_distribution(
        policy,
        temperature=float(cfg["ppo"].get("temperature", 0.7)),
    )
    write_json(sampling_distribution, output_dir / "ppo_sampling_distribution.json")

    reference_path = str(cfg["model"].get("reference_model_path", cfg["model"]["policy_model_path"]))
    if Path(reference_path).resolve() == Path(str(cfg["model"]["policy_model_path"])).resolve():
        reference = None
    else:
        reference = load_causal_model(reference_path, tokenizer, cfg["model"])
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    reward_path = str(cfg["model"]["reward_model_path"])
    value_path = str(cfg["model"].get("value_model_path", reward_path))
    reward_model = load_sequence_classification_model(reward_path, tokenizer, cfg["model"])
    value_model = load_sequence_classification_model(value_path, tokenizer, cfg["model"])
    reward_offset = load_reward_center(cfg["model"].get("reward_center_path"))
    apply_reward_center(reward_model, reward_offset)
    apply_reward_center(value_model, reward_offset)

    train_dataset = load_stage_dataset(cfg["data"]["cache_dir"], "ppo", cfg["data"].get("train_split", "train"))
    eval_dataset = load_stage_dataset(
        cfg["data"]["cache_dir"], "ppo", cfg["data"].get("eval_split", "validation")
    )
    train_dataset = train_dataset.select_columns(["input_ids"])
    eval_dataset = eval_dataset.select_columns(["input_ids"])
    if cfg["data"].get("max_train_samples"):
        train_dataset = train_dataset.select(range(min(len(train_dataset), int(cfg["data"]["max_train_samples"]))))
    if cfg["data"].get("max_eval_samples"):
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), int(cfg["data"]["max_eval_samples"]))))

    lora_cfg = dict(cfg.get("lora", {}))
    if float(lora_cfg.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("The TRL PPO policy requires lora_dropout: 0.0.")

    train_cfg = cfg["train"]
    ppo_cfg = cfg["ppo"]
    args = PPOConfig(
        output_dir=str(output_dir),
        seed=int(train_cfg.get("seed", 839)),
        data_seed=int(train_cfg.get("data_seed", train_cfg.get("seed", 839))),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "linear")),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy=str(train_cfg.get("save_strategy", "steps")),
        save_steps=int(train_cfg.get("save_steps", 25)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        tf32=bool(train_cfg.get("tf32", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        report_to=trainer_report_to(train_cfg.get("report_to")),
        run_name=train_cfg.get("run_name"),
        optim=str(train_cfg.get("optim", "adamw_torch_fused")),
        total_episodes=int(ppo_cfg.get("total_episodes", 2048)),
        num_ppo_epochs=int(ppo_cfg.get("num_ppo_epochs", 4)),
        num_mini_batches=int(ppo_cfg.get("num_mini_batches", 1)),
        local_rollout_forward_batch_size=int(ppo_cfg.get("local_rollout_forward_batch_size", 4)),
        response_length=int(ppo_cfg.get("response_length", 512)),
        stop_token=str(ppo_cfg.get("stop_token", "eos")),
        temperature=float(ppo_cfg.get("temperature", 0.7)),
        missing_eos_penalty=float(ppo_cfg.get("missing_eos_penalty", 1.0)),
        whiten_rewards=bool(ppo_cfg.get("whiten_rewards", False)),
        kl_coef=float(ppo_cfg.get("kl_coef", 0.05)),
        kl_estimator=str(ppo_cfg.get("kl_estimator", "k1")),
        cliprange=float(ppo_cfg.get("cliprange", 0.2)),
        cliprange_value=float(ppo_cfg.get("cliprange_value", 0.2)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.1)),
        gamma=float(ppo_cfg.get("gamma", 1.0)),
        lam=float(ppo_cfg.get("lam", 0.95)),
        num_sample_generations=int(ppo_cfg.get("num_sample_generations", 10)),
        sft_model_path=str(cfg["model"]["policy_model_path"]),
        reward_model_path=reward_path,
    )
    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=reference,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(lora_cfg),
        callbacks=build_callbacks(train_cfg),
    )
    trainer.train()

    adapter_dir = output_dir / "final_policy_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    trained_policy = unwrapped.policy
    trained_value = unwrapped.value_model
    value_offset = remove_reward_center(trained_value)
    value_dir = output_dir / "final_value_model"
    trained_value.save_pretrained(value_dir, safe_serialization=True)
    tokenizer.save_pretrained(value_dir)
    write_json(
        {"reward_offset": value_offset, "apply_on_load": True},
        value_dir / "reward_center.json",
    )

    merged_policy_dir = output_dir / "final_merged_policy"
    merge_peft_model(trained_policy, merged_policy_dir, tokenizer)

    log_history = list(trainer.state.log_history)
    write_json(log_history, output_dir / "trainer_log_history.json")
    summary = {
        "backend": "trl",
        "stage": "ppo",
        "train_prompts": len(train_dataset),
        "eval_prompts": len(eval_dataset),
        "total_episodes": int(args.total_episodes),
        "rollout_batch_size": int(args.batch_size),
        "num_updates": int(args.num_total_batches),
        "policy_adapter_dir": str(adapter_dir),
        "merged_policy_dir": str(merged_policy_dir),
        "value_model_dir": str(value_dir),
        "reference_model_path": reference_path,
        "reward_model_path": reward_path,
        "reward_offset": reward_offset,
        "sampling_distribution": sampling_distribution,
        "last_metrics": log_history[-1] if log_history else {},
    }
    write_json(summary, output_dir / "run_summary.json")
    finalize_experiment(output_dir, summary=summary)
    maybe_sync_tree(output_dir, train_cfg.get("final_sync_dir"))

    del trainer, reward_model, value_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir
