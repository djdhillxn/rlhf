#!/usr/bin/env python3

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path


REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "torchao",
    "trl",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "numpy",
    "pandas",
    "protobuf",
)


def _check_versions(errors):
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    for package in REQUIRED_PACKAGES:
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
            continue
        print(f"{package}: {installed}")


def _check_imports(errors):
    for module in (
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
    ):
        try:
            __import__(module)
        except Exception as exc:
            errors.append(f"import {module} failed: {type(exc).__name__}: {exc}")


def _check_trl_apis(errors):
    try:
        from trl import (
            DPOConfig,
            DPOTrainer,
            RewardConfig,
            RewardTrainer,
            SFTConfig,
            SFTTrainer,
        )
        from trl.experimental.ppo import PPOConfig, PPOTrainer
    except Exception as exc:
        errors.append(f"TRL trainer API check failed: {type(exc).__name__}: {exc}")
        return
    names = [
        SFTConfig,
        SFTTrainer,
        RewardConfig,
        RewardTrainer,
        DPOConfig,
        DPOTrainer,
        PPOConfig,
        PPOTrainer,
    ]
    print("TRL trainer APIs:", ", ".join(item.__name__ for item in names))


def _check_cuda(errors, require_cuda):
    try:
        import torch
    except Exception:
        return
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    elif require_cuda:
        errors.append("CUDA is unavailable; select a GPU runtime in Colab")


def _check_directory(path, label, errors):
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".rlhf_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        print(f"{label}: writable ({path})")
    except Exception as exc:
        errors.append(f"{label} is not writable ({path}): {type(exc).__name__}: {exc}")


def _check_data(cfg, stage, errors):
    cache_dir = Path(str(cfg.data.cache_dir))
    for split_key in ("train_split", "eval_split"):
        split = str(
            cfg.data.get(
                split_key, "train" if split_key == "train_split" else "validation"
            )
        )
        path = cache_dir / stage / split
        if path.exists():
            print(f"{stage} {split}: found ({path})")
        else:
            errors.append(
                f"prepared {stage} dataset is missing at {path}; "
                "rerun scripts/rlhf_trl_prepare_data.py with the same data.cache_dir"
            )


def _check_ppo_configuration(cfg, errors, tokenizer=None):
    ppo = cfg.ppo
    train = cfg.train
    rollout = cfg.get("rollout_optimization", {})
    guardrails = cfg.get("reward_guardrails", {})
    balance = cfg.get("rollout_balance", {})
    batch_size = int(train.get("per_device_train_batch_size", 1)) * int(
        train.get("gradient_accumulation_steps", 1)
    )
    domains = list(balance.get("domains", ("code", "general", "stem", "multilingual")))
    planned_updates = math.ceil(int(ppo.total_episodes) / batch_size)
    target_update = int(train.get("target_update", planned_updates))
    print(
        "PPO schedule: "
        f"batch={batch_size}, configured episodes={int(ppo.total_episodes)}, "
        f"planned updates={planned_updates}, target update={target_update}"
    )
    if tokenizer is not None:
        dense_gib = (
            batch_size
            * int(ppo.get("response_length", 768))
            * len(tokenizer)
            * 4
            / 2**30
        )
        print(
            f"Stock TRL batch-wide rollout logits avoided: {dense_gib:.2f} GiB float32"
        )

    if bool(ppo.get("whiten_rewards", False)):
        errors.append(
            "guarded PPO requires ppo.whiten_rewards=false; advantage whitening "
            "remains active inside TRL"
        )
    if not bool(ppo.get("fixed_length_generation", False)):
        errors.append("guarded PPO requires ppo.fixed_length_generation=true")
    if not bool(ppo.get("require_eos_for_reward", False)):
        errors.append("guarded PPO requires ppo.require_eos_for_reward=true")
    if not bool(guardrails.get("enabled", False)):
        errors.append("reward_guardrails.enabled must be true")
    if not bool(balance.get("enabled", False)):
        errors.append("rollout_balance.enabled must be true")
    if not bool(rollout.get("enabled", False)):
        errors.append("rollout_optimization.enabled must be true")
    if not bool(rollout.get("share_reference_backbone", False)):
        errors.append(
            "guarded LoRA PPO requires rollout_optimization."
            "share_reference_backbone=true"
        )
    if not bool(rollout.get("enable_generation_cache", False)):
        errors.append("rollout_optimization.enable_generation_cache must be true")
    if not bool(rollout.get("require_logits_to_keep", False)):
        errors.append("rollout_optimization.require_logits_to_keep must be true")
    policy_source = str(cfg.model.policy_model_path)
    reference_source = str(
        cfg.model.get("reference_model_path", cfg.model.policy_model_path)
    )
    if (
        bool(rollout.get("share_reference_backbone", False))
        and policy_source != reference_source
        and Path(policy_source).resolve() != Path(reference_source).resolve()
    ):
        errors.append(
            "shared PEFT reference requires model.reference_model_path to match "
            "model.policy_model_path"
        )
    for key in (
        "generation_batch_size_candidates",
        "logprob_batch_size_candidates",
    ):
        values = rollout.get(key)
        if not isinstance(values, (list, tuple)) or not values:
            errors.append(f"rollout_optimization.{key} must be a non-empty list")
        else:
            try:
                parsed = [int(value) for value in values]
            except (TypeError, ValueError):
                errors.append(
                    f"rollout_optimization.{key} values must be positive integers"
                )
                continue
            if any(value <= 0 for value in parsed):
                errors.append(f"rollout_optimization.{key} values must be positive")
            else:
                print(
                    f"PPO {key.replace('_', ' ')}: "
                    + " -> ".join(str(value) for value in parsed)
                )
    if not domains or batch_size % len(domains):
        errors.append(
            f"PPO rollout batch {batch_size} must be divisible by the "
            f"{len(domains)} configured domains"
        )
    if target_update < 1 or target_update > planned_updates:
        errors.append(
            f"train.target_update must be in [1, {planned_updates}], got {target_update}"
        )
    if int(train.get("save_steps", 25)) != 25:
        print(
            "Warning: the staged notebook is designed around save_steps=25; "
            f"current value is {train.get('save_steps')}."
        )

    from rlhf.trl_train_ppo import _resolve_ppo_resume_checkpoint

    try:
        checkpoint = _resolve_ppo_resume_checkpoint(
            Path(str(train.output_dir)), train.get("resume_from_checkpoint")
        )
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"PPO resume checkpoint could not be resolved: {exc}")
        checkpoint = None
    first_boundary = int(train.get("save_steps", 25))
    if (
        target_update > first_boundary
        and bool(train.get("require_resume_after_first_segment", True))
        and checkpoint is None
    ):
        errors.append(
            f"target update {target_update} requires a restored local checkpoint, "
            "but resume_from_checkpoint resolved to none"
        )
    if checkpoint is not None:
        marker = checkpoint / "exact_resume_complete.json"
        if not marker.is_file():
            errors.append(
                f"PPO resume checkpoint is incomplete or from the legacy workflow: "
                f"{marker} is missing"
            )
        else:
            print(f"Exact PPO resume checkpoint: {checkpoint}")
            try:
                state = json.loads(
                    (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
                )
                completed_update = int(state.get("global_step", 0))
                print(f"Completed PPO update: {completed_update}")
                if target_update <= completed_update:
                    errors.append(
                        f"target update {target_update} has already been reached; "
                        "run the next target cell or export this checkpoint"
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"could not read PPO trainer state: {exc}")

    guardrail_cache = Path(str(train.output_dir)) / "ppo_reward_guardrails.json"
    checkpoint_has_guardrails = bool(
        checkpoint and (checkpoint / "ppo_resume_metadata.json").is_file()
    )
    if guardrail_cache.is_file() or checkpoint_has_guardrails:
        print("PPO reward guardrail calibration: restorable")
    else:
        calibration_split = str(guardrails.get("calibration_split", "train"))
        calibration_data = Path(str(cfg.data.cache_dir)) / "reward" / calibration_split
        if calibration_data.is_dir():
            print(f"PPO reward calibration data: found ({calibration_data})")
        else:
            errors.append(
                "first guarded PPO segment requires prepared reward-training data "
                f"at {calibration_data}"
            )


def _tokenizer_source(cfg, stage):
    if stage == "sft":
        return str(cfg.model.name)
    if stage in {"reward", "dpo"}:
        return str(cfg.model.sft_model_path)
    return str(cfg.model.policy_model_path)


def _check_tokenizer(cfg, stage, errors):
    try:
        from rlhf.trl_common import load_tokenizer

        source = _tokenizer_source(cfg, stage)
        tokenizer = load_tokenizer(
            source,
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
            padding_side="left" if stage == "ppo" else "right",
        )
        print(f"Tokenizer source: {source}")
        print(f"EOS token/id: {tokenizer.eos_token!r} / {tokenizer.eos_token_id}")
        print(f"PAD token/id: {tokenizer.pad_token!r} / {tokenizer.pad_token_id}")
        if tokenizer.eos_token_id is None:
            errors.append(
                "Tokenizer has no EOS token; PPO stop_token=eos would be ill-defined."
            )
        if tokenizer.pad_token_id is None:
            errors.append("Tokenizer has no PAD token after setup.")
        if (
            tokenizer.eos_token_id is not None
            and tokenizer.pad_token_id == tokenizer.eos_token_id
        ):
            errors.append("Tokenizer PAD and EOS token IDs must be distinct.")
        if "qwen" in source.lower() and tokenizer.eos_token != "<|im_end|>":
            print(
                "Warning: Qwen chat models usually use '<|im_end|>' as EOS; "
                "verify this tokenizer before the full run."
            )
        return tokenizer
    except Exception as exc:
        errors.append(f"tokenizer check failed: {type(exc).__name__}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate a TRL/Colab runtime before starting an RLHF stage."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", choices=("sft", "reward", "dpo", "ppo"), required=True
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    ensure_repo_root_on_path()
    from rlhf.trl_common import load_config_with_overrides

    cfg = load_config_with_overrides(args.config, args.set)
    errors = []
    _check_versions(errors)
    _check_imports(errors)
    _check_trl_apis(errors)
    _check_cuda(errors, require_cuda=not args.allow_cpu)
    tokenizer = _check_tokenizer(cfg, args.stage, errors)
    _check_data(cfg, args.stage, errors)
    if args.stage == "ppo":
        _check_ppo_configuration(cfg, errors, tokenizer=tokenizer)
    _check_directory(Path(str(cfg.train.output_dir)), "local output", errors)
    for key, label in (
        ("checkpoint_sync_dir", "checkpoint sync"),
        ("final_sync_dir", "final sync"),
    ):
        value = cfg.train.get(key)
        if value:
            _check_directory(Path(os.path.expanduser(str(value))), label, errors)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        print(f"Git commit: {commit}")
    except Exception:
        pass

    if errors:
        print("\nTRL preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("\nTRL preflight passed.")


if __name__ == "__main__":
    main()
