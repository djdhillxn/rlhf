#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from packaging.version import Version

from _bootstrap import ensure_repo_root_on_path


REQUIRED_VERSIONS = {
    "torch": ("2.6.0", "2.6.1"),
    "torchvision": ("0.21.0", "0.21.1"),
    "torchaudio": ("2.6.0", "2.6.1"),
    "trl": ("1.6.0", "1.6.1"),
    "transformers": ("4.56.2", "5"),
    "datasets": ("4.7", "5"),
    "accelerate": ("1.6", "2"),
    "peft": ("0.15", "1"),
}


def _check_versions(errors: list[str]) -> None:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    for package, (minimum, maximum) in REQUIRED_VERSIONS.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
            continue
        print(f"{package}: {installed}")
        version = Version(installed.split("+", 1)[0])
        if not (Version(minimum) <= version < Version(maximum)):
            errors.append(f"{package} {installed} is outside [{minimum}, {maximum})")


def _check_imports(errors: list[str]) -> None:
    for module in (
        "torch",
        "torchvision",
        "torchaudio",
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


def _check_cuda(errors: list[str], require_cuda: bool) -> None:
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


def _check_directory(path: Path, label: str, errors: list[str]) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".rlhf_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        print(f"{label}: writable ({path})")
    except Exception as exc:
        errors.append(f"{label} is not writable ({path}): {type(exc).__name__}: {exc}")


def _check_data(cfg, stage: str, errors: list[str]) -> None:
    cache_dir = Path(str(cfg.data.cache_dir))
    for split_key in ("train_split", "eval_split"):
        split = str(cfg.data.get(split_key, "train" if split_key == "train_split" else "validation"))
        path = cache_dir / stage / split
        if path.exists():
            print(f"{stage} {split}: found ({path})")
        else:
            errors.append(
                f"prepared {stage} dataset is missing at {path}; "
                "rerun scripts/rlhf_trl_prepare_data.py with the same data.cache_dir"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a TRL/Colab runtime before starting an RLHF stage.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("sft", "reward", "ppo"), required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    ensure_repo_root_on_path()
    from rlhf.trl_common import load_config_with_overrides

    cfg = load_config_with_overrides(args.config, args.set)
    errors: list[str] = []
    _check_versions(errors)
    _check_imports(errors)
    _check_cuda(errors, require_cuda=not args.allow_cpu)
    _check_data(cfg, args.stage, errors)
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
