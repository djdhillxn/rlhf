from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


class CheckpointSyncCallback(TrainerCallback):
    """Copy completed Trainer checkpoints to persistent storage after each save."""

    def __init__(self, destination: str | Path | None) -> None:
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
        (self.destination / "latest_checkpoint.txt").write_text(target.name + "\n", encoding="utf-8")
        return control


def build_callbacks(cfg: dict[str, Any]) -> list[TrainerCallback]:
    destination = cfg.get("checkpoint_sync_dir")
    return [CheckpointSyncCallback(destination)] if destination else []
