"""Copy RLHF artifacts between local SSD, Google Drive, and the repository.

The lightweight profile keeps configs, logs, metrics, reports, tokenizers, and
Trainer state while excluding model weights and files over the size limit.
The full profile is intended for Colab restore and persistence.
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_RELATIVE_PATH = Path(
    "Colab Notebooks/8mile/rlhf_runs/"
    "qwen25_05b_helpsteer3_trl_a100_full/full"
)
DEFAULT_LIGHTWEIGHT_DESTINATION = (
    REPOSITORY_ROOT
    / "rlhf_runs_lightweight_export"
    / "qwen25_05b_helpsteer3_trl_a100_full"
    / "full"
)
IGNORED_NAMES = {".DS_Store", "__pycache__"}
WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
WEIGHT_NAMES = {
    "optimizer.pt",
    "pytorch_model.bin",
    "rng_state.pth",
    "scheduler.pt",
    "training_args.bin",
}
SYNC_WORKERS = 8


def discover_drive_run(explicit_path=None):
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    environment_path = os.environ.get("RLHF_DRIVE_RUN_ROOT")
    if environment_path:
        return Path(environment_path).expanduser().resolve()

    colab_path = Path("/content/drive/MyDrive") / RUN_RELATIVE_PATH
    if colab_path.is_dir():
        return colab_path

    cloud_storage = Path.home() / "Library" / "CloudStorage"
    patterns = (
        Path("GoogleDrive-*") / "My Drive" / RUN_RELATIVE_PATH,
        Path("GoogleDrive-*") / ".Encrypted" / "My Drive" / RUN_RELATIVE_PATH,
    )
    for pattern in patterns:
        matches = sorted(
            {path.resolve() for path in cloud_storage.glob(str(pattern)) if path.is_dir()}
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = "\n".join(f"  - {path}" for path in matches)
            raise RuntimeError(
                "Multiple Google Drive RLHF run folders were found. Pass --source:\n"
                + choices
            )
    raise FileNotFoundError(
        "Could not find the RLHF Drive run. Pass --source or set "
        "RLHF_DRIVE_RUN_ROOT."
    )


def _safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"--include must be relative to --source: {value!r}")
    return path


def selected_roots(source, includes):
    if not includes:
        return [source]
    roots = []
    for value in includes:
        relative = _safe_relative(value)
        path = source / relative
        if path.exists():
            roots.append(path)
        else:
            print(f"Skipping missing selection: {path}")
    return roots


def is_lightweight_file(path, max_bytes):
    if path.name in WEIGHT_NAMES or path.suffix.lower() in WEIGHT_SUFFIXES:
        return False
    try:
        return path.stat().st_size <= max_bytes
    except OSError:
        return False


def iter_files(source, roots, profile, max_bytes):
    for root in roots:
        if root.is_file():
            if profile == "full" or is_lightweight_file(root, max_bytes):
                yield root
            continue
        for directory, directory_names, file_names in os.walk(root):
            directory_names[:] = [
                name for name in directory_names if name not in IGNORED_NAMES
            ]
            for name in file_names:
                path = Path(directory) / name
                if name in IGNORED_NAMES:
                    continue
                if profile == "lightweight" and not is_lightweight_file(
                    path, max_bytes
                ):
                    continue
                yield path


def file_is_current(source, destination):
    try:
        source_stat = source.stat()
        destination_stat = destination.stat()
        return (
            destination.is_file()
            and source_stat.st_size == destination_stat.st_size
            and destination_stat.st_mtime_ns >= source_stat.st_mtime_ns
        )
    except OSError:
        return False


def copy_file(source, destination, dry_run=False):
    if file_is_current(source, destination):
        return False
    if dry_run:
        print(f"Would copy: {source} -> {destination}")
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.rlhf-sync.tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return True


def sync_tree(
    source,
    destination,
    *,
    profile="lightweight",
    includes=None,
    max_size_mb=50,
    dry_run=False,
):
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"Sync source does not exist: {source}")
    roots = selected_roots(source, includes or [])
    paths = list(
        iter_files(source, roots, profile, int(float(max_size_mb) * 1024 * 1024))
    )

    def copy_one(path):
        return copy_file(path, destination / path.relative_to(source), dry_run=dry_run)

    if dry_run or len(paths) < 2:
        copied = sum(bool(copy_one(path)) for path in paths)
    else:
        with ThreadPoolExecutor(max_workers=min(SYNC_WORKERS, len(paths))) as executor:
            copied = sum(bool(value) for value in executor.map(copy_one, paths))
    return {"source": str(source), "destination": str(destination), "files": len(paths), "copied": copied}


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize full or analysis-only RLHF run artifacts."
    )
    parser.add_argument("--source", default=None)
    parser.add_argument("--destination", default=None)
    parser.add_argument(
        "--profile", choices=("lightweight", "full"), default="lightweight"
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Relative source path to include. May be repeated.",
    )
    parser.add_argument("--max-size-mb", type=float, default=50.0)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        source = discover_drive_run(args.source)
    except FileNotFoundError:
        if args.allow_missing:
            print("Sync source is absent; nothing to restore.")
            return
        raise
    if not source.is_dir():
        if args.allow_missing:
            print(f"Sync source is absent; nothing to restore: {source}")
            return
        raise FileNotFoundError(f"Sync source does not exist: {source}")
    destination = Path(args.destination or DEFAULT_LIGHTWEIGHT_DESTINATION).expanduser()
    result = sync_tree(
        source,
        destination,
        profile=args.profile,
        includes=args.include,
        max_size_mb=args.max_size_mb,
        dry_run=args.dry_run,
    )
    result["profile"] = args.profile
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
