"""Small, local experiment manifests for reproducible RLHF runs."""

import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping

import yaml


MANIFEST_NAME = "experiment_manifest.json"
SCHEMA_VERSION = 1


def _write_json(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(record), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _collect_run_metadata(**kwargs):
    from .metrics import collect_run_metadata

    return collect_run_metadata(**kwargs)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_sha256(config):
    payload = json.dumps(
        _plain(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def _git(args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_source_state():
    # Generated run artifacts should not make an otherwise clean source tree
    # appear dirty merely because the output directory was created first.
    status = _git(
        [
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
            ":(exclude)outputs",
        ]
    )
    changed_paths = []
    if status:
        changed_paths = [
            line[3:] if len(line) > 3 else line for line in status.splitlines()[:100]
        ]
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_dirty": bool(status),
        "changed_paths": changed_paths,
        "changed_paths_truncated": bool(status and len(status.splitlines()) > 100),
    }


def _experiment_fields(config):
    raw = config.get("experiment", {})
    return _plain(raw) if isinstance(raw, Mapping) else {}


def initialize_experiment(
    output_dir,
    config,
    *,
    run_type,
    config_path=None,
    extra=None,
):
    """Create or resume the manifest in an output directory.

    Existing run files remain valid. A repeated invocation appends an attempt
    instead of replacing the original run identity.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    parameters = _plain(config)
    fingerprint = config_sha256(parameters)
    now = _utc_now()
    experiment = _experiment_fields(parameters)
    logical_id = str(experiment.get("id") or output_dir.name)
    attempt = {
        "started_at_utc": _iso(now),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": fingerprint,
        "source": collect_source_state(),
    }

    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    if existing:
        manifest = existing
        previous_fingerprint = manifest.get("config_sha256")
        manifest["status"] = "running"
        manifest["last_started_at_utc"] = _iso(now)
        manifest["config_changed_on_resume"] = bool(
            previous_fingerprint and previous_fingerprint != fingerprint
        )
        manifest["previous_config_sha256"] = previous_fingerprint
        manifest.setdefault("attempts", []).append(attempt)
    else:
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": f"{logical_id}-{timestamp}-{uuid.uuid4().hex[:8]}",
            "experiment_id": logical_id,
            "name": experiment.get("name") or logical_id,
            "group": experiment.get("group"),
            "run_type": run_type,
            "status": "running",
            "started_at_utc": _iso(now),
            "attempts": [attempt],
        }

    manifest.update(
        {
            "run_type": run_type,
            "tags": list(experiment.get("tags", [])),
            "hypothesis": experiment.get("hypothesis"),
            "notes": experiment.get("notes"),
            "parent_runs": list(experiment.get("parent_runs", [])),
            "config_path": str(config_path) if config_path is not None else None,
            "config_sha256": fingerprint,
            "parameters": parameters,
            "environment": _collect_run_metadata(
                run_type=run_type,
                config_path=config_path,
                extra=dict(extra or {}),
            ),
        }
    )
    _write_json(manifest, manifest_path)
    return manifest


def _read_first_json(output_dir, names):
    for name in names:
        path = output_dir / name
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(value, dict):
            return value
    return None


def finalize_experiment(
    output_dir,
    *,
    status="completed",
    summary=None,
):
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ended = _utc_now()
    manifest["status"] = status
    manifest["ended_at_utc"] = _iso(ended)
    started_raw = manifest.get("started_at_utc")
    if isinstance(started_raw, str):
        try:
            started = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
            manifest["duration_seconds"] = max(0.0, (ended - started).total_seconds())
        except ValueError:
            pass
    resolved_summary = (
        dict(summary)
        if summary is not None
        else _read_first_json(
            output_dir,
            (
                "run_summary.json",
                "policy_suite_summary.json",
                "eval_summary.json",
                "final_eval_metrics.json",
            ),
        )
    )
    if resolved_summary is not None:
        manifest["summary"] = _plain(resolved_summary)
    manifest["artifacts"] = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != MANIFEST_NAME
    )
    _write_json(manifest, manifest_path)
    return manifest


def load_parameters(path):
    """Load parameters from a run directory, manifest, or YAML config."""

    path = Path(path)
    if path.is_dir():
        manifest = path / MANIFEST_NAME
        resolved = path / "config_resolved.yaml"
        path = manifest if manifest.exists() else resolved
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name == MANIFEST_NAME:
            payload = payload.get("parameters", {})
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return _plain(payload)


def flatten_parameters(value, prefix=""):
    flat = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(flatten_parameters(item, name))
        else:
            flat[name] = _plain(item)
    return flat


def compare_parameters(left, right):
    left_flat = flatten_parameters(left)
    right_flat = flatten_parameters(right)
    missing = object()
    rows = []
    for key in sorted(set(left_flat) | set(right_flat)):
        left_value = left_flat.get(key, missing)
        right_value = right_flat.get(key, missing)
        if left_value == right_value:
            continue
        rows.append(
            {
                "parameter": key,
                "left": "<missing>" if left_value is missing else left_value,
                "right": "<missing>" if right_value is missing else right_value,
            }
        )
    return rows
