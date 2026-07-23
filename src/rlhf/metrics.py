import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


def masked_mean(x, mask, eps=1e-8):
    if x.shape != mask.shape:
        raise ValueError(
            f"masked_mean shape mismatch: x.shape={tuple(x.shape)} mask.shape={tuple(mask.shape)}"
        )
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum() / mask_f.sum().clamp_min(eps)


def masked_var(x, mask, eps=1e-8):
    mean = masked_mean(x, mask, eps=eps)
    return masked_mean((x - mean) ** 2, mask, eps=eps)


def explained_variance(y_pred, y_true, mask=None):
    with torch.no_grad():
        if mask is not None:
            if y_pred.shape != mask.shape or y_true.shape != mask.shape:
                raise ValueError(
                    "explained_variance shape mismatch: "
                    f"y_pred={tuple(y_pred.shape)} y_true={tuple(y_true.shape)} mask={tuple(mask.shape)}"
                )
            valid = mask.bool()
            y_pred = y_pred[valid]
            y_true = y_true[valid]
        if y_true.numel() < 2:
            return float("nan")
        var_y = torch.var(y_true.float(), unbiased=False)
        if var_y.item() == 0:
            return float("nan")
        return float(
            1.0
            - torch.var((y_true - y_pred).float(), unbiased=False).item() / var_y.item()
        )


def append_jsonl(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def write_csv(records, path):
    records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in records for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def jsonl_to_csv(jsonl_path, csv_path):
    write_csv(read_jsonl(jsonl_path), csv_path)


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def collect_run_metadata(*, run_type, config_path=None, extra=None):
    gpu_info = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu_info.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    record = {
        "run_type": run_type,
        "created_unix_time": time.time(),
        "created_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(config_path) if config_path is not None else None,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_info": gpu_info,
        "git_commit": _git_commit(),
    }
    if extra:
        record.update(extra)
    return record


def save_metric_plots(
    records,
    output_dir,
    *,
    x_key,
    y_keys,
    prefix,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []
    for y_key in y_keys:
        xs = []
        ys = []
        for row in records:
            if x_key not in row or y_key not in row:
                continue
            try:
                x = float(row[x_key])
                y = float(row[y_key])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"{prefix}: {y_key}")
        ax.grid(True, alpha=0.3)
        path = output_dir / f"{prefix}_{y_key}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths
