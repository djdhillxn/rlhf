#!/usr/bin/env python3
"""Show the exact resolved-configuration differences between two RLHF runs."""

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def _cell(value):
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two output directories, manifests, or YAML configuration files."
    )
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include differences under the non-behavioral experiment metadata block.",
    )
    args = parser.parse_args()

    ensure_repo_root_on_path()
    from rlhf.experiment import compare_parameters, load_parameters

    rows = compare_parameters(load_parameters(args.left), load_parameters(args.right))
    if not args.include_metadata:
        rows = [row for row in rows if not row["parameter"].startswith("experiment.")]
    if args.format == "json":
        rendered = json.dumps(rows, indent=2, ensure_ascii=False)
    elif rows:
        lines = ["| Parameter | Left | Right |", "|---|---|---|"]
        lines.extend(
            f"| `{row['parameter']}` | {_cell(row['left'])} | {_cell(row['right'])} |"
            for row in rows
        )
        rendered = "\n".join(lines)
    else:
        rendered = "No resolved-configuration differences."

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
