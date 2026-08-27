#!/usr/bin/env python3
"""Merge condition-part results produced by parallel MeanFlow evaluators."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from eval_meanflow_four_conditions import CONDITION_LABELS, write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    part_dirs = [Path(path).resolve() for path in args.part_dir]
    out_dir = Path(args.out_dir).resolve()
    out_raw = out_dir / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    merged_conditions = {}
    metadata = None
    for part_dir in part_dirs:
        summary_path = part_dir / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if metadata is None:
            metadata = payload["metadata"]
        merged_conditions.update(payload.get("conditions", {}))
        for raw_path in (part_dir / "raw").glob("*.json"):
            shutil.copy2(raw_path, out_raw / raw_path.name)

    if metadata is None:
        raise RuntimeError("No evaluation part summaries were found")
    metadata = dict(metadata)
    metadata["requested_conditions"] = list(CONDITION_LABELS)
    metadata["parallel_parts"] = [str(path) for path in part_dirs]
    write_summary(out_dir, metadata, merged_conditions)
    print(f"[merge] completed: {out_dir / 'result_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
