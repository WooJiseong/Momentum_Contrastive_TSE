#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a completed PN_MOCOCO evaluation run.")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def main() -> None:
    args = parse_args()
    run_dir = project_path(args.run_dir)
    config_path = run_dir / "config_tfgridnet_supervised.yaml"
    if not config_path.is_file():
        raise SystemExit(f"missing config: {config_path}")

    config = yaml.safe_load(config_path.read_text())
    eval_cfg = config.get("eval", {})
    out_json = project_path(eval_cfg.get("out_json", str(run_dir / "evaluation" / "results.json")))
    expected_n = int(eval_cfg.get("test_n", 0))
    if not out_json.is_file():
        raise SystemExit(f"missing evaluation json: {out_json}")

    result = json.loads(out_json.read_text())
    actual_n = int(result.get("n", -1))
    if actual_n != expected_n:
        raise SystemExit(f"unexpected eval count: expected {expected_n}, got {actual_n}")
    for key in ("si_sdr", "si_sdri", "snr", "snri"):
        if key not in result.get("summary", {}):
            raise SystemExit(f"missing summary metric: {key}")

    summary_path = run_dir / "result_summary.md"
    if not summary_path.is_file():
        raise SystemExit(f"missing result summary: {summary_path}")
    for name in ("command.txt", "environment.txt"):
        metadata_path = run_dir / name
        if not metadata_path.is_file():
            raise SystemExit(f"missing runtime metadata: {metadata_path}")
    print(f"ok: {run_dir} n={actual_n}")


if __name__ == "__main__":
    main()
