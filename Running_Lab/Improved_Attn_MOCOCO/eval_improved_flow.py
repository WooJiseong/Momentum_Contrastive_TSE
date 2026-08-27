#!/usr/bin/env python3
"""Run the unchanged sia_fm_tse Base benchmark with the improved PN loader."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

from sia_runtime import SIA_REPO, prepare_imports, load_module


LAB_DIR = Path(__file__).resolve().parent
EVAL_PATH = SIA_REPO / "eval" / "eval_benchmark_paper_metrics_v3.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(LAB_DIR / "configs/config_flow_oracle.yaml"))
    parser.add_argument("--stage0-ckpt", required=True)
    parser.add_argument("--flow-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--t-mode", choices=("oracle", "mean", "predicted"), default="oracle")
    parser.add_argument("--tpred-ckpt", default=None)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("paths", {})["pn_ckpt"] = str(Path(args.stage0_ckpt).expanduser().resolve())
    config.setdefault("enroll", {})["pn_ckpt"] = config["paths"]["pn_ckpt"]

    prepare_imports()
    evaluator = load_module("improved_attn_base_evaluator", EVAL_PATH)

    # The evaluator opens its config itself.  A temporary copy lets us inject
    # the exact Stage0 path without modifying the read-only reference script.
    with tempfile.TemporaryDirectory(prefix="improved_attn_eval_") as tmp:
        config_path = Path(tmp) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        sys.argv = [
            sys.argv[0],
            "--config", str(config_path),
            "--model", "flow",
            "--ckpt", str(Path(args.flow_ckpt).expanduser().resolve()),
            "--n", str(args.n),
            "--nfe-list", "1",
            "--t-mode", args.t_mode,
            "--split", args.split,
            "--batch-size", str(args.batch_size),
            "--out", str(Path(args.out).expanduser().resolve()),
        ]
        if args.tpred_ckpt:
            sys.argv.extend(["--tpred-ckpt", str(Path(args.tpred_ckpt).expanduser().resolve())])
        evaluator.main()


if __name__ == "__main__":
    main()

