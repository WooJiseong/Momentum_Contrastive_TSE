#!/usr/bin/env python3
"""Train the SIA t-predictor against the final improved Stage0 export."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sia_runtime import LAB_DIR, PNFLOW_ROOT, prepare_imports, load_module


TPRED_TRAINER = PNFLOW_ROOT / "train_t_predicter_pn.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(LAB_DIR / "configs/config_tpred.yaml"))
    parser.add_argument("--stage0-ckpt", default=os.environ.get("STAGE0_CKPT"))
    parser.add_argument(
        "--run-root",
        default=None,
        help="Override this run's log/checkpoint root without changing the read-only SIA config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    stage0 = Path(args.stage0_ckpt).expanduser().resolve() if args.stage0_ckpt else None
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if stage0 is None or not stage0.is_file():
        raise FileNotFoundError(f"Stage0 checkpoint not found: {stage0}")

    prepare_imports()
    trainer = load_module("improved_attn_tpred_trainer", TPRED_TRAINER)
    original_parse_config = trainer.parse_config
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else None

    def parse_config_with_stage0(path):
        config = original_parse_config(path)
        config.setdefault("paths", {})["pn_ckpt"] = str(stage0)
        if run_root is not None:
            config.setdefault("train", {})["log_dir"] = str(run_root / "logs")
            config.setdefault("checkpoint", {})["dir"] = str(run_root / "checkpoints")
        return config

    trainer.parse_config = parse_config_with_stage0
    sys.argv = [sys.argv[0], "--config", str(config_path)]
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/improved_attn_tpred_numba_cache")
    trainer.main()


if __name__ == "__main__":
    main()
