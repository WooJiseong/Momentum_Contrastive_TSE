#!/usr/bin/env python3
"""Train the MeanFlow t-predictor with the selected MOCOCO Stage0 encoder."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
PNFLOW_ROOT = PROJECT_ROOT.parent
TRAINER_PATH = PNFLOW_ROOT / "train_t_predicter_pn.py"


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train t-predictor with a MOCOCO Stage0 checkpoint"
    )
    parser.add_argument(
        "--config",
        default=str(LAB_DIR / "configs/config_tpred_mococo_soft50.yaml"),
    )
    parser.add_argument(
        "--stage0-ckpt",
        default=os.environ.get("STAGE0_CKPT"),
        required=False,
    )
    return parser.parse_args()


def _load_trainer():
    if not TRAINER_PATH.is_file():
        raise FileNotFoundError(f"t-predictor trainer not found: {TRAINER_PATH}")

    for path in (LAB_DIR, PNFLOW_ROOT, PROJECT_ROOT):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    for path in (PROJECT_ROOT, PNFLOW_ROOT, LAB_DIR):
        sys.path.insert(0, str(path))

    spec = importlib.util.spec_from_file_location(
        "mococo_t_predicter_trainer", TRAINER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = _parse_args()
    if not args.stage0_ckpt:
        raise ValueError(
            "Stage0 checkpoint is required via --stage0-ckpt or STAGE0_CKPT."
        )
    config_path = Path(args.config).expanduser().resolve()
    stage0_path = Path(args.stage0_ckpt).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"t-predictor config not found: {config_path}")
    if not stage0_path.is_file():
        raise FileNotFoundError(f"Stage0 checkpoint not found: {stage0_path}")

    trainer = _load_trainer()
    original_parse_config = trainer.parse_config

    def parse_config_with_stage0(path):
        config = original_parse_config(path)
        config.setdefault("paths", {})["pn_ckpt"] = str(stage0_path)
        return config

    trainer.parse_config = parse_config_with_stage0
    sys.argv = [sys.argv[0], "--config", str(config_path)]
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/meanflow_mococo_tpred_numba_cache")
    trainer.main()


if __name__ == "__main__":
    main()
