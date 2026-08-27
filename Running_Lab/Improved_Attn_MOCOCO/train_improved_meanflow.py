#!/usr/bin/env python3
"""Train the read-only sibling SIA Flow Generator with the improved PN path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from sia_runtime import LAB_DIR, SIA_REPO, prepare_imports, load_module


SIA_TRAINER = SIA_REPO / "train_meanflow.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(LAB_DIR / "configs/config_flow_oracle.yaml"))
    parser.add_argument("--stage0-ckpt", default=os.environ.get("STAGE0_CKPT"))
    parser.add_argument(
        "--run-root",
        default=None,
        help="Override this run's log/checkpoint root without changing the read-only SIA config.",
    )
    return parser.parse_args()


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (LAB_DIR / candidate).resolve()


def validate_tpred_encoder(config: dict) -> None:
    flow_path = resolve((config.get("paths", {}) or {}).get("pn_ckpt"))
    predictor_path = resolve((config.get("paths", {}) or {}).get("t_predicter_ckpt"))
    if flow_path is None or predictor_path is None:
        return
    if not predictor_path.is_file():
        raise FileNotFoundError(predictor_path)
    checkpoint = torch.load(predictor_path, map_location="cpu", weights_only=False)
    predictor_path_cfg = ((checkpoint.get("hyper_parameters", {}) or {}).get("paths", {}) or {}).get("pn_ckpt")
    predictor_stage0 = resolve(predictor_path_cfg)
    if predictor_stage0 != flow_path:
        raise RuntimeError(
            f"t-predictor was trained with {predictor_stage0}, but Flow uses {flow_path}"
        )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    stage0 = resolve(args.stage0_ckpt)
    if stage0 is None or not stage0.is_file():
        raise FileNotFoundError(f"Stage0 checkpoint not found: {stage0}")

    prepare_imports()
    trainer = load_module("improved_attn_sia_meanflow", SIA_TRAINER)
    original_parse_config = trainer.parse_config
    initial_config = original_parse_config(str(config_path))
    initial_config.setdefault("paths", {})["pn_ckpt"] = str(stage0)
    initial_config.setdefault("enroll", {})["pn_ckpt"] = str(stage0)
    run_root = resolve(args.run_root)
    if run_root is not None:
        initial_config.setdefault("train", {})["log_dir"] = str(run_root / "logs")
        initial_config.setdefault("checkpoint", {})["dir"] = str(run_root / "checkpoints")
        initial_config.setdefault("eval", {})["checkpoint"] = str(
            run_root / "checkpoints" / "improved_attn_flow_oracle_best.ckpt"
        )
    validate_tpred_encoder(initial_config)

    def parse_config_with_stage0(path):
        config = original_parse_config(path)
        config.setdefault("paths", {})["pn_ckpt"] = str(stage0)
        config.setdefault("enroll", {})["pn_ckpt"] = str(stage0)
        if run_root is not None:
            config.setdefault("train", {})["log_dir"] = str(run_root / "logs")
            config.setdefault("checkpoint", {})["dir"] = str(run_root / "checkpoints")
            config.setdefault("eval", {})["checkpoint"] = str(
                run_root / "checkpoints" / "improved_attn_flow_oracle_best.ckpt"
            )
        validate_tpred_encoder(config)
        return config

    trainer.parse_config = parse_config_with_stage0
    sys.argv = [sys.argv[0], "--config", str(config_path)]
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/improved_attn_meanflow_numba_cache")
    trainer.main()


if __name__ == "__main__":
    main()
