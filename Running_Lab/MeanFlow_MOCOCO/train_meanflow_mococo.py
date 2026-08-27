#!/usr/bin/env python3
"""Run sia_fm_tse's MeanFlow Stage1 with a MOCOCO Stage0 checkpoint.

The decoder, data adapter, and training loop remain in the sibling ``sia_fm_tse``
repository. This wrapper only resolves the two repositories and allows the frozen
MOCOCO encoder checkpoint to be selected without modifying the reference trainer.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
PNFLOW_ROOT = PROJECT_ROOT.parent
SIA_REPO = PNFLOW_ROOT / "sia_fm_tse"
SIA_TRAINER = SIA_REPO / "train_meanflow.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MeanFlow Stage1 using a frozen MOCOCO Stage0 encoder"
    )
    parser.add_argument(
        "--config",
        default=str(LAB_DIR / "configs/config_meanflow_mococo_soft50.yaml"),
    )
    parser.add_argument(
        "--stage0-ckpt",
        default=None,
        help="Override paths.pn_ckpt with a MOCOCO exported pn_encoder*.pt",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume Stage1 from a Lightning checkpoint, including optimizer state.",
    )
    return parser.parse_args()


def _load_sia_trainer():
    if not SIA_TRAINER.is_file():
        raise FileNotFoundError(f"sia_fm_tse trainer not found: {SIA_TRAINER}")

    # SIA packages must win for data/models/utils; pn_encoder.py is supplied by
    # the parent PNFlowTSE repository and loads the exported MOCOCO state dict.
    import_paths = (LAB_DIR, SIA_REPO, PNFLOW_ROOT, PROJECT_ROOT)
    for path in import_paths:
        while str(path) in sys.path:
            sys.path.remove(str(path))
    for path in reversed(import_paths):
        sys.path.insert(0, str(path))

    # The sibling repository's models/__init__.py imports an optional ECAPA
    # module that is not present in this checkout. Expose its actual models
    # directory as a lightweight namespace so UDiT and TPredicter can load
    # without importing that unrelated optional module.
    models_namespace = types.ModuleType("models")
    models_namespace.__path__ = [str(SIA_REPO / "models")]
    models_namespace.__package__ = "models"
    sys.modules["models"] = models_namespace

    # TPredicterPN imports ECAPA_TDNN as a sibling module. Its implementation
    # is shared by the parent PNFlowTSE repository, while this SIA checkout
    # does not carry that optional file.
    ecapa_path = PNFLOW_ROOT / "models" / "ecapa_tdnn.py"
    ecapa_spec = importlib.util.spec_from_file_location(
        "models.ecapa_tdnn", ecapa_path
    )
    if ecapa_spec is None or ecapa_spec.loader is None:
        raise ImportError(f"Unable to import shared ECAPA module: {ecapa_path}")
    ecapa_module = importlib.util.module_from_spec(ecapa_spec)
    sys.modules[ecapa_spec.name] = ecapa_module
    ecapa_spec.loader.exec_module(ecapa_module)

    spec = importlib.util.spec_from_file_location(
        "sia_fm_tse_train_meanflow", SIA_TRAINER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {SIA_TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_checkpoint_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PNFLOW_ROOT / candidate).resolve()


def _validate_tpredicter_encoder(config: dict) -> None:
    """Reject a t-predictor trained with a different frozen Stage0 encoder.

    The predictor consumes the PN embedding, so its input distribution is tied to
    the Stage0 checkpoint.  The old shared predictor was trained with
    ``proposed-monaural.pt`` and cannot be silently reused for a MOCOCO export.
    """
    flow_stage0 = _resolve_checkpoint_path(
        (config.get("paths", {}) or {}).get("pn_ckpt")
    )
    predictor_path = _resolve_checkpoint_path(
        (config.get("paths", {}) or {}).get("t_predicter_ckpt")
    )
    if flow_stage0 is None or predictor_path is None:
        return
    if not predictor_path.is_file():
        raise FileNotFoundError(f"t-predictor checkpoint not found: {predictor_path}")

    checkpoint = torch.load(predictor_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {}) or {}
    predictor_stage0 = _resolve_checkpoint_path(
        (hparams.get("paths", {}) or {}).get("pn_ckpt")
    )
    if predictor_stage0 is None:
        raise RuntimeError(
            "The t-predictor checkpoint has no recorded paths.pn_ckpt. "
            "Retrain it with the same Stage0 checkpoint used by MeanFlow."
        )
    if predictor_stage0 != flow_stage0:
        raise RuntimeError(
            "Incompatible t-predictor/Stage0 pair: "
            f"flow uses {flow_stage0}, but t-predictor was trained with "
            f"{predictor_stage0}. Retrain the t-predictor with the selected "
            "MOCOCO Stage0, or set paths.t_predicter_ckpt: null for an "
            "oracle-mixing-ratio diagnostic only."
        )


def _patch_rectified_flow_goal_time(trainer_module, config: dict) -> None:
    """Keep pure rectified-flow training and inference on the same r=t contract.

    ``loss_rectified_flow`` trains with r=t.  The reference validation loop
    requests r=1 for its one-step endpoint update, which activates the
    delta=r-t embedding at inference even though it was always zero in
    training.  MR-jitter/MeanFlow consistency modes intentionally train with
    r=1 and are therefore left untouched.
    """
    meanflow = config.get("meanflow", {}) or {}
    jitter = config.get("mr_jitter", {}) or {}
    pure_rectified = (
        float(meanflow.get("flow_ratio", 0.5)) >= 1.0
        and meanflow.get("alpha_schedule_end_epoch") is None
        and not bool(jitter.get("enabled", False))
    )
    if not pure_rectified:
        return

    udit = trainer_module.UDiT
    if getattr(udit, "_mococo_rectified_goal_patch", False):
        return
    original_forward = udit.forward

    def rectified_forward(self, x, t, r, enrollment):
        return original_forward(self, x, t, t, enrollment)

    udit.forward = rectified_forward
    udit._mococo_rectified_goal_patch = True
    print(
        "[MeanFlow_MOCOCO] pure rectified-flow compatibility: forcing r=t "
        "during validation/inference",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"MeanFlow config not found: {config_path}")

    stage0_ckpt = (
        Path(args.stage0_ckpt).expanduser().resolve()
        if args.stage0_ckpt
        else None
    )
    if stage0_ckpt is not None and not stage0_ckpt.is_file():
        raise FileNotFoundError(f"Stage0 checkpoint not found: {stage0_ckpt}")

    resume_ckpt = (
        Path(args.resume).expanduser().resolve()
        if args.resume
        else None
    )
    if resume_ckpt is not None and not resume_ckpt.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")

    # A resumed multi-GPU job can be blocked by a stale DataLoader worker or
    # multiprocessing temporary-directory cleanup before the next DDP epoch
    # collective.  Keep the experiment YAML unchanged, but allow a Slurm
    # rerun to use the main process for loading samples.
    num_workers_override = os.environ.get("MEANFLOW_NUM_WORKERS")
    if num_workers_override is not None:
        try:
            num_workers_override = int(num_workers_override)
        except ValueError as exc:
            raise ValueError(
                "MEANFLOW_NUM_WORKERS must be a non-negative integer"
            ) from exc
        if num_workers_override < 0:
            raise ValueError("MEANFLOW_NUM_WORKERS must be non-negative")

    trainer_module = _load_sia_trainer()
    original_parse_config = trainer_module.parse_config

    initial_config = original_parse_config(str(config_path))
    if stage0_ckpt is not None:
        initial_config.setdefault("paths", {})["pn_ckpt"] = str(stage0_ckpt)
        initial_config.setdefault("enroll", {})["pn_ckpt"] = str(stage0_ckpt)
    if resume_ckpt is not None:
        initial_config.setdefault("checkpoint", {})["resume"] = str(resume_ckpt)
    if num_workers_override is not None:
        initial_config.setdefault("train", {})["num_workers"] = num_workers_override
        print(
            "[MeanFlow_MOCOCO] runtime override: "
            f"train.num_workers={num_workers_override}",
            flush=True,
        )
    _validate_tpredicter_encoder(initial_config)
    _patch_rectified_flow_goal_time(trainer_module, initial_config)

    def parse_config_with_mococo(path):
        config = original_parse_config(path)
        if stage0_ckpt is not None:
            config.setdefault("paths", {})["pn_ckpt"] = str(stage0_ckpt)
            config.setdefault("enroll", {})["pn_ckpt"] = str(stage0_ckpt)
        if resume_ckpt is not None:
            config.setdefault("checkpoint", {})["resume"] = str(resume_ckpt)
        if num_workers_override is not None:
            config.setdefault("train", {})["num_workers"] = num_workers_override
        _validate_tpredicter_encoder(config)
        return config

    trainer_module.parse_config = parse_config_with_mococo
    sys.argv = [sys.argv[0], "--config", str(config_path)]
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/meanflow_mococo_numba_cache")
    trainer_module.main()


if __name__ == "__main__":
    main()
