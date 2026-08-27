"""Import helpers that keep sia_fm_tse read-only while swapping its PN loader."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
PNFLOW_ROOT = PROJECT_ROOT.parent
SIA_REPO = PNFLOW_ROOT / "sia_fm_tse"
PN_MOCOCO_DIR = PROJECT_ROOT / "Running_Lab" / "PN_MOCOCO"
BASE_DIR = PROJECT_ROOT / "Base" / "TSE-through-Positive-Negative-Enroll"


def prepare_imports() -> None:
    paths = (LAB_DIR, SIA_REPO, PNFLOW_ROOT, PROJECT_ROOT, PN_MOCOCO_DIR, BASE_DIR)
    for path in paths:
        while str(path) in sys.path:
            sys.path.remove(str(path))
    for path in reversed(paths):
        sys.path.insert(0, str(path))

    # Avoid importing the optional modules package initializer from the sibling
    # checkout.  This is the same namespace setup used by the existing wrapper.
    models_namespace = types.ModuleType("models")
    models_namespace.__path__ = [str(SIA_REPO / "models")]
    models_namespace.__package__ = "models"
    sys.modules["models"] = models_namespace

    ecapa_path = PNFLOW_ROOT / "models" / "ecapa_tdnn.py"
    ecapa_spec = importlib.util.spec_from_file_location("models.ecapa_tdnn", ecapa_path)
    if ecapa_spec is None or ecapa_spec.loader is None:
        raise ImportError(f"Unable to import shared ECAPA module: {ecapa_path}")
    ecapa_module = importlib.util.module_from_spec(ecapa_spec)
    sys.modules[ecapa_spec.name] = ecapa_module
    ecapa_spec.loader.exec_module(ecapa_module)

    # train_meanflow.py and train_t_predicter_pn.py import this name lazily.
    from improved_pn_encoder import load_pn_encoder

    local_pn = types.ModuleType("pn_encoder")
    local_pn.load_pn_encoder = load_pn_encoder
    local_pn.__file__ = str(LAB_DIR / "improved_pn_encoder.py")
    sys.modules["pn_encoder"] = local_pn


def load_module(module_name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
