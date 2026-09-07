#!/usr/bin/env python3
"""Run PN_MOCOCO TFGridNet Stage1 with the shared speedups patch installed first."""

from __future__ import annotations

import sys
import os
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parents[1]
SNIPPET_DIR = ROOT / "Base" / "Code_Snippet"
PN_MOCOCO_DIR = ROOT / "Running_Lab" / "PN_MOCOCO"

# This must happen before importing PN_MOCOCO/ESPnet modules.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    f"/tmp/dsnos_moco_stage1_tfgridnet_{os.environ.get('USER', 'user')}",
)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SNIPPET_DIR))
import speedups  # noqa: F401,E402

sys.path.insert(0, str(PN_MOCOCO_DIR))
from train_tfgridnet import main  # noqa: E402


if __name__ == "__main__":
    main()
