#!/usr/bin/env python3
"""Run the unchanged Soft_MOCOCO trainer after installing repository speedups."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPEEDUP_DIR = PROJECT_ROOT / "Base" / "Code_Snippet"
TRAINER = PROJECT_ROOT / "Running_Lab" / "Soft_MOCOCO" / "train_soft_moco_encoder.py"

sys.path.insert(0, str(SPEEDUP_DIR))
import speedups  # noqa: F401,E402  # must run before espnet/resemblyzer imports

runpy.run_path(str(TRAINER), run_name="__main__")
