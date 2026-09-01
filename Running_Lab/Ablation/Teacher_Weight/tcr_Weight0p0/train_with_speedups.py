#!/usr/bin/env python3
"""Load the repository speedups before starting the unchanged Improved_Attn trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path("/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum")
SPEEDUP_DIR = PROJECT_ROOT / "Base" / "Code_Snippet"
TRAINER = (
    PROJECT_ROOT
    / "Running_Lab"
    / "Improved_Attn_MOCOCO"
    / "train_improved_attn_moco_encoder.py"
)

sys.path.insert(0, str(SPEEDUP_DIR))
import speedups  # noqa: F401,E402  # must run before espnet/resemblyzer imports

runpy.run_path(str(TRAINER), run_name="__main__")
