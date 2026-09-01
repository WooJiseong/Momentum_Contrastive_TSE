#!/usr/bin/env python3
"""Run the Improved Attn Flow trainer after installing repository speedups."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[2]
SPEEDUPS_DIR = PROJECT_ROOT / "Base" / "Code_Snippet"
TRAINER = LAB_DIR / "train_improved_meanflow.py"

if str(SPEEDUPS_DIR) not in sys.path:
    sys.path.insert(0, str(SPEEDUPS_DIR))

# This must happen before the trainer imports the TFGridNet/resemblyzer stack.
import speedups  # noqa: F401,E402

runpy.run_path(str(TRAINER), run_name="__main__")
