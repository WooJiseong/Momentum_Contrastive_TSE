from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent
pn_mococo = project_dir.parent / "PN_MOCOCO"
sys.path.insert(0, str(pn_mococo))
sys.path.insert(0, str(project_dir))

base_spec = importlib.util.spec_from_file_location(
    "pn_mococo_base_train_attn_tfgridnet",
    pn_mococo / "train_tfgridnet.py",
)
if base_spec is None or base_spec.loader is None:
    raise ImportError(f"Cannot load PN_MOCOCO trainer from {pn_mococo / 'train_tfgridnet.py'}")
base_train = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base_train
base_spec.loader.exec_module(base_train)
base_train.main()
