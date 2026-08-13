from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRASTIVE_ROOT = PROJECT_DIR.parents[1]
PNFLOW_ROOT = PROJECT_DIR.parents[2]
PN_MOCOCO_DIR = PROJECT_DIR.parent / "PN_MOCOCO"


def add_repo_paths() -> None:
    for path in (PROJECT_DIR, PN_MOCOCO_DIR, CONTRASTIVE_ROOT, PNFLOW_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def project_path(path: str | Path) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else (PROJECT_DIR / value).resolve())


add_repo_paths()
