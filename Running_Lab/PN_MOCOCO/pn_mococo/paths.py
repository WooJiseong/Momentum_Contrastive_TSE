from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRASTIVE_ROOT = PROJECT_DIR.parents[1]
PNFLOW_ROOT = PROJECT_DIR.parents[2]


def add_repo_paths() -> None:
    """Make local PN_MOCOCO modules and top-level PNFlowTSE modules importable."""
    for path in (PROJECT_DIR, CONTRASTIVE_ROOT, PNFLOW_ROOT):
        s = str(path)
        while s in sys.path:
            sys.path.remove(s)
    for path in (PROJECT_DIR, CONTRASTIVE_ROOT, PNFLOW_ROOT):
        sys.path.insert(0, str(path))


def project_path(path: str | None) -> str | None:
    """Resolve a config path relative to PN_MOCOCO unless it is empty or absolute."""
    if path is None or path == "":
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_DIR / p).resolve())


def root_path(path: str | None) -> str | None:
    """Resolve a config path relative to the PNFlowTSE root unless it is empty or absolute."""
    if path is None or path == "":
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PNFLOW_ROOT / p).resolve())
