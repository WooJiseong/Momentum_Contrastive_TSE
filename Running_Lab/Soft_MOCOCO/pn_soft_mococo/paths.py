from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PNFLOW_ROOT = PROJECT_DIR.parents[2]
PN_MOCOCO_DIR = PROJECT_DIR.parent / "PN_MOCOCO"


def add_repo_paths() -> None:
    for path in (PROJECT_DIR, PN_MOCOCO_DIR, PNFLOW_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def project_path(path: str | Path) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else (PROJECT_DIR / value).resolve())


def normalize_training_paths(cfg: dict) -> dict:
    """Resolve Soft_MOCOCO training artifact paths relative to this project."""
    cfg = dict(cfg)

    def resolve(section: str, key: str) -> None:
        value = cfg.get(section, {}).get(key)
        if value:
            cfg[section][key] = project_path(value)

    cfg.setdefault("paths", {})
    cfg.setdefault("checkpoint", {})
    resolve("train", "log_dir")
    resolve("checkpoint", "dir")
    resolve("checkpoint", "resume")
    resolve("paths", "initial_pn_ckpt")
    resolve("contrastive", "initial_pn_ckpt")
    return cfg


add_repo_paths()
