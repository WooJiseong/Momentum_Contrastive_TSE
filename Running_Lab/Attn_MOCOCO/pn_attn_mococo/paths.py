from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PN_MOCOCO_DIR = PROJECT_DIR.parent / "PN_MOCOCO"
PNFLOW_ROOT = PROJECT_DIR.parents[2]


def add_repo_paths() -> None:
    for path in (PROJECT_DIR, PN_MOCOCO_DIR, PNFLOW_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def project_path(path: str | Path) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else (PROJECT_DIR / value).resolve())


def normalize_training_paths(cfg: dict) -> dict:
    """Resolve Attn_MOCOCO artifacts relative to this Lab."""
    cfg = dict(cfg)
    cfg.setdefault("paths", {})
    cfg.setdefault("checkpoint", {})

    def resolve(section: str, key: str) -> None:
        value = cfg.get(section, {}).get(key)
        if value:
            cfg[section][key] = project_path(value)

    resolve("train", "log_dir")
    resolve("checkpoint", "dir")
    resolve("checkpoint", "resume")
    resolve("paths", "initial_pn_ckpt")
    resolve("contrastive", "initial_pn_ckpt")
    return cfg


add_repo_paths()
