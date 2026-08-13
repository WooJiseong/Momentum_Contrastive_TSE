from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


LAB_DIR = Path(__file__).resolve().parents[1]
PN_MOCOCO_DIR = LAB_DIR.parent / "PN_MOCOCO"
sys.path.insert(0, str(PN_MOCOCO_DIR))
sys.path.insert(0, str(LAB_DIR))


def merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


def resolve_paths(config: dict) -> None:
    for section, keys in {
        "paths": ("initial_model_ckpt", "encoder_override_ckpt"),
        "train": ("log_dir",),
        "checkpoint": ("dir", "resume"),
        "eval": ("checkpoint", "out_json"),
    }.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value and not Path(value).is_absolute():
                config[section][key] = str((LAB_DIR / value).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    override_path = Path(args.config).resolve()
    override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    base_path = (LAB_DIR / override["base_config"]).resolve()
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    merge(config, override)
    resolve_paths(config)
    stage_dir = Path(config["train"]["log_dir"])
    resolved = stage_dir / "config_tfgridnet_resolved.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    sys.argv = ["train_tfgridnet_soft_indiv.py", "--config", str(resolved)]
    import train_tfgridnet_soft_indiv

    train_tfgridnet_soft_indiv.main()


if __name__ == "__main__":
    main()
