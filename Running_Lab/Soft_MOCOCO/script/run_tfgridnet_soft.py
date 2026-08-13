from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

project_dir = Path(__file__).resolve().parents[1]
pn_mococo = project_dir.parent / "PN_MOCOCO"
sys.path.insert(0, str(pn_mococo))
sys.path.insert(0, str(project_dir))


def merge(base, override):
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
with open(args.config) as f:
    override = yaml.safe_load(f)
with open((project_dir / override["base_config"]).resolve()) as f:
    config = yaml.safe_load(f)
merge(config, override)
for section, keys in {
    "paths": ("initial_model_ckpt", "encoder_override_ckpt"),
    "train": ("log_dir",),
    "checkpoint": ("dir", "resume"),
    "eval": ("checkpoint", "out_json"),
}.items():
    for key in keys:
        value = config.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            config[section][key] = str((project_dir / value).resolve())

# Keep the resolved runtime configuration with the experiment artifacts rather
# than in a fixed experiment directory.
runtime = Path(config["train"]["log_dir"]).resolve().parent / "config_tfgridnet_soft_runtime.yaml"
runtime.parent.mkdir(parents=True, exist_ok=True)
runtime.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
sys.argv = ["train_tfgridnet_soft.py", "--config", str(runtime)]
import train_tfgridnet_soft

train_tfgridnet_soft.main()
