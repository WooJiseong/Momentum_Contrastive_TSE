from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml

project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir))


def merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


parser = argparse.ArgumentParser(description="Run Attn_MOCOCO Stage1.")
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config, encoding="utf-8") as handle:
    override = yaml.safe_load(handle)
with open((project_dir / override["base_config"]).resolve(), encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
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

runtime = Path(config["train"]["log_dir"]).resolve().parent / "config_tfgridnet_attn_runtime.yaml"
runtime.parent.mkdir(parents=True, exist_ok=True)
runtime.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

sys.argv = ["train_tfgridnet_attn.py", "--config", str(runtime)]
if config.get("loss", {}).get("type") == "si_sdr_plus_target_orthogonal_leakage":
    importlib.import_module("train_tfgridnet_attn_leakage")
else:
    importlib.import_module("train_tfgridnet_attn")
