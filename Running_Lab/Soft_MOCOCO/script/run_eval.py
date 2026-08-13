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


def project_path(value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (project_dir / path).resolve())


def resolve_runtime_paths(config: dict) -> None:
    for section, keys in {
        "paths": ("initial_model_ckpt", "encoder_override_ckpt"),
        "train": ("log_dir",),
        "checkpoint": ("dir", "resume"),
        "eval": ("checkpoint", "out_json"),
    }.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value:
                config[section][key] = project_path(value)


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config_tfgridnet_soft.yaml")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--out", default=None)
args = parser.parse_args()

with open(args.config) as f:
    override = yaml.safe_load(f)

if "base_config" in override:
    with open((project_dir / override["base_config"]).resolve()) as f:
        config = yaml.safe_load(f)
    merge(config, override)
else:
    config = override

eval_cfg = config.setdefault("eval", {})
if args.checkpoint:
    eval_cfg["checkpoint"] = args.checkpoint
if args.out:
    eval_cfg["out_json"] = args.out

resolve_runtime_paths(config)

# Store the generated configuration alongside this experiment's evaluation
# artifacts, not under a hard-coded run directory.
runtime = Path(config["eval"]["out_json"]).resolve().parents[1] / "config_eval_runtime.yaml"
runtime.parent.mkdir(parents=True, exist_ok=True)
runtime.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

sys.argv = ["eval_tfgridnet.py", "--config", str(runtime)]
if args.checkpoint:
    sys.argv += ["--checkpoint", project_path(args.checkpoint)]
if args.out:
    sys.argv += ["--out", project_path(args.out)]

import eval_tfgridnet

eval_tfgridnet.main()
