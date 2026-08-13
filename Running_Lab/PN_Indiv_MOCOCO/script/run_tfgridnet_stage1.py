"""Resolve a PN_Indiv_MOCOCO Stage1 config into its experiment directory and run it."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage0-encoder", required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--resume")
    args = parser.parse_args()

    override_path = Path(args.config).resolve()
    override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    base_path = (LAB_DIR / override["base_config"]).resolve()
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    merge(config, override)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (LAB_DIR / run_dir).resolve()
    stage_dir = run_dir / "stage1_tfgridnet"
    checkpoint_dir = stage_dir / "checkpoints"
    config["train"]["log_dir"] = str(stage_dir)
    config["checkpoint"]["dir"] = str(checkpoint_dir)
    config["checkpoint"]["resume"] = str(Path(args.resume).resolve()) if args.resume else None
    config["paths"]["encoder_override_ckpt"] = str(Path(args.stage0_encoder).resolve())
    config.setdefault("ddp", {})["num_gpus"] = int(args.num_gpus)
    config["experiment"]["run_dir"] = str(run_dir)

    stage_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = stage_dir / "config_tfgridnet_resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    sys.argv = ["train_tfgridnet_indiv.py", "--config", str(resolved_path)]
    import train_tfgridnet_indiv

    train_tfgridnet_indiv.main()


if __name__ == "__main__":
    main()
