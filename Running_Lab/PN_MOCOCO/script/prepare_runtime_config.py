#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from shlex import quote

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create exp_setting-compliant runtime configs for PN_MOCOCO.")
    parser.add_argument("--moco-config", required=True)
    parser.add_argument("--tfgrid-config", required=True)
    parser.add_argument("--mode", choices=("all", "train", "eval", "moco", "tfgridnet"), required=True)
    parser.add_argument("--run-dir", default=os.environ.get("RUN_DIR", ""))
    parser.add_argument("--date", default=os.environ.get("EXP_DATE", ""))
    parser.add_argument("--name", default=os.environ.get("EXP_NAME", ""))
    return parser.parse_args()


def rel(path: Path) -> str:
    return os.path.relpath(path, PROJECT_DIR)


def unique_run_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for idx in range(1, 1000):
        candidate = base.with_name(f"{base.name}_run{idx:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No free run directory for {base}")


def read_yaml(path: str) -> tuple[Path, dict]:
    p = (PROJECT_DIR / path).resolve() if not Path(path).is_absolute() else Path(path)
    return p, yaml.safe_load(p.read_text())


def writes_moco(mode: str) -> bool:
    return mode in ("all", "train", "moco")


def writes_tfgrid(mode: str) -> bool:
    return mode in ("all", "train", "tfgridnet")


def main() -> None:
    args = parse_args()
    moco_path, moco_cfg = read_yaml(args.moco_config)
    tfgrid_path, tfgrid_cfg = read_yaml(args.tfgrid_config)

    base_experiment = tfgrid_cfg.setdefault("experiment", {})
    name = args.name or base_experiment.get("name") or "pn_mococo"
    if args.mode == "eval" and not name.endswith("_eval"):
        name = f"{name}_eval"
    date = args.date or base_experiment.get("date") or datetime.now().strftime("%Y%m%d")

    reuse_existing = False
    if args.run_dir:
        run_dir = (PROJECT_DIR / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
        if run_dir.exists() and os.environ.get("ALLOW_EXISTING_RUN_DIR", "0") != "1":
            raise RuntimeError(
                f"RUN_DIR already exists: {run_dir}. Set ALLOW_EXISTING_RUN_DIR=1 to reuse it explicitly."
            )
        reuse_existing = run_dir.exists()
    else:
        run_dir = unique_run_dir(PROJECT_DIR / "exp" / f"{date}_{name}")
    run_dir.mkdir(parents=True, exist_ok=reuse_existing)

    run_rel = rel(run_dir)
    moco_exp = moco_cfg.setdefault("experiment", {})
    tfgrid_exp = tfgrid_cfg.setdefault("experiment", {})
    for exp in (moco_exp, tfgrid_exp):
        exp.update(
            {
                "name": name.removesuffix("_eval"),
                "run_dir": run_rel,
            }
        )

    if writes_moco(args.mode):
        moco_stage = run_dir / "stage0_moco"
        moco_cfg.setdefault("train", {})["log_dir"] = rel(moco_stage)
        moco_cfg.setdefault("checkpoint", {})["dir"] = rel(moco_stage / "checkpoints")

    if writes_tfgrid(args.mode):
        tfgrid_stage = run_dir / "stage1_tfgridnet"
        tfgrid_cfg.setdefault("train", {})["log_dir"] = rel(tfgrid_stage)
        tfgrid_cfg.setdefault("checkpoint", {})["dir"] = rel(tfgrid_stage / "checkpoints")
        if writes_moco(args.mode):
            tfgrid_cfg.setdefault("paths", {})["encoder_override_ckpt"] = rel(
                run_dir / "stage0_moco" / "checkpoints" / "pn_encoder_best.pt"
            )

    tfgrid_cfg.setdefault("eval", {})["out_json"] = f"{run_rel}/evaluation/results.json"
    if writes_tfgrid(args.mode):
        ckpt_name = tfgrid_cfg.get("checkpoint", {}).get("ckpt_name", "tfgridnet_best")
        tfgrid_cfg["eval"]["checkpoint"] = f"{run_rel}/stage1_tfgridnet/checkpoints/{ckpt_name}.ckpt"

    moco_exp.update(
        {
            "source_config": rel(moco_path),
            "runtime_config": f"{run_rel}/config_moco_encoder.yaml",
        }
    )
    tfgrid_exp.update(
        {
            "source_config": rel(tfgrid_path),
            "runtime_config": f"{run_rel}/config_tfgridnet_supervised.yaml",
        }
    )

    moco_runtime = run_dir / "config_moco_encoder.yaml"
    tfgrid_runtime = run_dir / "config_tfgridnet_supervised.yaml"
    moco_runtime.write_text(yaml.safe_dump(moco_cfg, sort_keys=False), encoding="utf-8")
    tfgrid_runtime.write_text(yaml.safe_dump(tfgrid_cfg, sort_keys=False), encoding="utf-8")
    shutil.copy2(moco_path, run_dir / "source_config_moco_encoder.yaml")
    shutil.copy2(tfgrid_path, run_dir / "source_config_tfgridnet_supervised.yaml")

    print(f"MOCO_CFG={quote(rel(moco_runtime))}")
    print(f"TFGRID_CFG={quote(rel(tfgrid_runtime))}")
    print(f"RUN_DIR={quote(run_rel)}")


if __name__ == "__main__":
    main()
