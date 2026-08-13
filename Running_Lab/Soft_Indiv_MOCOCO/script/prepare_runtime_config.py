#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


LAB_DIR = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (LAB_DIR / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create immutable-per-attempt runtime configs for Soft_Indiv_MOCOCO.")
    parser.add_argument("--stage", required=True, choices=("stage0", "stage1"))
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-gpus", required=True, type=int)
    parser.add_argument("--stage-label", default="")
    parser.add_argument("--stage0-encoder", default="")
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def write_runtime_config(path: Path, config: dict, resume: bool) -> Path:
    if path.exists() and not resume:
        raise RuntimeError(f"Refusing to overwrite existing runtime config: {path}")
    text = yaml.safe_dump(config, sort_keys=False)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return path
    candidate = path
    for attempt in range(1, 1000):
        if not candidate.exists():
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(text, encoding="utf-8")
            return candidate
        candidate = path.with_name(f"{path.stem}_attempt{attempt:02d}{path.suffix}")
    raise RuntimeError(f"No free runtime-config path for {path}")


def main() -> None:
    args = parse_args()
    source = resolve(args.source_config)
    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot = run_dir / f"source_config_{args.stage}.yaml"
    if not snapshot.exists():
        shutil.copy2(source, snapshot)

    resume = resolve(args.resume) if args.resume else None
    if args.stage == "stage0":
        stage_dir = run_dir / "stage0_moco"
        initial_runtime = stage_dir / "config_stage0_runtime.yaml"
        config = yaml.safe_load(initial_runtime.read_text(encoding="utf-8")) if resume and initial_runtime.exists() else yaml.safe_load(source.read_text(encoding="utf-8"))
        config["experiment"].update({"name": run_dir.name, "run_dir": str(run_dir)})
        config["train"]["log_dir"] = str(stage_dir)
        config["checkpoint"]["dir"] = str(stage_dir / "checkpoints")
        config["checkpoint"]["resume"] = str(resume) if resume else None
        config.setdefault("ddp", {})["num_gpus"] = args.num_gpus
        name = "config_stage0_runtime.yaml" if resume is None else f"config_stage0_resume_{resume.stem}.yaml"
    else:
        if not args.stage_label or not args.stage0_encoder:
            raise ValueError("stage1 requires --stage-label and --stage0-encoder")
        stage_dir = run_dir / "stage1_sweep" / args.stage_label
        initial_runtime = stage_dir / "config_stage1_runtime.yaml"
        config = yaml.safe_load(initial_runtime.read_text(encoding="utf-8")) if resume and initial_runtime.exists() else yaml.safe_load(source.read_text(encoding="utf-8"))
        encoder = resolve(args.stage0_encoder)
        config["experiment"].update({
            "name": f"{run_dir.name}_{args.stage_label}",
            "run_dir": str(run_dir),
            "stage0_encoder": str(encoder),
        })
        config["train"]["log_dir"] = str(stage_dir)
        config["checkpoint"]["dir"] = str(stage_dir / "checkpoints")
        config["checkpoint"]["resume"] = str(resume) if resume else None
        config["paths"]["encoder_override_ckpt"] = str(encoder)
        config["eval"]["checkpoint"] = str(stage_dir / "checkpoints" / "tfgridnet_best.ckpt")
        config["eval"]["out_json"] = str(stage_dir / "evaluation" / "results.json")
        config.setdefault("ddp", {})["num_gpus"] = args.num_gpus
        name = "config_stage1_runtime.yaml" if resume is None else f"config_stage1_resume_{resume.stem}.yaml"

    print(write_runtime_config(stage_dir / name, config, resume is not None))


if __name__ == "__main__":
    main()
