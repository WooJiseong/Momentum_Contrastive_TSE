from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path("/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum")
DATE = "20260730"
DEPTHS = (3, 4, 6)
NUM_EPOCHS = 50

BASE_PROJECT = ROOT / "Running_Lab/TSE-through-Positive-Negative-Enroll"
MOCO_PROJECT = ROOT / "Running_Lab/PN_MOCOCO"
PNFLOW_ROOT = ROOT.parent


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def patch_tfgrid_config(cfg: dict, run_name: str, depth: int, is_moco: bool) -> dict:
    cfg = dict(cfg)
    cfg["experiment"] = dict(cfg.get("experiment", {}))
    cfg["experiment"].update(
        {
            "name": run_name,
            "date": DATE,
            "run_dir": f"exp/{run_name}",
            "runtime_config": f"exp/{run_name}/config_tfgridnet_supervised.yaml"
            if is_moco
            else f"exp/{run_name}/config_tfgridnet_baseline.yaml",
            "sweep_note": "fixed 50-epoch full-fusion depth sweep",
        }
    )

    cfg["train"] = dict(cfg["train"])
    cfg["train"]["num_epochs"] = NUM_EPOCHS
    cfg["train"]["log_dir"] = f"exp/{run_name}/stage1_tfgridnet"

    cfg["scheduler"] = dict(cfg.get("scheduler", {}))
    cfg["scheduler"]["type"] = "CosineAnnealingLR"
    cfg["scheduler"]["t_max"] = NUM_EPOCHS
    cfg["scheduler"]["warmup_epochs"] = min(5, NUM_EPOCHS)

    cfg["model"] = dict(cfg["model"])
    cfg["model"]["n_layers"] = depth
    cfg["model"]["fusion_layer"] = list(range(depth))

    cfg["paths"] = dict(cfg.get("paths", {}))
    cfg["paths"]["initial_model_ckpt"] = "../../../checkpoints/proposed-monaural.pt"
    cfg["paths"]["strict_initial_model"] = False
    if is_moco:
        cfg["paths"]["encoder_override_ckpt"] = "exp/moco_encoder/checkpoints/moco_best.ckpt"
        cfg["paths"]["strict_encoder_override"] = True

    cfg["trainable"] = dict(cfg.get("trainable", {}))
    cfg["trainable"]["encoder"] = False
    cfg["trainable"]["encoder_head"] = False
    cfg["trainable"]["separator"] = True

    cfg["early_stopping"] = dict(cfg.get("early_stopping", {}))
    cfg["early_stopping"]["enabled"] = False

    cfg["checkpoint"] = dict(cfg["checkpoint"])
    cfg["checkpoint"]["dir"] = f"exp/{run_name}/stage1_tfgridnet/checkpoints"
    cfg["checkpoint"]["ckpt_name"] = f"tfgridnet_depth{depth}_best"
    cfg["checkpoint"]["resume"] = None

    cfg["eval"] = dict(cfg["eval"])
    cfg["eval"]["checkpoint"] = (
        f"exp/{run_name}/stage1_tfgridnet/checkpoints/tfgridnet_depth{depth}_best.ckpt"
    )
    cfg["eval"]["out_json"] = f"exp/{run_name}/evaluation/results.json"
    cfg["eval"]["test_n"] = 5000
    cfg["eval"]["batch_size"] = 4
    cfg["eval"]["num_workers"] = 2
    return cfg


def patch_moco_config(cfg: dict, run_name: str) -> dict:
    cfg = dict(cfg)
    cfg["experiment"] = dict(cfg.get("experiment", {}))
    cfg["experiment"].update(
        {
            "name": run_name,
            "date": DATE,
            "run_dir": f"exp/{run_name}",
            "runtime_config": f"exp/{run_name}/config_moco_encoder.yaml",
            "sweep_note": "placeholder config; fixed pretrained MoCo encoder is reused",
        }
    )
    cfg["train"] = dict(cfg["train"])
    cfg["train"]["log_dir"] = f"exp/{run_name}/stage0_moco_encoder"
    cfg["checkpoint"] = dict(cfg["checkpoint"])
    cfg["checkpoint"]["dir"] = f"exp/{run_name}/stage0_moco_encoder/checkpoints"
    return cfg


def write_baseline_slurm(run_name: str, depth: int) -> Path:
    run_dir = BASE_PROJECT / "exp" / run_name
    script = run_dir / "slurm_train_eval.sh"
    text = f"""#!/usr/bin/env bash
#SBATCH -J dpth_b_n{depth}
#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output={run_dir}/slurm-%j.out

set -euo pipefail

PROJECT_DIR={BASE_PROJECT}
PNFLOW_ROOT={PNFLOW_ROOT}

cd "$PROJECT_DIR"

export RUN_DIR=exp/{run_name}
export CFG="$RUN_DIR/config_tfgridnet_baseline.yaml"
export PREPARED_RUN=1
export PY="${{PY:-python}}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT="${{MASTER_PORT:-29{depth}30}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="${{NUMBA_CACHE_DIR:-/tmp/tse_depth_sweep_numba_cache_${{USER:-user}}}}"
export PYTHONPATH="$PNFLOW_ROOT:$PROJECT_DIR:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

mkdir -p "$RUN_DIR" "$NUMBA_CACHE_DIR"

{{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "project=$PROJECT_DIR"
  echo "runtime_config=$CFG"
  echo "depth={depth}"
  printf 'fusion_layer=%s\\n' "$("$PY" -c 'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["model"]["fusion_layer"])' "$CFG")"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "slurm_job_id=${{SLURM_JOB_ID:-}}"
  "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))'
  nvidia-smi 2>/dev/null || true
}} > "$RUN_DIR/environment_torchrun.txt"

(
  for name in $(env | awk -F= '/^SLURM_/ {{print $1}}'); do
    unset "$name"
  done
  "$PY" -m torch.distributed.run \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node=4 \\
    train_tfgridnet_baseline.py --config "$CFG"
) > "$RUN_DIR/train_torchrun.log" 2>&1

CUDA_VISIBLE_DEVICES=0 \\
  PREPARED_RUN=1 RUN_DIR="$RUN_DIR" CFG="$CFG" \\
  bash script/train_eval.sh eval > "$RUN_DIR/eval5000.log" 2>&1
"""
    script.write_text(text)
    script.chmod(0o755)
    return script


def write_moco_slurm(run_name: str, depth: int) -> Path:
    run_dir = MOCO_PROJECT / "exp" / run_name
    script = run_dir / "slurm_train_eval.sh"
    text = f"""#!/usr/bin/env bash
#SBATCH -J dpth_m_n{depth}
#SBATCH -p gpu6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output={run_dir}/slurm-%j.out

set -euo pipefail

PROJECT_DIR={MOCO_PROJECT}
PNFLOW_ROOT={PNFLOW_ROOT}

cd "$PROJECT_DIR"

export RUN_DIR=exp/{run_name}
export MOCO_CFG="$RUN_DIR/config_moco_encoder.yaml"
export TFGRID_CFG="$RUN_DIR/config_tfgridnet_supervised.yaml"
export PREPARED_RUN=1
export PY="${{PY:-python}}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT="${{MASTER_PORT:-29{depth}40}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="${{NUMBA_CACHE_DIR:-/tmp/pn_depth_sweep_numba_cache_${{USER:-user}}}}"
export PYTHONPATH="$PNFLOW_ROOT:$PROJECT_DIR:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

mkdir -p "$RUN_DIR" "$NUMBA_CACHE_DIR"

{{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "project=$PROJECT_DIR"
  echo "runtime_moco_config=$MOCO_CFG"
  echo "runtime_tfgrid_config=$TFGRID_CFG"
  echo "depth={depth}"
  printf 'fusion_layer=%s\\n' "$("$PY" -c 'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["model"]["fusion_layer"])' "$TFGRID_CFG")"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "slurm_job_id=${{SLURM_JOB_ID:-}}"
  "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))'
  nvidia-smi 2>/dev/null || true
}} > "$RUN_DIR/environment_torchrun.txt"

(
  for name in $(env | awk -F= '/^SLURM_/ {{print $1}}'); do
    unset "$name"
  done
  "$PY" -m torch.distributed.run \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node=4 \\
    train_tfgridnet.py --config "$TFGRID_CFG"
) > "$RUN_DIR/train_torchrun.log" 2>&1

CUDA_VISIBLE_DEVICES=0 \\
  PREPARED_RUN=1 RUN_DIR="$RUN_DIR" MOCO_CFG="$MOCO_CFG" TFGRID_CFG="$TFGRID_CFG" \\
  bash script/train_eval.sh eval > "$RUN_DIR/eval5000.log" 2>&1
"""
    script.write_text(text)
    script.chmod(0o755)
    return script


def main() -> None:
    base_template = load_yaml(BASE_PROJECT / "configs/config_tfgridnet_baseline.yaml")
    moco_template = load_yaml(MOCO_PROJECT / "configs/config_moco_encoder.yaml")
    pn_template = load_yaml(MOCO_PROJECT / "configs/config_tfgridnet_supervised.yaml")

    scripts = []
    manifest = {"date": DATE, "num_epochs": NUM_EPOCHS, "runs": []}
    for depth in DEPTHS:
        base_name = f"{DATE}_depth_sweep_baseline_n{depth}_fullfusion_50ep"
        base_cfg = patch_tfgrid_config(base_template, base_name, depth, is_moco=False)
        base_cfg_path = BASE_PROJECT / "exp" / base_name / "config_tfgridnet_baseline.yaml"
        dump_yaml(base_cfg_path, base_cfg)
        base_script = write_baseline_slurm(base_name, depth)
        scripts.append(base_script)
        manifest["runs"].append(
            {
                "condition": "baseline",
                "depth": depth,
                "fusion_layer": list(range(depth)),
                "project": str(BASE_PROJECT),
                "run_dir": f"exp/{base_name}",
                "config": str(base_cfg_path),
                "slurm_script": str(base_script),
            }
        )

        moco_name = f"{DATE}_depth_sweep_mococo_mocobest_n{depth}_fullfusion_50ep"
        pn_cfg = patch_tfgrid_config(pn_template, moco_name, depth, is_moco=True)
        pn_cfg_path = MOCO_PROJECT / "exp" / moco_name / "config_tfgridnet_supervised.yaml"
        dump_yaml(pn_cfg_path, pn_cfg)
        moco_cfg = patch_moco_config(moco_template, moco_name)
        moco_cfg_path = MOCO_PROJECT / "exp" / moco_name / "config_moco_encoder.yaml"
        dump_yaml(moco_cfg_path, moco_cfg)
        moco_script = write_moco_slurm(moco_name, depth)
        scripts.append(moco_script)
        manifest["runs"].append(
            {
                "condition": "mococo",
                "depth": depth,
                "fusion_layer": list(range(depth)),
                "project": str(MOCO_PROJECT),
                "run_dir": f"exp/{moco_name}",
                "config": str(pn_cfg_path),
                "moco_config": str(moco_cfg_path),
                "slurm_script": str(moco_script),
            }
        )

    manifest_path = ROOT / f"depth_sweep_manifest_{DATE}.yaml"
    dump_yaml(manifest_path, manifest)
    print(manifest_path)
    for script in scripts:
        print(script)


if __name__ == "__main__":
    main()
