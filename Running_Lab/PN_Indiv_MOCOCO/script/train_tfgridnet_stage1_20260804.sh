#!/usr/bin/env bash
#
# Purpose:
#   Train PN_Indiv_MOCOCO TFGridNet Stage1 with actual individual mixture nuisances.
#
# Run from:
#   contrastive_momentum/Running_Lab/PN_Indiv_MOCOCO
#
# Single GPU:
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash script/train_tfgridnet_stage1_20260804.sh
#
# Multi GPU:
#   NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29850 \
#     bash script/train_tfgridnet_stage1_20260804.sh
#
# Optional environment:
#   STAGE0_CKPT=exp/20260802_indiv_mococo/stage0_moco/checkpoints/pn_encoder_best.pt
#   RUN_DIR=exp/YYYYMMDD_title
#   RESUME_STAGE1=/absolute/path/to/last.ckpt
#   REUSE_RUN=1 (required together with RESUME_STAGE1 for an existing run)
#
# Output:
#   exp/20260804_indiv_mococo_stage1_individual_nuisance/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm:
#   sbatch script/slurm_tfgridnet_stage1_20260804.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTRASTIVE_ROOT="$(cd "$LAB_DIR/../.." && pwd)"
PNFLOW_ROOT="$(cd "$LAB_DIR/../../.." && pwd)"
cd "$LAB_DIR"

PY="${PY:-python}"
RUN_DIR="${RUN_DIR:-exp/20260804_indiv_mococo_stage1_individual_nuisance}"
STAGE0_CKPT="${STAGE0_CKPT:-exp/20260802_indiv_mococo/stage0_moco/checkpoints/pn_encoder_best.pt}"
TFGRID_CFG="${TFGRID_CFG:-configs/config_tfgridnet_indiv_stage1_20260804.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"

export PYTHONPATH="$LAB_DIR:$LAB_DIR/../PN_MOCOCO:$CONTRASTIVE_ROOT:$PNFLOW_ROOT:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29850}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/pn_indiv_mococo_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "$STAGE0_CKPT" ]]; then
  echo "[PN_Indiv_MOCOCO] missing Stage0 encoder export: $STAGE0_CKPT" >&2
  exit 1
fi
if [[ ! -f "$TFGRID_CFG" ]]; then
  echo "[PN_Indiv_MOCOCO] missing Stage1 config: $TFGRID_CFG" >&2
  exit 1
fi
if [[ -e "$RUN_DIR/run_manifest.txt" && "${REUSE_RUN:-0}" != "1" ]]; then
  echo "[PN_Indiv_MOCOCO] existing run is protected: $RUN_DIR" >&2
  echo "[PN_Indiv_MOCOCO] use REUSE_RUN=1 with RESUME_STAGE1=/path/to/last.ckpt to continue it." >&2
  exit 1
fi
if [[ "${REUSE_RUN:-0}" == "1" && -z "${RESUME_STAGE1:-}" ]]; then
  echo "[PN_Indiv_MOCOCO] REUSE_RUN=1 requires RESUME_STAGE1 to avoid overwriting a run." >&2
  exit 1
fi
if [[ -n "${RESUME_STAGE1:-}" && ! -f "$RESUME_STAGE1" ]]; then
  echo "[PN_Indiv_MOCOCO] missing resume checkpoint: $RESUME_STAGE1" >&2
  exit 1
fi

mkdir -p "$NUMBA_CACHE_DIR" "$RUN_DIR/stage1_tfgridnet"
if [[ ! -e "$RUN_DIR/run_manifest.txt" ]]; then
  {
    echo "experiment=pn_indiv_mococo_stage1_individual_nuisance"
    echo "created_at=$(date -Is)"
    echo "seed=42"
    echo "stage0_encoder=$STAGE0_CKPT"
    echo "stage1_loss=negative_si_sdr + 0.1*target_orthogonal_leakage"
    echo "leakage_interferers=actual_sample[1:]"
  } > "$RUN_DIR/run_manifest.txt"
fi

environment_suffix="initial"
if [[ "${REUSE_RUN:-0}" == "1" ]]; then
  environment_suffix="resume_$(date +%Y%m%dT%H%M%S)"
fi
{
  echo "date=$(date -Is)"
  echo "run_dir=$RUN_DIR"
  echo "stage0_ckpt=$STAGE0_CKPT"
  echo "resume_stage1=${RESUME_STAGE1:-}"
  echo "num_gpus=$NUM_GPUS"
  echo "seed=42"
  echo "python=$($PY -c 'import sys; print(sys.executable)')"
  "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
  "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("gpu_count=" + str(torch.cuda.device_count()))'
  git -C "$LAB_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || echo "git_commit=unavailable"
  nvidia-smi 2>/dev/null || true
} > "$RUN_DIR/environment_${environment_suffix}.txt"
cp "$TFGRID_CFG" "$RUN_DIR/source_config_tfgridnet_indiv_stage1.yaml"
sha256sum "$STAGE0_CKPT" > "$RUN_DIR/stage0_encoder_sha256.txt"

command=(
  "$PY" script/run_tfgridnet_stage1.py
  --config "$TFGRID_CFG"
  --run-dir "$RUN_DIR"
  --stage0-encoder "$STAGE0_CKPT"
  --num-gpus "$NUM_GPUS"
)
if [[ -n "${RESUME_STAGE1:-}" ]]; then
  command+=(--resume "$RESUME_STAGE1")
fi

echo "[PN_Indiv_MOCOCO] ===== Stage1: TFGridNet with individual-source leakage ====="
"${command[@]}" 2>&1 | tee -a "$RUN_DIR/stage1_tfgridnet/train.log"
