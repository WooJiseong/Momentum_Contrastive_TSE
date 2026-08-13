#!/usr/bin/env bash
# Purpose: train and evaluate Stage1 with the Stage0 epoch-50 encoder checkpoint.
# Run from: Running_Lab/Soft_MOCOCO
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/train_stage1_softleakage_20260803.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PNFLOW_ROOT="$(cd "$PROJECT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-all}"
case "$MODE" in
  all|train|eval|tfgridnet) ;;
  *)
    echo "Usage: bash script/train_stage1_softleakage_20260803.sh [all|train|eval|tfgridnet]"
    exit 1
    ;;
esac

PY="${PY:-python}"
RUN_DIR="${RUN_DIR:-exp/20260803_stage0_50ep_softleakage_test}"
TFGRID_CFG="${TFGRID_CFG:-configs/config_tfgridnet_soft_leakage_20260803_stage0_50ep_test.yaml}"
STAGE0_CKPT="$RUN_DIR/stage0_moco/checkpoints/pn_encoder_50ep.pt"
STAGE1_CKPT="$RUN_DIR/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/../PN_MOCOCO:$PNFLOW_ROOT:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29930}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_mococo_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$NUMBA_CACHE_DIR" "$RUN_DIR/stage0_moco/checkpoints" "$RUN_DIR/stage1_tfgridnet" "$RUN_DIR/evaluation"

record_runtime() {
  {
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "user=$(whoami)"
    echo "project=$PROJECT_DIR"
    echo "mode=$MODE"
    echo "stage0_checkpoint=$STAGE0_CKPT"
    echo "tfgrid_config=$TFGRID_CFG"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "master_port=$MASTER_PORT"
    echo "python=$($PY -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))' 2>/dev/null || true
    git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || echo "git_commit=unavailable"
    nvidia-smi 2>/dev/null || true
  } > "$RUN_DIR/environment.txt"
}

snapshot_config() {
  cp "$TFGRID_CFG" "$RUN_DIR/source_config_tfgridnet_soft_leakage.yaml"
}

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "$path" ]]; then
    echo "[Soft_MOCOCO] missing required file: $path"
    echo "[Soft_MOCOCO] $hint"
    exit 1
  fi
}

run_tfgridnet() {
  require_file "$STAGE0_CKPT" \
    "Copy the epoch-50 Stage0 export to this exact path before submitting Stage1."
  echo "[Soft_MOCOCO] ===== Stage 1: TFGridNet Soft-leakage training ====="
  "$PY" script/run_tfgridnet_soft.py --config "$TFGRID_CFG" 2>&1 | tee -a "$RUN_DIR/stage1_tfgridnet/train.log"
}

run_eval() {
  require_file "$STAGE1_CKPT" "Run Stage1 first."
  echo "[Soft_MOCOCO] ===== Stage 2: fixed-count evaluation ====="
  "$PY" script/run_eval.py --config "$TFGRID_CFG" 2>&1 | tee "$RUN_DIR/evaluation/eval.log"
}

record_runtime
snapshot_config

case "$MODE" in
  all)
    run_tfgridnet
    run_eval
    ;;
  train|tfgridnet)
    run_tfgridnet
    ;;
  eval)
    run_eval
    ;;
esac

echo "[Soft_MOCOCO] checkpoint=$STAGE1_CKPT"
echo "[Soft_MOCOCO] eval_json=$RUN_DIR/evaluation/results.json"
