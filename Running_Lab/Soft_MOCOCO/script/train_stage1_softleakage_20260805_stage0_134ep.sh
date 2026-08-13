#!/usr/bin/env bash
#
# Purpose:
#   Train and evaluate Soft_MOCOCO Stage1 from the preserved Stage0 epoch-134 encoder.
#
# Run from:
#   contrastive_momentum/Running_Lab/Soft_MOCOCO
#
# Single GPU:
#   Not supported by this fixed 4-GPU sweep script.
#
# Multi GPU:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29934 \
#     bash script/train_stage1_softleakage_20260805_stage0_134ep.sh all
#
# Required arg:
#   all | train | eval
#
# Output:
#   exp/20260805_stage0_134ep_softleakage_test/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm:
#   sbatch script/slurm_stage1_softleakage_20260805_stage0_134ep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTRASTIVE_ROOT="$(cd "$LAB_DIR/../.." && pwd)"
PNFLOW_ROOT="$(cd "$LAB_DIR/../../.." && pwd)"
cd "$LAB_DIR"

MODE="${1:-all}"
case "$MODE" in
  all|train|eval) ;;
  *)
    echo "Usage: bash script/train_stage1_softleakage_20260805_stage0_134ep.sh [all|train|eval]" >&2
    exit 1
    ;;
esac

PY="${PY:-python}"
RUN_DIR="exp/20260805_stage0_134ep_softleakage_test"
TFGRID_CFG="configs/config_tfgridnet_soft_leakage_20260805_stage0_134ep_test.yaml"
STAGE0_CKPT="$RUN_DIR/stage0_moco/checkpoints/pn_encoder_134ep.pt"
STAGE1_CKPT="$RUN_DIR/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt"
REQUEUE_COUNT="${SLURM_RESTART_COUNT:-0}"
ALLOW_EXISTING_RUN="${SOFT_MOCOCO_ALLOW_EXISTING_RUN:-0}"

export PYTHONPATH="$LAB_DIR:$LAB_DIR/../PN_MOCOCO:$CONTRASTIVE_ROOT:$PNFLOW_ROOT:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29934}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_mococo_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "$path" ]]; then
    echo "[Soft_MOCOCO] missing required file: $path" >&2
    echo "[Soft_MOCOCO] $hint" >&2
    exit 1
  fi
}

if [[ "$MODE" != "eval" && -e "$RUN_DIR/stage1_tfgridnet/train.log" && "$REQUEUE_COUNT" == "0" && "$ALLOW_EXISTING_RUN" != "1" ]]; then
  echo "[Soft_MOCOCO] protected existing Stage1 run: $RUN_DIR" >&2
  echo "[Soft_MOCOCO] use its saved last.ckpt with a separate explicit resume experiment." >&2
  exit 1
fi

mkdir -p "$NUMBA_CACHE_DIR" "$RUN_DIR/stage1_tfgridnet" "$RUN_DIR/evaluation"
require_file "$TFGRID_CFG" "The Stage1 sweep config is required."
require_file "$STAGE0_CKPT" "The frozen epoch-134 Stage0 encoder copy is required."

record_runtime() {
  {
    echo "date=$(date -Is)"
    echo "mode=$MODE"
    echo "stage0_checkpoint=$STAGE0_CKPT"
    echo "stage0_epoch=134"
    echo "tfgrid_config=$TFGRID_CFG"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "master_port=$MASTER_PORT"
    echo "seed=42"
    echo "python=$($PY -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))' 2>/dev/null || true
    git -C "$LAB_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || echo "git_commit=unavailable"
    nvidia-smi 2>/dev/null || true
  } > "$RUN_DIR/environment.txt"
}

run_train() {
  echo "[Soft_MOCOCO] ===== Stage1: epoch-134 Soft-leakage TFGridNet ====="
  "$PY" script/run_tfgridnet_soft.py --config "$TFGRID_CFG" \
    2>&1 | tee -a "$RUN_DIR/stage1_tfgridnet/train.log"
}

run_eval() {
  require_file "$STAGE1_CKPT" "Run the Stage1 training before evaluation."
  echo "[Soft_MOCOCO] ===== Stage2: fixed-count evaluation ====="
  "$PY" script/run_eval.py --config "$TFGRID_CFG" \
    2>&1 | tee "$RUN_DIR/evaluation/eval.log"
}

record_runtime
cp "$TFGRID_CFG" "$RUN_DIR/source_config_tfgridnet_soft_leakage.yaml"
sha256sum "$STAGE0_CKPT" > "$RUN_DIR/stage0_encoder_sha256.txt"

case "$MODE" in
  all)
    run_train
    run_eval
    ;;
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
esac
