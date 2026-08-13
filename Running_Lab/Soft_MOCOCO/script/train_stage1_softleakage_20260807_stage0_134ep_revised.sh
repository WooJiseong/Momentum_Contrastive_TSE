#!/usr/bin/env bash
#
# Purpose:
#   Train and evaluate a clean Stage1 run from the preserved Stage0 epoch-134
#   encoder using the revised Leakage Loss.
#
# Run from Running_Lab/Soft_MOCOCO:
#   bash script/train_stage1_softleakage_20260807_stage0_134ep_revised.sh all
#
# Required arg:
#   all | train | eval

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
    echo "Usage: bash script/train_stage1_softleakage_20260807_stage0_134ep_revised.sh [all|train|eval]" >&2
    exit 1
    ;;
esac

PY="${PY:-python}"
RUN_DIR="exp/20260807_stage0_134ep_softleakage_revised_loss_test"
TFGRID_CFG="configs/config_tfgridnet_soft_leakage_20260807_stage0_134ep_revised_test.yaml"
STAGE0_CKPT="$RUN_DIR/stage0_moco/checkpoints/pn_encoder_134ep.pt"
STAGE1_CKPT="$RUN_DIR/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt"

export PYTHONPATH="$LAB_DIR:$LAB_DIR/../PN_MOCOCO:$CONTRASTIVE_ROOT:$PNFLOW_ROOT:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29974}"
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

if [[ "$MODE" != "eval" && -e "$RUN_DIR/stage1_tfgridnet/train.log" ]]; then
  echo "[Soft_MOCOCO] protected existing Stage1 run: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$NUMBA_CACHE_DIR" "$RUN_DIR/stage1_tfgridnet" "$RUN_DIR/evaluation"
require_file "$TFGRID_CFG" "The revised-loss Stage1 config is required."
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
    echo "stage1_resume=null"
    echo "loss_revision=normalized_residual_energy_activity_gate"
    echo "python=$($PY -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))' 2>/dev/null || true
    git -C "$CONTRASTIVE_ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || echo "git_commit=unavailable"
    nvidia-smi 2>/dev/null || true
  } > "$RUN_DIR/environment.txt"
}

run_train() {
  echo "[Soft_MOCOCO] ===== Stage1: epoch-134 revised Soft-leakage TFGridNet ====="
  "$PY" script/run_tfgridnet_soft.py --config "$TFGRID_CFG" \
    2>&1 | tee "$RUN_DIR/stage1_tfgridnet/train.log"
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
