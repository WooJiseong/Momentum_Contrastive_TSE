#!/usr/bin/env bash
#
# Purpose:
#   Run the 20260802 PN_Indiv_MOCOCO Stage0, Stage1, and fixed-count Eval workflow.
#
# Run from:
#   Running_Lab/PN_Indiv_MOCOCO
#
# Single GPU:
#   Not supported by this fixed 4-GPU experiment script.
#
# Multi GPU:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29820 bash script/train_eval_20260802.sh all
#
# Required arg:
#   all | train | eval | moco | tfgridnet
#
# Output:
#   exp/20260802_indiv_mococo/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm:
#   sbatch script/slurm_train_eval_20260802.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTRASTIVE_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
PNFLOW_ROOT="$(cd "$PROJECT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-all}"
case "$MODE" in
  all|train|eval|moco|tfgridnet) ;;
  *)
    echo "Usage: bash script/train_eval_20260802.sh [all|train|eval|moco|tfgridnet]"
    exit 1
    ;;
esac

PY="${PY:-python}"
RUN_DIR="${RUN_DIR:-exp/20260802_indiv_mococo}"
MOCO_CFG="${MOCO_CFG:-configs/config_indiv_moco.yaml}"
TFGRID_CFG="${TFGRID_CFG:-configs/config_tfgridnet_indiv.yaml}"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/../PN_MOCOCO:$CONTRASTIVE_ROOT:$PNFLOW_ROOT:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29820}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/pn_indiv_mococo_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "$NUMBA_CACHE_DIR" \
  "$RUN_DIR/stage0_moco" \
  "$RUN_DIR/stage1_tfgridnet" \
  "$RUN_DIR/evaluation"

record_runtime() {
  {
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "user=$(whoami)"
    echo "project=$PROJECT_DIR"
    echo "contrastive_root=$CONTRASTIVE_ROOT"
    echo "pnflow_root=$PNFLOW_ROOT"
    echo "mode=$MODE"
    echo "moco_config=$MOCO_CFG"
    echo "tfgrid_config=$TFGRID_CFG"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "master_port=$MASTER_PORT"
    echo "python=$("$PY" -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))' 2>/dev/null || true
    git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || echo "git_commit=unavailable"
    nvidia-smi 2>/dev/null || true
  } > "$RUN_DIR/environment.txt"
}

snapshot_configs() {
  cp "$MOCO_CFG" "$RUN_DIR/source_config_indiv_moco.yaml"
  cp "$TFGRID_CFG" "$RUN_DIR/source_config_tfgridnet_indiv.yaml"
}

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "$path" ]]; then
    echo "[PN_Indiv_MOCOCO] missing required file: $path"
    echo "[PN_Indiv_MOCOCO] $hint"
    exit 1
  fi
}

run_moco() {
  echo "[PN_Indiv_MOCOCO] ===== Stage 0: individual-negative MoCo encoder training ====="
  "$PY" train_moco_encoder.py --config "$MOCO_CFG" 2>&1 | tee "$RUN_DIR/stage0_moco/train.log"
}

run_tfgridnet() {
  require_file "$RUN_DIR/stage0_moco/checkpoints/pn_encoder_best.pt" \
    "Run Stage0 first, or update paths.encoder_override_ckpt in $TFGRID_CFG."
  echo "[PN_Indiv_MOCOCO] ===== Stage 1: TFGridNet supervised training ====="
  "$PY" script/run_tfgridnet.py --config "$TFGRID_CFG" 2>&1 | tee "$RUN_DIR/stage1_tfgridnet/train.log"
}

run_eval() {
  require_file "$RUN_DIR/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt" \
    "Run Stage1 first, or update eval.checkpoint in $TFGRID_CFG."
  echo "[PN_Indiv_MOCOCO] ===== Stage 2: fixed-count evaluation ====="
  "$PY" script/run_eval.py \
    --config "$TFGRID_CFG" \
    --checkpoint "$RUN_DIR/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt" \
    --out "$RUN_DIR/evaluation/results.json" \
    2>&1 | tee "$RUN_DIR/evaluation/eval.log"
}

record_runtime
snapshot_configs

case "$MODE" in
  all)
    run_moco
    run_tfgridnet
    run_eval
    ;;
  train)
    run_moco
    run_tfgridnet
    ;;
  moco)
    run_moco
    ;;
  tfgridnet)
    run_tfgridnet
    ;;
  eval)
    run_eval
    ;;
esac

echo "[PN_Indiv_MOCOCO] done"
echo "[PN_Indiv_MOCOCO] checkpoint=$RUN_DIR/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt"
echo "[PN_Indiv_MOCOCO] eval_json=$RUN_DIR/evaluation/results.json"
