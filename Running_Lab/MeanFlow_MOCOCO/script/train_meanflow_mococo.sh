#!/usr/bin/env bash
# Launch MeanFlow Stage1 with a MOCOCO Stage0 enrollment encoder.
# Run from Running_Lab/MeanFlow_MOCOCO.

set -euo pipefail

ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum"
LAB_DIR="$ROOT/Running_Lab/MeanFlow_MOCOCO"
PNFLOW_ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE"
SIA_REPO="$PNFLOW_ROOT/sia_fm_tse"
PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
CONFIG="${MF_CONFIG:-$LAB_DIR/configs/config_meanflow_mococo_soft50.yaml}"

cd "$LAB_DIR"
export PYTHONPATH="$LAB_DIR:$SIA_REPO:$PNFLOW_ROOT:$ROOT:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/meanflow_mococo_numba_cache_${USER:-user}}"
mkdir -p "$NUMBA_CACHE_DIR"

args=("$LAB_DIR/train_meanflow_mococo.py" --config "$CONFIG")
if [[ -n "${STAGE0_CKPT:-}" ]]; then
  args+=(--stage0-ckpt "$STAGE0_CKPT")
fi
if [[ -n "${RESUME_CKPT:-}" ]]; then
  args+=(--resume "$RESUME_CKPT")
fi

echo "config=$CONFIG"
echo "stage0_ckpt=${STAGE0_CKPT:-from-config}"
echo "resume_ckpt=${RESUME_CKPT:-none}"
echo "meanflow_num_workers=${MEANFLOW_NUM_WORKERS:-from-config}"
echo "sia_repo=$SIA_REPO"
exec "$PYTHON_BIN" "${args[@]}"
