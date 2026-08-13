#!/usr/bin/env bash
# Purpose: run Soft_MOCOCO with Stage1 target-orthogonal leakage regularization.
# Run from: Running_Lab/Soft_MOCOCO
# Single GPU: CUDA_VISIBLE_DEVICES=0 bash script/run_train_soft_leakage.sh all
# Multi GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train_soft_leakage.sh all

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
MODE="${1:-all}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/../PN_MOCOCO:$PROJECT_DIR/../../..:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29920}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_mococo_numba_cache_${USER:-user}}"
mkdir -p "$NUMBA_CACHE_DIR" exp/20260801_soft_mococo/stage0_moco exp/20260801_soft_mococo/stage1_tfgridnet
if [[ "$MODE" == "moco" || "$MODE" == "all" ]]; then
  python train_soft_moco_encoder.py --config configs/config_soft_moco.yaml 2>&1 | tee exp/20260801_soft_mococo/stage0_moco/train.log
fi
if [[ "$MODE" == "tfgridnet" || "$MODE" == "all" ]]; then
  python script/run_tfgridnet_soft.py --config configs/config_tfgridnet_soft_leakage.yaml 2>&1 | tee exp/20260801_soft_mococo/stage1_tfgridnet/train.log
fi

