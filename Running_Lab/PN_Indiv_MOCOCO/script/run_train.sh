#!/usr/bin/env bash
# Purpose: train the individual-negative MoCo stage or its shared Stage1 separator.
# Run from: Running_Lab/PN_Indiv_MOCOCO
# Single GPU: CUDA_VISIBLE_DEVICES=0 bash script/run_train.sh moco
# Multi GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train.sh moco

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
MODE="${1:-}"
case "$MODE" in moco|tfgridnet) ;; *) echo "Usage: bash script/run_train.sh {moco|tfgridnet}"; exit 1 ;; esac

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/../PN_MOCOCO:$PROJECT_DIR/../../..:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29820}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/pn_indiv_mococo_numba_cache_${USER:-user}}"
mkdir -p "$NUMBA_CACHE_DIR"

if [[ "$MODE" == "moco" ]]; then
  python train_moco_encoder.py --config configs/config_indiv_moco.yaml 2>&1 | tee exp/20260801_indiv_mococo/stage0_moco/train.log
else
  python script/run_tfgridnet.py --config configs/config_tfgridnet_indiv.yaml 2>&1 | tee exp/20260801_indiv_mococo/stage1_tfgridnet/train.log
fi

