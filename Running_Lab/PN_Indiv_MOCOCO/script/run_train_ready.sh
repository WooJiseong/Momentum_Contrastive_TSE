#!/usr/bin/env bash
# Purpose: reliable launcher that creates experiment log directories before tee.
# Run from: Running_Lab/PN_Indiv_MOCOCO
# Single GPU: CUDA_VISIBLE_DEVICES=0 bash script/run_train_ready.sh moco
# Multi GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train_ready.sh tfgridnet

set -euo pipefail
MODE="${1:-}"
RUN_DIR="exp/20260801_indiv_mococo"
case "$MODE" in
  moco) mkdir -p "$RUN_DIR/stage0_moco"; exec bash script/run_train.sh moco > >(tee "$RUN_DIR/stage0_moco/train.log") 2>&1 ;;
  tfgridnet) mkdir -p "$RUN_DIR/stage1_tfgridnet"; exec bash script/run_train.sh tfgridnet > >(tee "$RUN_DIR/stage1_tfgridnet/train.log") 2>&1 ;;
  *) echo "Usage: bash script/run_train_ready.sh {moco|tfgridnet}"; exit 1 ;;
esac

