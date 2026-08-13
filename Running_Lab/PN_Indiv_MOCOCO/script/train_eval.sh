#!/usr/bin/env bash
# Purpose: run individual-negative MoCo Stage0 and/or shared TFGridNet Stage1.
# Run from: Running_Lab/PN_Indiv_MOCOCO
# Single GPU: CUDA_VISIBLE_DEVICES=0 bash script/train_eval.sh all
# Multi GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/train_eval.sh all

set -euo pipefail
MODE="${1:-all}"
case "$MODE" in
  moco) bash script/run_train.sh moco ;;
  tfgridnet) bash script/run_train.sh tfgridnet ;;
  all) bash script/run_train.sh moco && bash script/run_train.sh tfgridnet ;;
  *) echo "Usage: bash script/train_eval.sh {moco|tfgridnet|all}"; exit 1 ;;
esac

