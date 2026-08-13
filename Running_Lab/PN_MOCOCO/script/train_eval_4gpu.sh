#!/usr/bin/env bash

# Purpose:
#   Compatibility alias for the PN_MOCOCO train/eval workflow.
#
# Run from:
#   Running_Lab/PN_MOCOCO
#
# Single GPU:
#   Use script/train_eval.sh with matching YAML ddp.num_gpus.
#
# Multi GPU:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/train_eval_4gpu.sh all
#
# Arguments:
#   Forwarded to script/train_eval.sh.
#
# Output:
#   exp/YYYYMMDD_pn_mococo[_runNN]/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm interactive example:
#   srun -p gpu6 --gres=gpu:4 --cpus-per-task=16 --mem=48G --time=72:00:00 --pty bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[PN_MOCOCO] script/train_eval_4gpu.sh is kept as a compatibility alias."
echo "[PN_MOCOCO] GPU count is now controlled by YAML ddp.num_gpus; default configs use 4 GPUs."
exec bash "$SCRIPT_DIR/train_eval.sh" "$@"
