#!/usr/bin/env bash
#
# Run a fixed-count Soft_MOCOCO evaluation on a CPU-only Slurm node.
#
# Required environment variables:
#   EVAL_CONFIG     Runtime TFGridNet YAML configuration.
#   EVAL_CHECKPOINT Stage1 TFGridNet checkpoint.
#   EVAL_OUT        JSON result path.
#
# The supplied configs use eval.test_n=5000. Submit with --output outside the
# evaluation result file path, for example:
#   sbatch --export=ALL,EVAL_CONFIG=...,EVAL_CHECKPOINT=...,EVAL_OUT=... \
#     --output=/path/to/experiment/evaluation/slurm-cpu-%j.out \
#     script/slurm_eval_cpu.sh

#SBATCH -p cpu2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=soft_eval_cpu

set -euo pipefail

ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum"
PY="${PY:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
CONFIG="${EVAL_CONFIG:?EVAL_CONFIG is required}"
CHECKPOINT="${EVAL_CHECKPOINT:?EVAL_CHECKPOINT is required}"
OUT="${EVAL_OUT:?EVAL_OUT is required}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_mococo_eval_numba_cache_${USER:-user}}"

mkdir -p "$(dirname "$OUT")" "$NUMBA_CACHE_DIR"
echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "out=$OUT"
echo "device=cpu"
echo "python=$($PY -c 'import sys; print(sys.executable)')"

exec "$PY" Running_Lab/Soft_MOCOCO/script/run_eval.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT"
