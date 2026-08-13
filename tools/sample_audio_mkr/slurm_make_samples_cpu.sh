#!/usr/bin/env bash
#
# Run the sample_audio_mkr on a CPU-only Slurm node.
#
# Required environment variables:
#   SAMPLE_CONFIG  Runtime or source TFGridNet YAML configuration.
#   SAMPLE_OUT     Experiment-local output directory.
#
# Optional environment variables:
#   SAMPLE_BACKEND     auto, pn_mococo, or baseline. Default: auto.
#   SAMPLE_NUM_SAMPLES Number of samples. Default: 3.
#   PY                Python executable. Default: pnflowtse environment Python.
#
# Example submission:
#   mkdir -p /path/to/experiment/example_cpu_3sample
#   sbatch --export=ALL,SAMPLE_CONFIG=...,SAMPLE_OUT=...,SAMPLE_BACKEND=...,SAMPLE_NUM_SAMPLES=3 \
#     --output=/path/to/experiment/slurm-sample-cpu-%j.out \
#     tools/sample_audio_mkr/slurm_make_samples_cpu.sh

#SBATCH -p cpu2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=tse_samples_cpu

set -euo pipefail

ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum"
PY="${PY:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
CONFIG="${SAMPLE_CONFIG:?SAMPLE_CONFIG is required}"
OUT="${SAMPLE_OUT:?SAMPLE_OUT is required}"
BACKEND="${SAMPLE_BACKEND:-auto}"
NUM_SAMPLES="${SAMPLE_NUM_SAMPLES:-3}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/sample_audio_mkr_numba_cache_${USER:-user}}"

mkdir -p "$OUT" "$NUMBA_CACHE_DIR"
echo "config=$CONFIG"
echo "output=$OUT"
echo "backend=$BACKEND"
echo "num_samples=$NUM_SAMPLES"
echo "device=cpu"
echo "python=$($PY -c 'import sys; print(sys.executable)')"

exec "$PY" tools/sample_audio_mkr/make_samples.py \
  --config "$CONFIG" \
  --out "$OUT" \
  --backend "$BACKEND" \
  --num-samples "$NUM_SAMPLES" \
  --device cpu
