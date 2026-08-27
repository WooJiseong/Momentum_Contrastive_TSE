#!/usr/bin/env bash
# One-GPU, four-condition paper-style evaluation for the completed MeanFlow run.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=mf_eval4c
#SBATCH --output=exp/20260820_soft50_meanflow_gamma1_mrjitter_nfe1/eval-baseline4c-slurm-%j.out

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/meanflow_mococo_eval_numba_cache}"
export PYTHONPATH="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO"
EXP_DIR="$LAB_DIR/exp/20260820_soft50_meanflow_gamma1_mrjitter_nfe1"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"

mkdir -p "$EXP_DIR/evaluation_four_conditions"
exec "$PYTHON_BIN" "$LAB_DIR/eval_meanflow_four_conditions.py" \
  --config "$LAB_DIR/configs/config_meanflow_mococo_gamma1_mrjitter_nfe1.yaml" \
  --checkpoint "$EXP_DIR/checkpoints/meanflow_soft50_gamma1_mrjitter_nfe1_best.ckpt" \
  --out-dir "$EXP_DIR/evaluation_four_conditions" \
  --n 5000 \
  --batch-size 8 \
  --nfe 1 \
  --t-mode predicted \
  --precision bf16 \
  --split test \
  --mixture-seconds 6 \
  --enroll-seconds 3 \
  --snr-db-range -2.5 2.5 \
  --neg-partial-range 0.33 1.0
