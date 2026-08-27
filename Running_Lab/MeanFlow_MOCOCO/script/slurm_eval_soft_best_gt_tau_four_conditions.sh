#!/usr/bin/env bash
# 5000-sample validation evaluation for Soft_MOCOCO GT-tau MeanFlow Stage1.
# Writes per-item rows plus mean, sample std, and two-sided 95% Student-t CI
# for the four PN-Enroll speaker-count conditions.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=soft_gt_eval5k
#SBATCH --output=exp/20260822_soft_mococo_best_meanflow_gt_tau_stage1/eval-validation-5k-slurm-%j.out

set -euo pipefail

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO"
EXP_DIR="$LAB_DIR/exp/20260822_soft_mococo_best_meanflow_gt_tau_stage1"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_mococo_gt_tau_eval_numba_cache}"
export PYTHONPATH="$LAB_DIR:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

CONFIG="$LAB_DIR/configs/config_meanflow_soft_best_gt_tau.yaml"
CHECKPOINT="$EXP_DIR/checkpoints/meanflow_soft_mococo_best_gt_tau_best.ckpt"
OUT_DIR="$EXP_DIR/evaluation_validation_four_conditions_5000"

mkdir -p "$OUT_DIR" "$NUMBA_CACHE_DIR"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }

echo "[eval] config=$CONFIG"
echo "[eval] checkpoint=$CHECKPOINT"
echo "[eval] split=dev n=5000 t_mode=oracle nfe=1"
echo "[eval] output=$OUT_DIR"

exec "$PYTHON_BIN" "$LAB_DIR/eval_meanflow_four_conditions.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --n 5000 \
  --batch-size 8 \
  --nfe 1 \
  --t-mode oracle \
  --precision bf16 \
  --split dev \
  --mixture-seconds 6 \
  --enroll-seconds 3 \
  --snr-db-range -2.5 2.5 \
  --neg-partial-range 0.33 1.0
