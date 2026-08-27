#!/usr/bin/env bash
# 5000-sample four-condition validation using the Soft Best-specific t-predictor.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu2
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=softbest_pred5k
#SBATCH --output=exp/20260822_soft_mococo_best_meanflow_gt_tau_stage1/eval-predicted-5k-slurm-%j.out

set -euo pipefail

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO"
EXP_DIR="$LAB_DIR/exp/20260822_soft_mococo_best_meanflow_gt_tau_stage1"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"
CONFIG="$LAB_DIR/configs/config_meanflow_soft_best_gt_tau.yaml"
CHECKPOINT="$EXP_DIR/checkpoints/meanflow_soft_mococo_best_gt_tau_best.ckpt"
TPRED_CHECKPOINT="$LAB_DIR/exp/20260825_soft_best_tpred/checkpoints/tpred_soft_best.ckpt"
OUT_DIR="$EXP_DIR/evaluation_predicted_t_5000"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_best_pred_eval_numba_cache}"
export PYTHONPATH="$LAB_DIR:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR" "$NUMBA_CACHE_DIR"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Flow checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[[ -f "$TPRED_CHECKPOINT" ]] || { echo "T-predictor checkpoint not found: $TPRED_CHECKPOINT" >&2; exit 1; }

echo "[eval] Soft Best Stage1 with Soft Best t-predictor"
echo "[eval] split=dev n=5000 t_mode=predicted nfe=1"
echo "[eval] output=$OUT_DIR"

exec "$PYTHON_BIN" "$LAB_DIR/eval_meanflow_four_conditions.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --tpred-ckpt "$TPRED_CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --n 5000 \
  --batch-size 8 \
  --nfe 1 \
  --t-mode predicted \
  --precision bf16 \
  --split dev \
  --mixture-seconds 6 \
  --enroll-seconds 3 \
  --snr-db-range -2.5 2.5 \
  --neg-partial-range 0.33 1.0
