#!/usr/bin/env bash
# Parallel 5K evaluation of the no-jitter Soft Best Stage1.
# Two worker processes use one allocated GPU each and evaluate two conditions.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu2
#SBATCH --gres=gpu:a10:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=120:00:00
#SBATCH --job-name=soft_nj_eval2g
#SBATCH --output=exp/20260822_soft_mococo_best_meanflow_gt_tau_stage1/eval-nojitter-2gpu-slurm-%j.out

set -euo pipefail

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO"
EXP_DIR="$LAB_DIR/exp/20260822_soft_mococo_best_meanflow_gt_tau_stage1"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"
EVAL_SCRIPT="$LAB_DIR/eval_meanflow_four_conditions.py"
MERGE_SCRIPT="$LAB_DIR/script/merge_meanflow_eval_parts.py"
CONFIG="$LAB_DIR/configs/config_meanflow_soft_best_gt_tau.yaml"
CHECKPOINT="$EXP_DIR/checkpoints/meanflow_soft_mococo_best_gt_tau_best.ckpt"
MODE="${EVAL_MODE:-oracle}"

case "$MODE" in
  oracle)
    OUT_DIR="$EXP_DIR/evaluation_nojitter_oracle_5000_2gpu"
    TPRED_ARGS=()
    ;;
  predicted)
    OUT_DIR="$EXP_DIR/evaluation_nojitter_soft_multicondition_tpred_5000_2gpu"
    TPRED_CHECKPOINT="$EXP_DIR/../20260826_soft_best_tpred_multicondition/checkpoints/tpred_soft_best_multicondition.ckpt"
    [[ -f "$TPRED_CHECKPOINT" ]] || { echo "T-predictor checkpoint not found: $TPRED_CHECKPOINT" >&2; exit 1; }
    TPRED_ARGS=(--tpred-ckpt "$TPRED_CHECKPOINT")
    ;;
  *)
    echo "EVAL_MODE must be oracle or predicted, got: $MODE" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONPATH="$LAB_DIR:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR" "$OUT_DIR/part_0" "$OUT_DIR/part_1"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Flow checkpoint not found: $CHECKPOINT" >&2; exit 1; }

echo "[eval] mode=$MODE"
echo "[eval] config=$CONFIG"
echo "[eval] checkpoint=$CHECKPOINT"
echo "[eval] split=dev n=5000 t_mode=$MODE nfe=1"
echo "[eval] output=$OUT_DIR"

COMMON_ARGS=(
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --n 5000
  --batch-size 8
  --nfe 1
  --t-mode "$MODE"
  --precision bf16
  --split dev
  --mixture-seconds 6
  --enroll-seconds 3
  --snr-db-range -2.5 2.5
  --neg-partial-range 0.33 1.0
)

CUDA_VISIBLE_DEVICES=0 NUMBA_CACHE_DIR="/tmp/soft_best_nojitter_eval_${SLURM_JOB_ID:-local}_gpu0" \
  "$PYTHON_BIN" "$EVAL_SCRIPT" "${COMMON_ARGS[@]}" "${TPRED_ARGS[@]}" \
  --conditions 2mix_2enroll 2mix_3enroll \
  --out-dir "$OUT_DIR/part_0" >"$OUT_DIR/part_0/worker.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 NUMBA_CACHE_DIR="/tmp/soft_best_nojitter_eval_${SLURM_JOB_ID:-local}_gpu1" \
  "$PYTHON_BIN" "$EVAL_SCRIPT" "${COMMON_ARGS[@]}" "${TPRED_ARGS[@]}" \
  --conditions 3mix_2enroll 3mix_3enroll \
  --out-dir "$OUT_DIR/part_1" >"$OUT_DIR/part_1/worker.log" 2>&1 &
PID1=$!

status=0
wait "$PID0" || status=$?
wait "$PID1" || status=$?
(( status == 0 )) || { echo "At least one evaluation worker failed" >&2; exit "$status"; }

exec "$PYTHON_BIN" "$MERGE_SCRIPT" \
  --part-dir "$OUT_DIR/part_0" \
  --part-dir "$OUT_DIR/part_1" \
  --out-dir "$OUT_DIR"
