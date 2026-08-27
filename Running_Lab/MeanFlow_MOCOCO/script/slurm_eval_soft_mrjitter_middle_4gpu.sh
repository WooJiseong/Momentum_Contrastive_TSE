#!/usr/bin/env bash
# Four-GPU, four-condition evaluation for a fixed Soft MR-jitter Stage1 epoch.
# One worker process is assigned to each GPU and one speaker condition.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu2
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=soft_mrj_eval4g
#SBATCH --output=exp/20260827_soft_w_mrjitter_middle_74_360only/eval-%x-slurm-%j.out

set -euo pipefail

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO"
EXP_DIR="$LAB_DIR/exp/20260827_soft_w_mrjitter_middle_74_360only"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"
EVAL_SCRIPT="$LAB_DIR/eval_meanflow_four_conditions.py"
MERGE_SCRIPT="$LAB_DIR/script/merge_meanflow_eval_parts.py"
CONFIG="$EXP_DIR/config_eval.yaml"
CHECKPOINT="$EXP_DIR/checkpoints/meanflow_soft_best_mrjitter_360only_epoch74_best.ckpt"
MODE="${EVAL_MODE:-oracle}"

case "$MODE" in
  oracle)
    OUT_DIR="$EXP_DIR/oracle_5k"
    TPRED_ARGS=()
    ;;
  predicted)
    OUT_DIR="$EXP_DIR/tpred_5k"
    TPRED_CHECKPOINT="$LAB_DIR/exp/20260826_soft_best_tpred_multicondition/checkpoints/tpred_soft_best_multicondition.ckpt"
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

mkdir -p "$OUT_DIR" "$OUT_DIR/part_0" "$OUT_DIR/part_1" "$OUT_DIR/part_2" "$OUT_DIR/part_3"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Flow checkpoint not found: $CHECKPOINT" >&2; exit 1; }

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

labels=(2mix_2enroll 2mix_3enroll 3mix_2enroll 3mix_3enroll)
pids=()
for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$index" NUMBA_CACHE_DIR="/tmp/soft_mrjitter_middle_eval_${SLURM_JOB_ID:-local}_gpu${index}" \
    "$PYTHON_BIN" "$EVAL_SCRIPT" "${COMMON_ARGS[@]}" "${TPRED_ARGS[@]}" \
    --conditions "${labels[$index]}" \
    --out-dir "$OUT_DIR/part_$index" >"$OUT_DIR/part_$index/worker.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
(( status == 0 )) || { echo "At least one evaluation worker failed" >&2; exit "$status"; }

exec "$PYTHON_BIN" "$MERGE_SCRIPT" \
  --part-dir "$OUT_DIR/part_0" \
  --part-dir "$OUT_DIR/part_1" \
  --part-dir "$OUT_DIR/part_2" \
  --part-dir "$OUT_DIR/part_3" \
  --out-dir "$OUT_DIR"
