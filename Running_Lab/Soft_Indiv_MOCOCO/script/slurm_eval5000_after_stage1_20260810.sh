#!/usr/bin/env bash
# Evaluate the no-1/n Stage1 best-checkpoint sweep after successful training.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum
#SBATCH -p cpu2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=soft_indiv_e5k
#SBATCH --output=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_Indiv_MOCOCO/exp/20260806_soft_indiv_mococo_no_1n/stage1_sweep/stage0_best/slurm-eval-%j.out

set -euo pipefail

ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum"
LAB_DIR="$ROOT/Running_Lab/Soft_Indiv_MOCOCO"
PY="${PY:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
RUN_DIR="$LAB_DIR/exp/20260806_soft_indiv_mococo_no_1n"
STAGE1_DIR="$RUN_DIR/stage1_sweep/stage0_best"
CONFIG="$STAGE1_DIR/config_stage1_runtime.yaml"
CHECKPOINT="$STAGE1_DIR/checkpoints/tfgridnet_best.ckpt"
OUT="$STAGE1_DIR/evaluation/results.json"
RUNTIME_CONFIG="$STAGE1_DIR/config_eval_runtime.yaml"
BASELINE="$ROOT/Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260729_tfgridnet_baseline_resume200_best_eval/evaluation/results.json"

cd "$ROOT"
export PYTHONPATH="$LAB_DIR:$ROOT/Running_Lab/PN_MOCOCO:$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_indiv_mococo_eval_numba_cache_${USER:-user}}"

mkdir -p "$(dirname "$OUT")" "$NUMBA_CACHE_DIR"
echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "out=$OUT"
echo "baseline=$BASELINE"

"$PY" Running_Lab/Soft_Indiv_MOCOCO/script/run_eval.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT"

"$PY" tools/fill_eval_baseline.py \
  --config "$RUNTIME_CONFIG" \
  --result-json "$OUT" \
  --baseline-json "$BASELINE" \
  --summary "$RUN_DIR/result_summary.md" \
  --baseline-name "tfgridnet_baseline_resume200_best_eval"
