#!/usr/bin/env bash
# Run one Attn teacher-weight sweep member's Stage1 SI-SDR experiment.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn_s1_sweep
#SBATCH --output=exp/20260819_attn_teacher_weight_sweep/%x-%j.out

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
SWEEP_TAG="${SWEEP_TAG:?Set SWEEP_TAG to teacher_0p03, teacher_0p1, or teacher_0p3}"
SWEEP_CONFIG="${SWEEP_CONFIG:?Set SWEEP_CONFIG to the sweep Stage1 YAML}"
SWEEP_DIR="exp/20260819_attn_teacher_weight_sweep/$SWEEP_TAG"

export MASTER_PORT="${MASTER_PORT:-29992}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_teacher_stage1_${SWEEP_TAG}_${USER:-user}}"
export PYTHONPATH="$PWD:$PWD/../PN_MOCOCO:$PWD/../../..:${PYTHONPATH:-}"

mkdir -p "$NUMBA_CACHE_DIR" "$SWEEP_DIR/stage1_tfgridnet_si_sdr"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$SWEEP_CONFIG" ]] || { echo "Config not found: $SWEEP_CONFIG" >&2; exit 1; }
[[ -f "$SWEEP_DIR/stage0_moco/checkpoints/pn_encoder_best.pt" ]] || {
  echo "Stage0 Best export not found for $SWEEP_TAG" >&2
  exit 1
}

echo "[Attn sweep] tag=$SWEEP_TAG"
echo "[Attn sweep] config=$SWEEP_CONFIG"
echo "[Attn sweep] stage0=$SWEEP_DIR/stage0_moco/checkpoints/pn_encoder_best.pt"

"$PYTHON_BIN" script/run_tfgridnet_attn.py \
  --config "$SWEEP_CONFIG" \
  2>&1 | tee "$SWEEP_DIR/stage1_tfgridnet_si_sdr/train.log"
