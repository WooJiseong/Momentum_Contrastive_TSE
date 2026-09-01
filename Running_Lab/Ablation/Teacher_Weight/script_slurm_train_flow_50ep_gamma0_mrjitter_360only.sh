#!/usr/bin/env bash
# Ablation Teacher Weight Stage0 epoch-50 export -> Improved Attn Flow Stage1.
# Submit twice with ABLATION_WEIGHT=0p0 and ABLATION_WEIGHT=0p1.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Ablation/Teacher_Weight
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=abl50_mrflow
#SBATCH --output=slurm_stage1_50ep/slurm-%j.out

set -euo pipefail

: "${ABLATION_WEIGHT:?Set ABLATION_WEIGHT to 0p0 or 0p1}"
case "$ABLATION_WEIGHT" in
  0p0|0p1) ;;
  *) echo "Unsupported ABLATION_WEIGHT=$ABLATION_WEIGHT" >&2; exit 2 ;;
esac

BASE_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Ablation/Teacher_Weight"
PROJECT_ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum"
TCR_DIR="$BASE_DIR/tcr_Weight$ABLATION_WEIGHT"
RUN_ROOT="$TCR_DIR/stage1_flow_50ep_gamma0_mrjitter_360only"
CONFIG="$TCR_DIR/config_flow_50ep_gamma0_mrjitter_360only.yaml"
STAGE0_CKPT="$TCR_DIR/checkpoints/pn_encoder_50ep.pt"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"
IMPROVED_LAB="$PROJECT_ROOT/Running_Lab/Improved_Attn_MOCOCO"

export RUN_ROOT
export STAGE0_CKPT
export MASTER_PORT="${MASTER_PORT:-30123}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/ablation_${ABLATION_WEIGHT}_50ep_mrjitter_numba_cache_${USER:-user}}"
export PYTHONPATH="$PROJECT_ROOT/Base/Code_Snippet:$IMPROVED_LAB:$PROJECT_ROOT/sia_fm_tse:$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT/checkpoints" "$NUMBA_CACHE_DIR"
cp -f "$CONFIG" "$RUN_ROOT/config_flow_50ep_gamma0_mrjitter_360only.yaml"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }

echo "[Ablation] teacher_weight=$ABLATION_WEIGHT"
echo "[Ablation] stage0=epoch-50 export"
echo "[Ablation] stage0_ckpt=$STAGE0_CKPT"
echo "[Ablation] loss.gamma=0.0"
echo "[Ablation] mr_jitter=true sigma=0.25"
echo "[Ablation] train_root=train-clean-360"
echo "[Ablation] speedups=Base/Code_Snippet/speedups.py"

exec "$PYTHON_BIN" "$IMPROVED_LAB/run_improved_meanflow_speedups.py" \
  --config "$CONFIG" \
  --stage0-ckpt "$STAGE0_CKPT" \
  --run-root "$RUN_ROOT"
