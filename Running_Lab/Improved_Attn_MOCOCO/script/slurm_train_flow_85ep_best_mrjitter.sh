#!/usr/bin/env bash
# Improved_Attn Best Stage0 at the epoch-85 capture -> MR-jitter Flow Stage1.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=imp85_mrflow
#SBATCH --output=exp/20260827_improved_attn_85ep_best_mrjitter/stage1_flow_mrjitter/slurm-%j.out

set -euo pipefail

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO"
RUN_ROOT="$LAB_DIR/exp/20260827_improved_attn_85ep_best_mrjitter"
STAGE1_ROOT="$RUN_ROOT/stage1_flow_mrjitter"
CONFIG="$LAB_DIR/configs/config_flow_85ep_best_mrjitter.yaml"
STAGE0_CKPT="$RUN_ROOT/stage0_input/pn_encoder_best_at_epoch85.pt"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"

# Lightning may re-execute this wrapper for DDP child ranks. Keep the paths
# available to those processes even when the original CLI arguments are gone.
export RUN_ROOT
export STAGE0_CKPT

export MASTER_PORT="${MASTER_PORT:-30119}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_85ep_mrflow_numba_cache_${USER:-user}}"
export PYTHONPATH="$LAB_DIR:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$STAGE1_ROOT/checkpoints" "$NUMBA_CACHE_DIR"
cp -f "$CONFIG" "$RUN_ROOT/config_flow_85ep_best_mrjitter.yaml"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$STAGE0_CKPT" ]] || { echo "Frozen Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }

echo "[Improved_Attn] Stage0=Best captured at epoch 85"
echo "[Improved_Attn] stage0_ckpt=$STAGE0_CKPT"
echo "[Improved_Attn] mr_jitter=true sigma=0.25"
echo "[Improved_Attn] train_root=train-clean-360"

exec "$PYTHON_BIN" "$LAB_DIR/train_improved_meanflow.py" \
  --config "$CONFIG" \
  --stage0-ckpt "$STAGE0_CKPT" \
  --run-root "$STAGE1_ROOT"
