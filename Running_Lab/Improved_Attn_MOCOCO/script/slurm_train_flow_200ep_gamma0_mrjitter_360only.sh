#!/usr/bin/env bash
# Improved Attn Stage0 epoch-200 export -> gamma=0 MR-jitter Flow Stage1.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=imp200_mrflow
#SBATCH --output=exp/20260829_improved_attn_200ep_gamma0_mrjitter_360only/stage1_flow_mrjitter/slurm-%j.out

set -euo pipefail

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO"
PROJECT_ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum"
RUN_ROOT="$LAB_DIR/exp/20260829_improved_attn_200ep_gamma0_mrjitter_360only/stage1_flow_mrjitter"
CONFIG="$LAB_DIR/configs/config_flow_200ep_gamma0_mrjitter_360only.yaml"
STAGE0_CKPT="$LAB_DIR/exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter/stage0_moco/checkpoints/pn_encoder_200ep.pt"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"

export RUN_ROOT
export STAGE0_CKPT
export MASTER_PORT="${MASTER_PORT:-30122}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/improved_attn_200ep_gamma0_mrjitter_numba_cache_${USER:-user}}"
export PYTHONPATH="$PROJECT_ROOT/Base/Code_Snippet:$LAB_DIR:$PROJECT_ROOT/sia_fm_tse:$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT/checkpoints" "$NUMBA_CACHE_DIR"
cp -f "$CONFIG" "$LAB_DIR/exp/20260829_improved_attn_200ep_gamma0_mrjitter_360only/config_flow_200ep_gamma0_mrjitter_360only.yaml"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }

echo "[Improved_Attn] stage0=epoch-200 export"
echo "[Improved_Attn] stage0_ckpt=$STAGE0_CKPT"
echo "[Improved_Attn] loss.gamma=0.0"
echo "[Improved_Attn] mr_jitter=true sigma=0.25"
echo "[Improved_Attn] train_root=train-clean-360"
echo "[Improved_Attn] speedups=Base/Code_Snippet/speedups.py"

exec "$PYTHON_BIN" "$LAB_DIR/run_improved_meanflow_speedups.py" \
  --config "$CONFIG" \
  --stage0-ckpt "$STAGE0_CKPT" \
  --run-root "$RUN_ROOT"
