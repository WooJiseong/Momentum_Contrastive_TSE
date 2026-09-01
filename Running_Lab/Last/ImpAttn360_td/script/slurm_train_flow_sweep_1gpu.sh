#!/usr/bin/env bash
# Improved Attn Stage0 checkpoint sweep -> one-GPU Flow Stage1.
# Submit with SWEEP_POINT=50ep, 100ep, 200ep, or best.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Last/ImpAttn360_td
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=imp360_td_s1
#SBATCH --output=slurm_logs/slurm-%j.out

set -euo pipefail

: "${SWEEP_POINT:?Set SWEEP_POINT to 50ep, 100ep, 200ep, or best}"
case "$SWEEP_POINT" in
  50ep|100ep|200ep)
    STAGE0_FILE="pn_encoder_${SWEEP_POINT}.pt"
    RUN_NAME="stage0_${SWEEP_POINT}"
    ;;
  best)
    STAGE0_FILE="pn_encoder_best.pt"
    RUN_NAME="stage0_best"
    ;;
  *)
    echo "Unsupported SWEEP_POINT=$SWEEP_POINT" >&2
    exit 2
    ;;
esac

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Last/ImpAttn360_td"
PROJECT_ROOT="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum"
IMPROVED_LAB="$PROJECT_ROOT/Running_Lab/Improved_Attn_MOCOCO"
STAGE0_DIR="$IMPROVED_LAB/exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter/stage0_moco/checkpoints"
STAGE0_CKPT="$STAGE0_DIR/$STAGE0_FILE"
RUN_ROOT="$LAB_DIR/exp/20260829_impattn360_td_$RUN_NAME/stage1_flow_mrjitter"
CONFIG="$LAB_DIR/configs/config_flow_1gpu_batch8_mrjitter_360only.yaml"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"

export RUN_ROOT
export STAGE0_CKPT
export MASTER_PORT="${MASTER_PORT:-30150}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/impattn360_td_${SWEEP_POINT}_numba_cache_${USER:-user}}"
export PYTHONPATH="$PROJECT_ROOT/Base/Code_Snippet:$LAB_DIR:$IMPROVED_LAB:$PROJECT_ROOT/sia_fm_tse:$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT/checkpoints" "$NUMBA_CACHE_DIR"
cp -f "$CONFIG" "$RUN_ROOT/config_flow_1gpu_batch8_mrjitter_360only.yaml"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }

echo "[ImpAttn360_td] sweep_point=$SWEEP_POINT"
echo "[ImpAttn360_td] stage0_ckpt=$STAGE0_CKPT"
echo "[ImpAttn360_td] batch_size=8 num_gpus=1"
echo "[ImpAttn360_td] loss.gamma=0.0"
echo "[ImpAttn360_td] mr_jitter=true sigma=0.25"
echo "[ImpAttn360_td] train_root=train-clean-360"
echo "[ImpAttn360_td] speedups=Base/Code_Snippet/speedups.py"

exec "$PYTHON_BIN" "$LAB_DIR/run_flow_speedups.py" \
  --config "$CONFIG" \
  --stage0-ckpt "$STAGE0_CKPT" \
  --run-root "$RUN_ROOT"
