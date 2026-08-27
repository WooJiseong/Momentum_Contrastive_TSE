#!/usr/bin/env bash
# Resume the interrupted Attn_MOCOCO Stage0 run from its Lightning checkpoint.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn_mococo_resume
#SBATCH --output=exp/20260813_attn_mococo_40frame_lightweight_attention/slurm-%j.out

set -euo pipefail

PROJECT_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO"
EXP_DIR="$PROJECT_DIR/exp/20260813_attn_mococo_40frame_lightweight_attention"
PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"

export MASTER_PORT="${MASTER_PORT:-29990}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/../PN_MOCOCO:$PROJECT_DIR/../../..:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_mococo_numba_cache_${USER:-user}}"

cd "$PROJECT_DIR"
mkdir -p "$NUMBA_CACHE_DIR" "$EXP_DIR/stage0_moco"

RESUME_CKPT="$EXP_DIR/stage0_moco/checkpoints/last.ckpt"
RESUME_CONFIG="$EXP_DIR/source_config_attn_moco.yaml"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$RESUME_CKPT" ]] || { echo "Checkpoint not found: $RESUME_CKPT" >&2; exit 1; }
[[ -f "$RESUME_CONFIG" ]] || { echo "Config not found: $RESUME_CONFIG" >&2; exit 1; }

echo "[Attn_MOCOCO] Resuming Stage0 from $RESUME_CKPT"
echo "[Attn_MOCOCO] Config: $RESUME_CONFIG"
"$PYTHON_BIN" train_attn_moco_encoder.py \
  --config "$RESUME_CONFIG" \
  2>&1 | tee "$EXP_DIR/stage0_moco/resume_train.log"
