#!/usr/bin/env bash
# Train both Attn_MOCOCO Stage1 variants from the completed Stage0 Best export.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn_stage1_best
#SBATCH --output=exp/20260813_attn_mococo_40frame_lightweight_attention/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29991}"
export PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"

EXP_DIR="exp/20260813_attn_mococo_40frame_lightweight_attention"
STAGE0_BEST="$EXP_DIR/stage0_moco/checkpoints/pn_encoder_best.pt"

cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$STAGE0_BEST" ]] || { echo "Stage0 Best export not found: $STAGE0_BEST" >&2; exit 1; }

echo "[Attn_MOCOCO] Stage1 encoder override: $STAGE0_BEST"
echo "[Attn_MOCOCO] Running SI-SDR and SI-SDR+Leakage variants"

bash script/run_train.sh stage1
