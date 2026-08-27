#!/usr/bin/env bash
# Train the default rectified-flow Oracle Stage1 from Attn teacher=0.03.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn03_oracle_mf
#SBATCH --output=exp/20260824_attn_teacher_0p03_default_flow_oracle_stage1/slurm-%j.out

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-30027}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MF_CONFIG="${MF_CONFIG:-$PWD/configs/config_meanflow_attn_teacher_0p03_oracle.yaml}"
export STAGE0_CKPT="${STAGE0_CKPT:-/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO/exp/20260819_attn_teacher_weight_sweep/teacher_0p03/stage0_moco/checkpoints/pn_encoder_best.pt}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_teacher_0p03_oracle_mf_${USER:-user}}"
export RESUME_CKPT="${RESUME_CKPT:-}"
# Pure rectified flow has a fixed autograd graph; avoid unnecessary DDP graph
# traversal and the epoch-end collective ordering issue seen in 896371/896460.
export DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-0}"

mkdir -p exp/20260824_attn_teacher_0p03_default_flow_oracle_stage1 "$NUMBA_CACHE_DIR"
[[ -f "$MF_CONFIG" ]] || { echo "MeanFlow config not found: $MF_CONFIG" >&2; exit 1; }
[[ -f "$STAGE0_CKPT" ]] || { echo "Stage0 checkpoint not found: $STAGE0_CKPT" >&2; exit 1; }

echo "[Attn teacher=0.03 Oracle] config=$MF_CONFIG"
echo "[Attn teacher=0.03 Oracle] stage0=$STAGE0_CKPT"
echo "[Attn teacher=0.03 Oracle] DDP_FIND_UNUSED=$DDP_FIND_UNUSED"

bash script/train_meanflow_mococo.sh
