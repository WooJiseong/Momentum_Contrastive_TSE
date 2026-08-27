#!/usr/bin/env bash
# Four-GPU Attn_MOCOCO Stage1 SI-SDR + revised leakage training.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn_leak_4gpu
#SBATCH --output=exp/20260813_attn_mococo_40frame_lightweight_attention/slurm-leak-rerun-4gpu-%j.out

set -euo pipefail

export PROJECT_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO"
export PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
export MASTER_PORT="${MASTER_PORT:-30022}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_mococo_leakage_rerun4_numba_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="1"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export TORCH_NCCL_BLOCKING_WAIT="1"
export NCCL_ASYNC_ERROR_HANDLING="1"

cd "$PROJECT_DIR"
mkdir -p "$NUMBA_CACHE_DIR" \
  exp/20260813_attn_mococo_40frame_lightweight_attention/stage1_tfgridnet_si_sdr_leakage_rerun_4gpu
export PYTHONPATH="$PROJECT_DIR:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/PN_MOCOCO:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

RUN_LOG="exp/20260813_attn_mococo_40frame_lightweight_attention/stage1_tfgridnet_si_sdr_leakage_rerun_4gpu/train.log"
echo "[Attn leakage 4GPU] python=$PYTHON_BIN"
echo "[Attn leakage 4GPU] config=configs/config_tfgridnet_attn_si_sdr_leakage_rerun_4gpu.yaml"
"$PYTHON_BIN" -u script/run_tfgridnet_attn.py \
  --config configs/config_tfgridnet_attn_si_sdr_leakage_rerun_4gpu.yaml \
  2>&1 | tee -a "$RUN_LOG"
