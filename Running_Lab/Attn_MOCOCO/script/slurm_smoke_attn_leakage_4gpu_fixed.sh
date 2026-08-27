#!/usr/bin/env bash
# Fixed four-GPU DDP smoke test for Attn leakage Stage1.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=attn_leak_sm4fx
#SBATCH --output=exp/20260813_attn_mococo_40frame_lightweight_attention/slurm-leak-smoke4-fixed-%j.out

set -euo pipefail

export PROJECT_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO"
export PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
export MASTER_PORT="${MASTER_PORT:-30022}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_mococo_leakage_smoke4_fixed_numba_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="1"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export TORCH_NCCL_BLOCKING_WAIT="1"
export TORCH_NCCL_ENABLE_MONITORING="1"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="120"
export ATTN_LEAKAGE_STEP_DEBUG="1"

cd "$PROJECT_DIR"
mkdir -p "$NUMBA_CACHE_DIR" \
  exp/20260813_attn_mococo_40frame_lightweight_attention/stage1_tfgridnet_si_sdr_leakage_smoke_4gpu_fixed
export PYTHONPATH="$PROJECT_DIR:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/PN_MOCOCO:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

RUN_LOG="exp/20260813_attn_mococo_40frame_lightweight_attention/stage1_tfgridnet_si_sdr_leakage_smoke_4gpu_fixed/train.log"
echo "[Attn leakage 4GPU fixed smoke] python=$PYTHON_BIN"
echo "[Attn leakage 4GPU fixed smoke] config=configs/config_tfgridnet_attn_si_sdr_leakage_smoke_4gpu_fixed.yaml"
"$PYTHON_BIN" -u script/run_tfgridnet_attn.py \
  --config configs/config_tfgridnet_attn_si_sdr_leakage_smoke_4gpu_fixed.yaml \
  2>&1 | tee -a "$RUN_LOG"
