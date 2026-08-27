#!/usr/bin/env bash
# Run only the Attn_MOCOCO SI-SDR + Leakage Stage1 variant.
# It is intentionally a separate Slurm process from the SI-SDR-only run.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO
#SBATCH -p gpu4
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=attn_leak_rerun
#SBATCH --output=exp/20260813_attn_mococo_40frame_lightweight_attention/slurm-leak-rerun-%j.out

set -euo pipefail

export PROJECT_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Attn_MOCOCO"
export PYTHON_BIN="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"
export MASTER_PORT="${MASTER_PORT:-30011}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_mococo_leakage_rerun_numba_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Fail the distributed job promptly if one rank dies instead of leaving the
# remaining ranks in a silent NCCL collective wait.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

cd "$PROJECT_DIR"
mkdir -p "$NUMBA_CACHE_DIR" \
  exp/20260813_attn_mococo_40frame_lightweight_attention/stage1_tfgridnet_si_sdr_leakage_rerun
export PYTHONPATH="$PROJECT_DIR:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/PN_MOCOCO:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

RUN_LOG="exp/20260813_attn_mococo_40frame_lightweight_attention/stage1_tfgridnet_si_sdr_leakage_rerun/train.log"
echo "[Attn leakage rerun] python=$PYTHON_BIN"
echo "[Attn leakage rerun] config=configs/config_tfgridnet_attn_si_sdr_leakage_rerun.yaml"
echo "[Attn leakage rerun] train_log=$RUN_LOG"

# Keep Slurm's output and an experiment-local copy synchronized.  `-u` is
# required because this script is launched without an interactive TTY.
"$PYTHON_BIN" -u script/run_tfgridnet_attn.py \
  --config configs/config_tfgridnet_attn_si_sdr_leakage_rerun.yaml \
  2>&1 | tee -a "$RUN_LOG"
