#!/usr/bin/env bash
# Soft_MOCOCO Stage0: train-clean-360 only, fixed teacher weight 0.03.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=soft360_t03_s0
#SBATCH --output=exp/20260831_soft_mococo_360only_teacher_0p03_fixed/stage0_moco/slurm-%j.out

set -euo pipefail

RUN_ROOT="$PWD/exp/20260831_soft_mococo_360only_teacher_0p03_fixed"
CONFIG="$PWD/configs/config_soft_moco_360only_teacher_0p03_fixed.yaml"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"

export MASTER_PORT="${MASTER_PORT:-30230}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft360_teacher_0p03_numba_cache_${USER:-user}}"
export PYTHONPATH="$PWD:$PWD/../PN_MOCOCO:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT/stage0_moco/checkpoints" "$NUMBA_CACHE_DIR"
cp -f "$CONFIG" "$RUN_ROOT/config_soft_moco_360only_teacher_0p03_fixed.yaml"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }

echo "[Soft_MOCOCO] encoder=standard num_blocks=3"
echo "[Soft_MOCOCO] initial_checkpoint=proposed-monaural.pt (not improved)"
echo "[Soft_MOCOCO] train_root=LibriSpeech/train-clean-360"
echo "[Soft_MOCOCO] teacher_weight=0.03 fixed"
echo "[Soft_MOCOCO] speaker_aware_queue=true"
echo "[Soft_MOCOCO] metric_aggregation=manual_global_epoch_mean"
echo "[Soft_MOCOCO] num_gpus=4 batch_size_per_gpu=2"
echo "[Soft_MOCOCO] periodic_exports=50,100,150,200,250,300 epochs"

exec "$PYTHON_BIN" "$PWD/train_soft_moco_teacher_decay.py" --config "$CONFIG"
