#!/usr/bin/env bash
# Teacher-weight ablation: fixed teacher coefficient 0.1.

#SBATCH -D /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Ablation/Teacher_Weight/tcr_Weight0p1
#SBATCH -p gpu6
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=tcr0p1_s0
#SBATCH --output=slurm-%j.out

set -euo pipefail

LAB_DIR="/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Ablation/Teacher_Weight/tcr_Weight0p1"
CONFIG="$LAB_DIR/config_stage0.yaml"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"

export MASTER_PORT="${MASTER_PORT:-30131}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/tcr_weight0p1_numba_cache_${USER:-user}}"
export PYTHONPATH="$LAB_DIR:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Base/Code_Snippet:/gpfs/home1/wjs6800/SIALab/PNFlowTSE/sia_fm_tse:/gpfs/home1/wjs6800/SIALab/PNFlowTSE:${PYTHONPATH:-}"

mkdir -p "$LAB_DIR/logs" "$LAB_DIR/checkpoints" "$NUMBA_CACHE_DIR"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }

echo "[TeacherWeightAblation] teacher_weight=0.1"
echo "[TeacherWeightAblation] config=$CONFIG"
echo "[TeacherWeightAblation] output=$LAB_DIR"
echo "[TeacherWeightAblation] speedups=$LAB_DIR/train_with_speedups.py"

exec "$PYTHON_BIN" "$LAB_DIR/train_with_speedups.py" --config "$CONFIG"
