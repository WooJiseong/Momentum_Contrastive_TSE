#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$LAB_DIR/../.." && pwd)"
CONFIG="${1:?Usage: bash script/run_stage0.sh /absolute/path/to/config.yaml}"
PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }

export PYTHONPATH="$PROJECT_ROOT/Running_Lab/Soft_MOCOCO:$PROJECT_ROOT/Running_Lab/PN_MOCOCO:$PROJECT_ROOT:$PROJECT_ROOT/..:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/data_robust_debug_numba_cache_${USER:-user}}"
mkdir -p "$NUMBA_CACHE_DIR"

echo "[data_robust_debug] config=$CONFIG"
echo "[data_robust_debug] init=/gpfs/home1/wjs6800/SIALab/PNFlowTSE/checkpoints/proposed-monaural.pt"
echo "[data_robust_debug] trainer=Soft_MOCOCO with Base/Code_Snippet/speedups.py"
echo "[data_robust_debug] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[data_robust_debug] MASTER_PORT=${MASTER_PORT:-unset}"

TRAIN_ARGS=(--config "$CONFIG")
if [[ -n "${RESUME_CKPT:-}" ]]; then
    [[ -f "$RESUME_CKPT" ]] || { echo "Resume checkpoint not found: $RESUME_CKPT" >&2; exit 1; }
    TRAIN_ARGS+=(--resume "$RESUME_CKPT")
    echo "[data_robust_debug] resume=$RESUME_CKPT"
fi

exec "$PYTHON_BIN" "$LAB_DIR/script/train_soft_moco_encoder_with_speedups.py" "${TRAIN_ARGS[@]}"
