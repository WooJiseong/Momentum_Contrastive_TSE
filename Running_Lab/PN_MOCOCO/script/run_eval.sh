#!/usr/bin/env bash

# Purpose:
#   Evaluate a PN_MOCOCO TFGridNet checkpoint on a fixed-count test set.
#
# Run from:
#   Running_Lab/PN_MOCOCO
#
# Single GPU:
#   CUDA_VISIBLE_DEVICES=0 bash script/run_eval.sh
#
# Multi GPU:
#   Not supported. Evaluation runs on one visible GPU.
#
# Arguments:
#   None. Override CFG, CKPT, or OUT through environment variables.
#
# Output:
#   exp/YYYYMMDD_pn_mococo_eval[_runNN]/evaluation/results.json
#
# Environment:
#   conda activate pnflowtse
#   Set ALLOW_CPU_EVAL=1 only for intentional CPU evaluation.
#
# Slurm interactive example:
#   srun -p gpu6 --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=08:00:00 --pty bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PNFLOW_ROOT="$(cd "$PROJECT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

PY="${PY:-python}"
CFG="${CFG:-configs/config_tfgridnet_supervised.yaml}"
MOCO_CFG="${MOCO_CFG:-configs/config_moco_encoder.yaml}"
SOURCE_CFG="$CFG"
SOURCE_MOCO_CFG="$MOCO_CFG"
CKPT="${CKPT:-}"
OUT="${OUT:-}"

export PYTHONPATH="$PNFLOW_ROOT:$PROJECT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/pn_mococo_numba_cache_${USER:-user}}"
mkdir -p "$NUMBA_CACHE_DIR" exp

if [[ "${ALLOW_CPU_EVAL:-0}" != "1" ]]; then
  if ! "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "[PN_MOCOCO] CUDA is not available. Evaluation is too slow for the default fixed-count protocol on CPU."
    echo "[PN_MOCOCO] Run on a GPU node, or set ALLOW_CPU_EVAL=1 explicitly."
    exit 1
  fi
fi

record_runtime() {
  local run_dir="$1"
  mkdir -p "$run_dir"
  echo "bash script/run_eval.sh" > "$run_dir/command.txt"
  {
    echo "date=$(date -Is)"
    echo "project=$PROJECT_DIR"
    echo "pnflow_root=$PNFLOW_ROOT"
    echo "source_moco_config=$SOURCE_MOCO_CFG"
    echo "source_tfgrid_config=$SOURCE_CFG"
    echo "runtime_tfgrid_config=$CFG"
    echo "python=$("$PY" -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_version=" + str(torch.version.cuda)); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))' 2>/dev/null || true
    (git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unavailable) | sed 's/^/git_commit=/'
    nvidia-smi 2>/dev/null || true
  } > "$run_dir/environment.txt"
}

if [[ "${PREPARED_RUN:-0}" != "1" ]]; then
  eval "$("$PY" script/prepare_runtime_config.py --moco-config "$SOURCE_MOCO_CFG" --tfgrid-config "$SOURCE_CFG" --mode eval)"
  CFG="$TFGRID_CFG"
  export CFG RUN_DIR
  record_runtime "$RUN_DIR"
else
  RUN_DIR="${RUN_DIR:-$(dirname "$CFG")}"
fi

cmd=("$PY" eval_tfgridnet.py --config "$CFG")
if [[ -n "$CKPT" ]]; then
  cmd+=(--checkpoint "$CKPT")
fi
if [[ -n "$OUT" ]]; then
  cmd+=(--out "$OUT")
fi

echo "[PN_MOCOCO] CFG=$CFG"
echo "[PN_MOCOCO] CKPT=${CKPT:-config default}"
"${cmd[@]}"
