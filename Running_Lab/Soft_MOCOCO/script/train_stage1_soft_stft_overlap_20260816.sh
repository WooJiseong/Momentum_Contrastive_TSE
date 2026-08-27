#!/usr/bin/env bash
# Train and evaluate Soft_MOCOCO Stage1 with the STFT magnitude-overlap Gate.
# Run from Running_Lab/Soft_MOCOCO:
#   bash script/train_stage1_soft_stft_overlap_20260816.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTRASTIVE_ROOT="$(cd "$LAB_DIR/../.." && pwd)"
PNFLOW_ROOT="$(cd "$LAB_DIR/../../.." && pwd)"
cd "$LAB_DIR"

MODE="${1:-all}"
case "$MODE" in
  all|train|eval) ;;
  *)
    echo "Usage: bash script/train_stage1_soft_stft_overlap_20260816.sh [all|train|eval]" >&2
    exit 1
    ;;
esac

PY="${PY:-python}"
RUN_DIR="exp/20260816_stage0_50ep_soft_stft_overlap_gate"
TFGRID_CFG="configs/config_tfgridnet_soft_stft_overlap_20260816_stage0_50ep.yaml"
STAGE0_CKPT="exp/20260803_stage0_50ep_soft_wo_leakage_test/stage0_moco/checkpoints/pn_encoder_50ep.pt"
STAGE1_CKPT="$RUN_DIR/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt"

export PYTHONPATH="$LAB_DIR:$LAB_DIR/../PN_MOCOCO:$CONTRASTIVE_ROOT:$PNFLOW_ROOT:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29986}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_mococo_stft_overlap_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$NUMBA_CACHE_DIR" "$RUN_DIR/stage1_tfgridnet/checkpoints" "$RUN_DIR/evaluation"

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "$path" ]]; then
    echo "[Soft_MOCOCO] missing required file: $path" >&2
    echo "[Soft_MOCOCO] $hint" >&2
    exit 1
  fi
}

record_runtime() {
  {
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "mode=$MODE"
    echo "stage0_checkpoint=$STAGE0_CKPT"
    echo "stage0_source_experiment=20260803_stage0_50ep_soft_wo_leakage_test"
    echo "stage0_epoch=50"
    echo "tfgrid_config=$TFGRID_CFG"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "master_port=$MASTER_PORT"
    echo "python=$($PY -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))' 2>/dev/null || true
    git -C "$CONTRASTIVE_ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || echo "git_commit=unavailable"
    nvidia-smi 2>/dev/null || true
  } > "$RUN_DIR/environment.txt"
}

run_train() {
  require_file "$STAGE0_CKPT" \
    "The pinned Soft_MOCOCO Stage0 epoch-50 checkpoint is required."
  echo "[Soft_MOCOCO] ===== Stage1: STFT magnitude-overlap Gate ====="
  "$PY" script/run_tfgridnet_soft.py --config "$TFGRID_CFG" \
    2>&1 | tee "$RUN_DIR/stage1_tfgridnet/train.log"
}

run_eval() {
  require_file "$STAGE1_CKPT" "Run Stage1 before evaluation."
  echo "[Soft_MOCOCO] ===== 5000-item fixed-count evaluation ====="
  "$PY" script/run_eval.py --config "$TFGRID_CFG" \
    2>&1 | tee "$RUN_DIR/evaluation/eval.log"
}

require_file "$TFGRID_CFG" "The STFT overlap Gate config is required."
require_file "$STAGE0_CKPT" "The pinned Stage0 epoch-50 checkpoint is required."
record_runtime
cp "$TFGRID_CFG" "$RUN_DIR/source_config_tfgridnet_stft_overlap.yaml"
sha256sum "$STAGE0_CKPT" > "$RUN_DIR/stage0_encoder_sha256.txt"

case "$MODE" in
  all)
    run_train
    run_eval
    ;;
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
esac

echo "[Soft_MOCOCO] checkpoint=$STAGE1_CKPT"
echo "[Soft_MOCOCO] eval_json=$RUN_DIR/evaluation/results.json"
