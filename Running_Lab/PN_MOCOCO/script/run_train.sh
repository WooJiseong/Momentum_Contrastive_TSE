#!/usr/bin/env bash

# Purpose:
#   Train one or both PN_MOCOCO stages using runtime configs under exp/.
#
# Run from:
#   Running_Lab/PN_MOCOCO
#
# Single GPU:
#   CUDA_VISIBLE_DEVICES=0 bash script/run_train.sh moco
#
# Multi GPU:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train.sh all
#
# Arguments:
#   moco | tfgridnet | all
#
# Output:
#   exp/YYYYMMDD_pn_mococo[_runNN]/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm interactive example:
#   srun -p gpu6 --gres=gpu:4 --cpus-per-task=16 --mem=48G --time=72:00:00 --pty bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PNFLOW_ROOT="$(cd "$PROJECT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

PY="${PY:-python}"
MODE="${1:-}"
MOCO_CFG="${MOCO_CFG:-configs/config_moco_encoder.yaml}"
TFGRID_CFG="${TFGRID_CFG:-configs/config_tfgridnet_supervised.yaml}"
SOURCE_MOCO_CFG="$MOCO_CFG"
SOURCE_TFGRID_CFG="$TFGRID_CFG"

export PYTHONPATH="$PNFLOW_ROOT:$PROJECT_DIR:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29720}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/pn_mococo_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$NUMBA_CACHE_DIR" exp

record_runtime() {
  local run_dir="$1"
  mkdir -p "$run_dir"
  echo "bash script/run_train.sh $MODE" > "$run_dir/command.txt"
  {
    echo "date=$(date -Is)"
    echo "project=$PROJECT_DIR"
    echo "pnflow_root=$PNFLOW_ROOT"
    echo "source_moco_config=$SOURCE_MOCO_CFG"
    echo "source_tfgrid_config=$SOURCE_TFGRID_CFG"
    echo "runtime_moco_config=$MOCO_CFG"
    echo "runtime_tfgrid_config=$TFGRID_CFG"
    echo "python=$("$PY" -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_version=" + str(torch.version.cuda)); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))' 2>/dev/null || true
    (git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unavailable) | sed 's/^/git_commit=/'
    nvidia-smi 2>/dev/null || true
  } > "$run_dir/environment.txt"
}

case "$MODE" in
  moco|tfgridnet|train|all) ;;
  *)
    echo "Usage: bash script/run_train.sh {moco|tfgridnet|all}"
    echo "  moco      = pretrain PN encoder with momentum contrastive learning"
    echo "  tfgridnet = fine-tune original causal TFGridNet using the MOCO encoder"
    echo "  all       = run moco -> tfgridnet"
    exit 1
    ;;
esac

if [[ "$MODE" == "train" ]]; then
  MODE="tfgridnet"
fi

if [[ "${PREPARED_RUN:-0}" != "1" ]]; then
  eval "$("$PY" script/prepare_runtime_config.py --moco-config "$SOURCE_MOCO_CFG" --tfgrid-config "$SOURCE_TFGRID_CFG" --mode "$MODE")"
  export MOCO_CFG TFGRID_CFG RUN_DIR
  record_runtime "$RUN_DIR"
else
  RUN_DIR="${RUN_DIR:-$(dirname "$TFGRID_CFG")}"
fi

yaml_value() {
  local cfg="$1"
  local expr="$2"
  "$PY" -c "import yaml; cfg=yaml.safe_load(open('$cfg')); print($expr)"
}

visible_gpu_list() {
  local n="$1"
  "$PY" - "$n" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(i) for i in range(n)))
PY
}

require_file() {
  local path="$1"
  local hint="$2"
  if [[ -n "$path" && ! -f "$path" ]]; then
    echo "[PN_MOCOCO] missing required file: $path"
    echo "[PN_MOCOCO] $hint"
    exit 1
  fi
}

run_moco() {
  echo "[PN_MOCOCO] ===== Stage 0: momentum contrastive PN encoder ====="
  echo "[PN_MOCOCO] CFG=$MOCO_CFG"
  local init_ckpt
  init_ckpt="$(yaml_value "$MOCO_CFG" "cfg['contrastive']['initial_pn_ckpt']")"
  require_file "$init_ckpt" "Update $MOCO_CFG contrastive.initial_pn_ckpt."
  mkdir -p "$(yaml_value "$MOCO_CFG" "cfg['train']['log_dir']")"
  "$PY" train_moco_encoder.py --config "$MOCO_CFG" 2>&1 | tee "$(yaml_value "$MOCO_CFG" "cfg['train']['log_dir']")/train.log"
}

run_tfgridnet() {
  echo "[PN_MOCOCO] ===== Stage 1: supervised causal TFGridNet fine-tuning ====="
  echo "[PN_MOCOCO] CFG=$TFGRID_CFG"
  local init_ckpt encoder_ckpt
  init_ckpt="$(yaml_value "$TFGRID_CFG" "cfg.get('paths', {}).get('initial_model_ckpt') or ''")"
  encoder_ckpt="$(yaml_value "$TFGRID_CFG" "cfg.get('paths', {}).get('encoder_override_ckpt') or ''")"
  require_file "$init_ckpt" "Update $TFGRID_CFG paths.initial_model_ckpt."
  require_file "$encoder_ckpt" "Run 'bash script/run_train.sh moco' first, or set paths.encoder_override_ckpt to ''."
  mkdir -p "$(yaml_value "$TFGRID_CFG" "cfg['train']['log_dir']")"
  "$PY" train_tfgridnet.py --config "$TFGRID_CFG" 2>&1 | tee "$(yaml_value "$TFGRID_CFG" "cfg['train']['log_dir']")/train.log"
}

echo "[PN_MOCOCO] project=$PROJECT_DIR"
echo "[PN_MOCOCO] pnflow_root=$PNFLOW_ROOT"
echo "[PN_MOCOCO] python=$("$PY" -c 'import sys; print(sys.executable)')"
CFG_GPUS="$(yaml_value "$MOCO_CFG" "cfg.get('ddp', {}).get('num_gpus', 1)")"
if [[ "$MODE" == "tfgridnet" || "$MODE" == "train" ]]; then
  CFG_GPUS="$(yaml_value "$TFGRID_CFG" "cfg.get('ddp', {}).get('num_gpus', 1)")"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(visible_gpu_list "$CFG_GPUS")}"
echo "[PN_MOCOCO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MASTER_PORT=$MASTER_PORT"

case "$MODE" in
  moco)
    run_moco
    ;;
  tfgridnet)
    run_tfgridnet
    ;;
  all)
    run_moco
    run_tfgridnet
    ;;
esac
