#!/usr/bin/env bash

# Purpose:
#   Run the PN_MOCOCO momentum-contrastive encoder, TFGridNet fine-tuning, and evaluation workflow.
#
# Run from:
#   Running_Lab/PN_MOCOCO
#
# Single GPU:
#   CUDA_VISIBLE_DEVICES=0 bash script/train_eval.sh all
#
# Multi GPU:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/train_eval.sh all
#
# Arguments:
#   all | train | eval | moco | tfgridnet
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

MODE="${1:-all}"
case "$MODE" in
  all|train|eval|moco|tfgridnet) ;;
  *)
    echo "Usage: bash script/train_eval.sh [all|train|eval|moco|tfgridnet]"
    exit 1
    ;;
esac

PY="${PY:-python}"
MOCO_CFG="${MOCO_CFG:-configs/config_moco_encoder.yaml}"
TFGRID_CFG="${TFGRID_CFG:-configs/config_tfgridnet_supervised.yaml}"
SOURCE_MOCO_CFG="$MOCO_CFG"
SOURCE_TFGRID_CFG="$TFGRID_CFG"

export PYTHONPATH="$PNFLOW_ROOT:$PROJECT_DIR:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29724}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/pn_mococo_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DRY_RUN="${DRY_RUN:-0}"
RECORD_RUNTIME_ONLY="${RECORD_RUNTIME_ONLY:-0}"
mkdir -p "$NUMBA_CACHE_DIR" exp

if [[ "$DRY_RUN" != "1" && "$RECORD_RUNTIME_ONLY" != "1" && ("$MODE" == "all" || "$MODE" == "eval") && "${ALLOW_CPU_EVAL:-0}" != "1" ]]; then
  if ! "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "[PN_MOCOCO train_eval] CUDA is not available. Evaluation is too slow for the default fixed-count protocol on CPU."
    echo "[PN_MOCOCO train_eval] Run on a GPU node, or set ALLOW_CPU_EVAL=1 explicitly."
    exit 1
  fi
fi

record_runtime() {
  local run_dir="$1"
  mkdir -p "$run_dir"
  printf 'bash %q' "$0" > "$run_dir/command.txt"
  for arg in "$@"; do
    if [[ "$arg" != "$run_dir" ]]; then
      printf ' %q' "$arg" >> "$run_dir/command.txt"
    fi
  done
  printf '\n' >> "$run_dir/command.txt"
  {
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "user=$(whoami)"
    echo "project=$PROJECT_DIR"
    echo "pnflow_root=$PNFLOW_ROOT"
    echo "source_moco_config=$SOURCE_MOCO_CFG"
    echo "source_tfgrid_config=$SOURCE_TFGRID_CFG"
    echo "runtime_moco_config=$MOCO_CFG"
    echo "runtime_tfgrid_config=$TFGRID_CFG"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "python=$("$PY" -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda_version=" + str(torch.version.cuda)); print("cuda_available=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))' 2>/dev/null || true
    (git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unavailable) | sed 's/^/git_commit=/'
    nvidia-smi 2>/dev/null || true
  } > "$run_dir/environment.txt"
}

SHOULD_RECORD_RUNTIME=0
if [[ "${PREPARED_RUN:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  eval "$("$PY" script/prepare_runtime_config.py --moco-config "$SOURCE_MOCO_CFG" --tfgrid-config "$SOURCE_TFGRID_CFG" --mode "$MODE")"
  export MOCO_CFG TFGRID_CFG RUN_DIR
  SHOULD_RECORD_RUNTIME=1
else
  RUN_DIR="${RUN_DIR:-$(dirname "$TFGRID_CFG")}"
  if [[ "$DRY_RUN" != "1" || "$RECORD_RUNTIME_ONLY" == "1" ]]; then
    if [[ ! -f "$RUN_DIR/command.txt" && ! -f "$RUN_DIR/environment.txt" ]]; then
      SHOULD_RECORD_RUNTIME=1
    fi
  fi
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
    echo "[PN_MOCOCO train_eval] missing required file: $path"
    echo "[PN_MOCOCO train_eval] $hint"
    exit 1
  fi
}

MOCO_GPUS="$(yaml_value "$MOCO_CFG" "cfg.get('ddp', {}).get('num_gpus', 1)")"
TFGRID_GPUS="$(yaml_value "$TFGRID_CFG" "cfg.get('ddp', {}).get('num_gpus', 1)")"
if [[ "$MOCO_GPUS" != "$TFGRID_GPUS" ]]; then
  echo "[PN_MOCOCO train_eval] MOCO_CFG ddp.num_gpus=$MOCO_GPUS but TFGRID_CFG ddp.num_gpus=$TFGRID_GPUS"
  echo "[PN_MOCOCO train_eval] Use matching GPU counts in YAML configs."
  exit 1
fi
CFG_GPUS="$MOCO_GPUS"

if [[ "$MODE" == "all" || "$MODE" == "train" || "$MODE" == "moco" || "$MODE" == "tfgridnet" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(visible_gpu_list "$CFG_GPUS")}"
  VISIBLE_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"

  if [[ "$VISIBLE_GPUS" != "$CFG_GPUS" ]]; then
    echo "[PN_MOCOCO train_eval] CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPUS GPU(s), but YAML ddp.num_gpus=$CFG_GPUS"
    echo "[PN_MOCOCO train_eval] Fix CUDA_VISIBLE_DEVICES or ddp.num_gpus before running."
    exit 1
  fi
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

if [[ "$SHOULD_RECORD_RUNTIME" == "1" ]]; then
  record_runtime "$RUN_DIR" "$MODE"
fi

if [[ "$RECORD_RUNTIME_ONLY" == "1" ]]; then
  echo "[PN_MOCOCO train_eval] RECORD_RUNTIME_ONLY=1, stopping after metadata capture."
  exit 0
fi

MOCO_LOG_DIR="$(yaml_value "$MOCO_CFG" "cfg['train']['log_dir']")"
TFGRID_LOG_DIR="$(yaml_value "$TFGRID_CFG" "cfg['train']['log_dir']")"
MOCO_INIT="$(yaml_value "$MOCO_CFG" "cfg['contrastive']['initial_pn_ckpt']")"
TFGRID_INIT="$(yaml_value "$TFGRID_CFG" "cfg.get('paths', {}).get('initial_model_ckpt') or ''")"
TFGRID_ENCODER="$(yaml_value "$TFGRID_CFG" "cfg.get('paths', {}).get('encoder_override_ckpt') or ''")"
TFGRID_CKPT="$(yaml_value "$TFGRID_CFG" "cfg.get('eval', {}).get('checkpoint', '')")"
TFGRID_OUT="$(yaml_value "$TFGRID_CFG" "cfg.get('eval', {}).get('out_json', '')")"

mkdir -p "$MOCO_LOG_DIR" "$TFGRID_LOG_DIR"

echo "[PN_MOCOCO train_eval] MODE=$MODE"
echo "[PN_MOCOCO train_eval] project=$PROJECT_DIR"
echo "[PN_MOCOCO train_eval] pnflow_root=$PNFLOW_ROOT"
echo "[PN_MOCOCO train_eval] python=$("$PY" -c 'import sys; print(sys.executable)')"
echo "[PN_MOCOCO train_eval] MOCO_CFG=$MOCO_CFG"
echo "[PN_MOCOCO train_eval] TFGRID_CFG=$TFGRID_CFG"
echo "[PN_MOCOCO train_eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  YAML ddp.num_gpus=$CFG_GPUS  MASTER_PORT=$MASTER_PORT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[PN_MOCOCO train_eval] DRY_RUN=1, stopping before train/eval."
  exit 0
fi

if [[ "$MODE" == "all" || "$MODE" == "train" || "$MODE" == "moco" ]]; then
  require_file "$MOCO_INIT" "Update $MOCO_CFG contrastive.initial_pn_ckpt."
  echo "[PN_MOCOCO train_eval] ===== Stage 0: MOCO encoder training ====="
  PREPARED_RUN=1 RUN_DIR="$RUN_DIR" MOCO_CFG="$MOCO_CFG" TFGRID_CFG="$TFGRID_CFG" bash script/run_train.sh moco
fi

if [[ "$MODE" == "all" || "$MODE" == "train" || "$MODE" == "tfgridnet" ]]; then
  require_file "$TFGRID_INIT" "Update $TFGRID_CFG paths.initial_model_ckpt."
  require_file "$TFGRID_ENCODER" "Run MOCO first, or set $TFGRID_CFG paths.encoder_override_ckpt to an existing file."
  echo "[PN_MOCOCO train_eval] ===== Stage 1: TFGridNet supervised training ====="
  PREPARED_RUN=1 RUN_DIR="$RUN_DIR" MOCO_CFG="$MOCO_CFG" TFGRID_CFG="$TFGRID_CFG" bash script/run_train.sh tfgridnet
fi

if [[ "$MODE" == "all" || "$MODE" == "eval" ]]; then
  require_file "$TFGRID_CKPT" "Run TFGridNet training first, or update $TFGRID_CFG eval.checkpoint."
  echo "[PN_MOCOCO train_eval] ===== Stage 2: Evaluation ====="
  PREPARED_RUN=1 RUN_DIR="$RUN_DIR" CFG="$TFGRID_CFG" bash script/run_eval.sh
fi

echo "[PN_MOCOCO train_eval] done"
if [[ -n "$TFGRID_CKPT" ]]; then
  echo "[PN_MOCOCO train_eval] checkpoint=$TFGRID_CKPT"
fi
if [[ -n "$TFGRID_OUT" ]]; then
  echo "[PN_MOCOCO train_eval] eval_json=$TFGRID_OUT"
fi
