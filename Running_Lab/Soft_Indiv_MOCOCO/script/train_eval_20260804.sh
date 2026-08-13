#!/usr/bin/env bash
#
# Purpose:
#   Run Soft_Indiv_MOCOCO Stage0, a selected Stage0 checkpoint's Stage1, and evaluation.
#
# Run from:
#   contrastive_momentum/Running_Lab/Soft_Indiv_MOCOCO
#
# Single GPU:
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash script/train_eval_20260804.sh moco
#
# Multi GPU:
#   NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29840 \
#     bash script/train_eval_20260804.sh all
#
# Required arg:
#   all | moco | stage1 | eval
#
# Optional environment:
#   RUN_DIR=exp/YYYYMMDD_title
#   STAGE0_SELECTOR=best | 50 | 100 | ...
#   RESUME_STAGE0=/path/to/stage0_epoch_050.ckpt
#   RESUME_STAGE1=/path/to/tfgridnet_last.ckpt
#   REUSE_RUN=1 (required when continuing an existing run directory)
#
# Output:
#   exp/YYYYMMDD_soft_indiv_mococo/
#
# Environment:
#   conda activate pnflowtse
#
# Slurm:
#   mkdir -p exp/20260804_soft_indiv_mococo
#   sbatch script/slurm_train_eval_20260804.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTRASTIVE_ROOT="$(cd "$LAB_DIR/../.." && pwd)"
PNFLOW_ROOT="$(cd "$LAB_DIR/../../.." && pwd)"
cd "$LAB_DIR"

MODE="${1:-all}"
case "$MODE" in
  all|moco|stage1|eval) ;;
  *)
    echo "Usage: bash script/train_eval_20260804.sh [all|moco|stage1|eval]"
    exit 1
    ;;
esac

PY="${PY:-python}"
RUN_DIR="${RUN_DIR:-exp/20260804_soft_indiv_mococo}"
NUM_GPUS="${NUM_GPUS:-4}"
STAGE0_SELECTOR="${STAGE0_SELECTOR:-best}"
MOCO_CFG="${MOCO_CFG:-configs/config_soft_indiv_moco.yaml}"
TFGRID_CFG="${TFGRID_CFG:-configs/config_tfgridnet_soft_indiv.yaml}"

export PYTHONPATH="$LAB_DIR:$LAB_DIR/../PN_Indiv_MOCOCO:$LAB_DIR/../PN_MOCOCO:$CONTRASTIVE_ROOT:$PNFLOW_ROOT:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29840}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/soft_indiv_mococo_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# PyTorch 2.6 defaults torch.load to weights_only=True. The trusted local
# Lightning checkpoint used for an explicit Stage0 resume contains full
# optimizer/trainer state and must be loaded with weights_only=False.
if [[ -n "${RESUME_STAGE0:-}" ]]; then
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
fi

mkdir -p "$NUMBA_CACHE_DIR"

ensure_run_dir() {
  if [[ -e "$RUN_DIR/run_manifest.txt" && "${REUSE_RUN:-0}" != "1" ]]; then
    echo "[Soft_Indiv_MOCOCO] existing run is protected: $RUN_DIR"
    echo "[Soft_Indiv_MOCOCO] set REUSE_RUN=1 and a RESUME_STAGE{0,1} checkpoint to continue it."
    exit 1
  fi
  mkdir -p "$RUN_DIR"
  if [[ ! -e "$RUN_DIR/run_manifest.txt" ]]; then
    {
      echo "experiment=soft_indiv_mococo"
      echo "created_at=$(date -Is)"
      echo "seed=42"
      echo "stage0_loss=individual_negative_moco + 0.1*frozen_teacher_cosine"
      echo "stage1_loss=negative_si_sdr + 0.1*target_orthogonal_leakage"
      echo "stage0_sweep_interval_epochs=50"
    } > "$RUN_DIR/run_manifest.txt"
  fi
}

record_environment() {
  local suffix="initial"
  if [[ "${REUSE_RUN:-0}" == "1" ]]; then
    suffix="resume_$(date +%Y%m%dT%H%M%S)"
  fi
  {
    echo "date=$(date -Is)"
    echo "mode=$MODE"
    echo "run_dir=$RUN_DIR"
    echo "stage0_selector=$STAGE0_SELECTOR"
    echo "num_gpus=$NUM_GPUS"
    echo "seed=42"
    echo "python=$($PY -c 'import sys; print(sys.executable)')"
    "$PY" -c 'import sys; print("python_version=" + sys.version.replace("\n", " "))'
    "$PY" -c 'import torch; print("torch_version=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("gpu_count=" + str(torch.cuda.device_count()))'
    git -C "$LAB_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/' || echo "git_commit=unavailable"
    nvidia-smi 2>/dev/null || true
  } > "$RUN_DIR/environment_${suffix}.txt"
}

stage0_encoder_path() {
  if [[ "$STAGE0_SELECTOR" == "best" ]]; then
    printf '%s\n' "$RUN_DIR/stage0_moco/checkpoints/pn_encoder_best.pt"
    return
  fi
  if [[ ! "$STAGE0_SELECTOR" =~ ^[0-9]+$ ]]; then
    echo "STAGE0_SELECTOR must be best or an epoch number such as 50." >&2
    exit 1
  fi
  printf '%s\n' "$RUN_DIR/stage0_moco/checkpoints/pn_encoder_epoch_$(printf '%03d' "$STAGE0_SELECTOR").pt"
}

stage1_label() {
  if [[ "$STAGE0_SELECTOR" == "best" ]]; then
    printf '%s\n' "stage0_best"
  else
    printf 'stage0_epoch_%03d\n' "$STAGE0_SELECTOR"
  fi
}

prepare_stage0() {
  local command=("$PY" script/prepare_runtime_config.py --stage stage0 --source-config "$MOCO_CFG" --run-dir "$RUN_DIR" --num-gpus "$NUM_GPUS")
  if [[ -n "${RESUME_STAGE0:-}" ]]; then
    command+=(--resume "$RESUME_STAGE0")
  fi
  "${command[@]}"
}

prepare_stage1() {
  local encoder label
  encoder="$(stage0_encoder_path)"
  label="$(stage1_label)"
  if [[ ! -f "$encoder" ]]; then
    echo "[Soft_Indiv_MOCOCO] missing Stage0 export: $encoder" >&2
    exit 1
  fi
  local command=("$PY" script/prepare_runtime_config.py --stage stage1 --source-config "$TFGRID_CFG" --run-dir "$RUN_DIR" --num-gpus "$NUM_GPUS" --stage-label "$label" --stage0-encoder "$encoder")
  if [[ -n "${RESUME_STAGE1:-}" ]]; then
    command+=(--resume "$RESUME_STAGE1")
  fi
  "${command[@]}"
}

run_stage0() {
  local runtime
  runtime="$(prepare_stage0)"
  echo "[Soft_Indiv_MOCOCO] ===== Stage0: Soft Teacher + individual-negative MoCo ====="
  "$PY" train_moco_encoder.py --config "$runtime" 2>&1 | tee -a "$RUN_DIR/stage0_moco/train.log"
}

run_stage1() {
  local runtime label
  runtime="$(prepare_stage1)"
  label="$(stage1_label)"
  echo "[Soft_Indiv_MOCOCO] ===== Stage1: SI-SDR + individual-source leakage ====="
  "$PY" script/run_tfgridnet.py --config "$runtime" 2>&1 | tee -a "$RUN_DIR/stage1_sweep/$label/train.log"
}

run_eval() {
  local label checkpoint out runtime
  label="$(stage1_label)"
  checkpoint="$RUN_DIR/stage1_sweep/$label/checkpoints/tfgridnet_best.ckpt"
  out="$RUN_DIR/stage1_sweep/$label/evaluation/results.json"
  runtime="$RUN_DIR/stage1_sweep/$label/config_stage1_runtime.yaml"
  if [[ ! -f "$checkpoint" ]]; then
    echo "[Soft_Indiv_MOCOCO] missing Stage1 checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ ! -f "$runtime" ]]; then
    echo "[Soft_Indiv_MOCOCO] missing Stage1 runtime config: $runtime" >&2
    exit 1
  fi
  echo "[Soft_Indiv_MOCOCO] ===== Eval: $label ====="
  "$PY" script/run_eval.py --config "$runtime" --checkpoint "$checkpoint" --out "$out" 2>&1 | tee "$RUN_DIR/stage1_sweep/$label/evaluation/eval.log"
}

ensure_run_dir
record_environment

case "$MODE" in
  all)
    run_stage0
    run_stage1
    run_eval
    ;;
  moco)
    run_stage0
    ;;
  stage1)
    run_stage1
    ;;
  eval)
    run_eval
    ;;
esac
