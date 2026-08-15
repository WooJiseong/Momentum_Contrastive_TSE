#!/usr/bin/env bash
# Run Attn_MOCOCO Stage0 and the two controlled Stage1 loss variants.
# Run from Running_Lab/Attn_MOCOCO.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-all}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/../PN_MOCOCO:$PROJECT_DIR/../../..:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29990}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/attn_mococo_numba_cache_${USER:-user}}"

# Slurm batch shells do not necessarily initialize the conda `python`
# command. Prefer the project environment explicitly, then fall back to PATH.
PYTHON_BIN="${PYTHON_BIN:-${PY:-}}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python" ]]; then
    PYTHON_BIN="/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python"
  else
    PYTHON_BIN="$(command -v python || true)"
  fi
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[Attn_MOCOCO] Python interpreter not found. Set PYTHON_BIN=/path/to/python." >&2
  exit 127
fi
echo "[Attn_MOCOCO] python=$PYTHON_BIN"

EXP_DIR="exp/20260813_attn_mococo_40frame_lightweight_attention"
mkdir -p "$NUMBA_CACHE_DIR" "$EXP_DIR/stage0_moco" "$EXP_DIR/stage1_tfgridnet_si_sdr" "$EXP_DIR/stage1_tfgridnet_si_sdr_leakage"
cp -n configs/config_attn_moco.yaml "$EXP_DIR/source_config_attn_moco.yaml" 2>/dev/null || true
cp -n configs/config_tfgridnet_attn_si_sdr.yaml "$EXP_DIR/source_config_tfgridnet_attn_si_sdr.yaml" 2>/dev/null || true
cp -n configs/config_tfgridnet_attn_si_sdr_leakage.yaml "$EXP_DIR/source_config_tfgridnet_attn_si_sdr_leakage.yaml" 2>/dev/null || true

REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
if [[ ! -f "$EXP_DIR/environment.txt" ]]; then
  {
    date --iso-8601=seconds
    echo "repo=$REPO_ROOT"
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true
    "$PYTHON_BIN" --version
    "$PYTHON_BIN" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda))'
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  } > "$EXP_DIR/environment.txt"
fi
printf '%s\n' "$0 $*" > "$EXP_DIR/last_command.txt"

run_stage0() {
  "$PYTHON_BIN" train_attn_moco_encoder.py --config configs/config_attn_moco.yaml \
    2>&1 | tee "$EXP_DIR/stage0_moco/train.log"
}

run_stage1_si_sdr() {
  "$PYTHON_BIN" script/run_tfgridnet_attn.py --config configs/config_tfgridnet_attn_si_sdr.yaml \
    2>&1 | tee "$EXP_DIR/stage1_tfgridnet_si_sdr/train.log"
}

run_stage1_leakage() {
  "$PYTHON_BIN" script/run_tfgridnet_attn.py --config configs/config_tfgridnet_attn_si_sdr_leakage.yaml \
    2>&1 | tee "$EXP_DIR/stage1_tfgridnet_si_sdr_leakage/train.log"
}

case "$MODE" in
  moco) run_stage0 ;;
  si_sdr) run_stage1_si_sdr ;;
  leakage) run_stage1_leakage ;;
  stage1) run_stage1_si_sdr; run_stage1_leakage ;;
  all) run_stage0; run_stage1_si_sdr; run_stage1_leakage ;;
  *) echo "Usage: bash script/run_train.sh {moco|si_sdr|leakage|stage1|all}" >&2; exit 1 ;;
esac
