#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_DIR/exp/20260804_indiv_mococo_stage1_individual_nuisance"
CONFIG="$RUN_DIR/config_eval_5000.yaml"
CHECKPOINT="$RUN_DIR/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt"
EVAL_DIR="$RUN_DIR/evaluation"
OUT="$EVAL_DIR/results.json"
PY="${PYTHON_BIN:-/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python}"

mkdir -p "$EVAL_DIR"

[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }

if [[ -e "$OUT" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "output exists: $OUT (set OVERWRITE=1 to replace)" >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/../PN_MOCOCO:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/pn_indiv_mococo_numba_cache_${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "python=$PY"
    echo "config=$CONFIG"
    echo "checkpoint=$CHECKPOINT"
    echo "out=$OUT"
    echo "test_n=5000"
    "$PY" --version
} > "$EVAL_DIR/environment.txt"

cd "$PROJECT_DIR"
"$PY" script/run_eval.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    2>&1 | tee "$EVAL_DIR/eval.log"
