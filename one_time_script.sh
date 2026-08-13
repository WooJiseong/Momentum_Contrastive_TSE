#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PN_DIR="$ROOT_DIR/Running_Lab/PN_MOCOCO"

if [[ ! -d "$PN_DIR" ]]; then
  echo "[one_time] PN_MOCOCO directory not found: $PN_DIR" >&2
  exit 1
fi

cd "$PN_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pnflowtse}"
if [[ -z "${CONDA_PREFIX:-}" || "$(basename "${CONDA_PREFIX:-}")" != "$CONDA_ENV_NAME" ]]; then
  CONDA_BASE=""
  if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$("$CONDA_EXE" info --base)"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  fi
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
  else
    echo "[one_time] conda activation skipped; using current python." >&2
  fi
fi

export PY="${PY:-python}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/pn_mococo_numba_cache_${USER:-user}}"
export EXP_DATE="${EXP_DATE:-$(date +%Y%m%d)}"
export EXP_NAME="${EXP_NAME:-pn_mococo_stage2_lowlr_ft_$(date +%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29750}"
export PREPARED_RUN=1

BEST_CKPT="${BEST_CKPT:-exp/tfgridnet_mococo/checkpoints/tfgridnet_best-v1.ckpt}"
MODEL_ONLY_CKPT="${MODEL_ONLY_CKPT:-exp/finetune_init/tfgridnet_best-v1_model_only.pt}"
NUM_GPUS="${NUM_GPUS:-4}"
LOW_LR="${LOW_LR:-2.0e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"

mkdir -p "$NUMBA_CACHE_DIR" "$(dirname "$MODEL_ONLY_CKPT")"

if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[one_time] missing BEST_CKPT: $BEST_CKPT" >&2
  exit 1
fi

VISIBLE_GPU_COUNT="$("$PY" - <<'PY'
import os
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len([x for x in visible.split(",") if x.strip()]))
PY
)"
if [[ "$VISIBLE_GPU_COUNT" != "$NUM_GPUS" ]]; then
  echo "[one_time] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPU_COUNT GPU(s), but NUM_GPUS=$NUM_GPUS" >&2
  exit 1
fi

"$PY" - "$BEST_CKPT" "$MODEL_ONLY_CKPT" <<'PY'
import sys
from pathlib import Path

import torch

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

ckpt = torch.load(src, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)
raw = {
    key.removeprefix("model."): value
    for key, value in state_dict.items()
    if key.startswith("model.")
}
if not raw:
    raise RuntimeError(f"No model.* keys found in {src}")

dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(raw, dst)
print(f"[one_time] saved model-only checkpoint: {dst}")
PY

eval "$("$PY" script/prepare_runtime_config.py \
  --moco-config configs/config_moco_encoder.yaml \
  --tfgrid-config configs/config_tfgridnet_supervised.yaml \
  --mode tfgridnet)"

export RUN_DIR MOCO_CFG TFGRID_CFG

"$PY" - "$MOCO_CFG" "$TFGRID_CFG" "$MODEL_ONLY_CKPT" "$NUM_GPUS" "$LOW_LR" "$NUM_EPOCHS" "$PATIENCE" <<'PY'
import sys
from pathlib import Path

import yaml

moco_path = Path(sys.argv[1])
tfgrid_path = Path(sys.argv[2])
model_only_ckpt = sys.argv[3]
num_gpus = int(sys.argv[4])
low_lr = float(sys.argv[5])
num_epochs = int(sys.argv[6])
patience = int(sys.argv[7])

moco = yaml.safe_load(moco_path.read_text())
cfg = yaml.safe_load(tfgrid_path.read_text())

moco.setdefault("ddp", {})["num_gpus"] = num_gpus
cfg.setdefault("ddp", {})["num_gpus"] = num_gpus

paths = cfg.setdefault("paths", {})
paths["initial_model_ckpt"] = model_only_ckpt
paths["strict_initial_model"] = True
paths["encoder_override_ckpt"] = None
paths["strict_encoder_override"] = True

train = cfg.setdefault("train", {})
train["num_epochs"] = num_epochs

scheduler = cfg.setdefault("scheduler", {})
scheduler["type"] = None

optim = cfg.setdefault("optim", {})
optim["lr"] = low_lr
optim["separator_lr"] = low_lr
optim["encoder_lr"] = 1.0e-6
optim["encoder_head_lr"] = 2.0e-6

trainable = cfg.setdefault("trainable", {})
trainable["encoder"] = False
trainable["encoder_head"] = False
trainable["separator"] = True

early_stopping = cfg.setdefault("early_stopping", {})
early_stopping["enabled"] = True
early_stopping["patience"] = patience

checkpoint = cfg.setdefault("checkpoint", {})
checkpoint["ckpt_name"] = "tfgridnet_lowlr_best"
checkpoint["resume"] = None

run_dir = cfg.setdefault("experiment", {})["run_dir"]
cfg.setdefault("eval", {})["checkpoint"] = (
    f"{run_dir}/stage1_tfgridnet/checkpoints/tfgridnet_lowlr_best.ckpt"
)

moco_path.write_text(yaml.safe_dump(moco, sort_keys=False))
tfgrid_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

print(f"[one_time] RUN_DIR={run_dir}")
print(f"[one_time] MOCO_CFG={moco_path}")
print(f"[one_time] TFGRID_CFG={tfgrid_path}")
print(f"[one_time] low_lr={low_lr} num_epochs={num_epochs} num_gpus={num_gpus}")
PY

echo "[one_time] starting Stage2 low-LR fine-tuning"
echo "[one_time] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MASTER_PORT=$MASTER_PORT"

PREPARED_RUN=1 RUN_DIR="$RUN_DIR" MOCO_CFG="$MOCO_CFG" TFGRID_CFG="$TFGRID_CFG" \
  bash script/train_eval.sh tfgridnet
