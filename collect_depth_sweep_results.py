from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path("/gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum")
MANIFEST = ROOT / "depth_sweep_manifest_20260730.yaml"
OUT_JSON = ROOT / "depth_sweep_summary_20260730.json"
OUT_MD = ROOT / "depth_sweep_summary_20260730.md"


def scalar_rows(run_root: Path) -> list[dict]:
    log_root = run_root / "stage1_tfgridnet/lightning_logs/0"
    rows: list[dict] = []
    for event_file in sorted(log_root.glob("events.out.tfevents.*")):
        ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if "val_loss" not in tags:
            continue
        vals = {
            tag: ea.Scalars(tag)
            for tag in ("val_loss", "val_si_sdr", "val_si_sdri", "val_snr", "val_snri")
            if tag in tags
        }
        n = len(vals["val_loss"])
        for idx in range(n):
            rows.append(
                {
                    "idx": len(rows),
                    "step": vals["val_loss"][idx].step,
                    "val_loss": vals["val_loss"][idx].value,
                    "val_si_sdr": vals["val_si_sdr"][idx].value,
                    "val_si_sdri": vals["val_si_sdri"][idx].value,
                    "val_snr": vals["val_snr"][idx].value,
                    "val_snri": vals["val_snri"][idx].value,
                }
            )
    return rows


def log_scalar_rows(run_root: Path) -> list[dict]:
    log_path = run_root / "train_torchrun.log"
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    train_start = text.find("Training:")
    if train_start >= 0:
        text = text[train_start:]
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+\|\s+"
        r"val_loss=(?P<val_loss>[-+0-9.]+)\s+\|\s+"
        r"val_si_sdr=(?P<val_si_sdr>[-+0-9.]+)\s+\|\s+"
        r"val_si_sdri=(?P<val_si_sdri>[-+0-9.]+)\s+\|\s+"
        r"val_snr=(?P<val_snr>[-+0-9.]+)\s+\|\s+"
        r"val_snri=(?P<val_snri>[-+0-9.]+)"
    )
    by_epoch: dict[int, dict] = {}
    for match in pattern.finditer(text):
        epoch = int(match.group("epoch"))
        by_epoch[epoch] = {
            "idx": epoch,
            "epoch": epoch,
            "step": (epoch + 1) * 313 - 1,
            "val_loss": float(match.group("val_loss")),
            "val_si_sdr": float(match.group("val_si_sdr")),
            "val_si_sdri": float(match.group("val_si_sdri")),
            "val_snr": float(match.group("val_snr")),
            "val_snri": float(match.group("val_snri")),
        }
    return [by_epoch[e] for e in sorted(by_epoch)]


def load_eval(run_root: Path) -> dict | None:
    result_path = run_root / "evaluation/results.json"
    if not result_path.exists():
        return None
    data = json.loads(result_path.read_text())
    return {
        "path": str(result_path),
        "checkpoint": data.get("checkpoint"),
        "n": data.get("n"),
        **data.get("summary", {}),
    }


def main() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    runs = []
    for run in manifest["runs"]:
        project = Path(run["project"])
        run_root = project / run["run_dir"]
        event_rows = scalar_rows(run_root)
        log_rows = log_scalar_rows(run_root)
        rows = log_rows if len(log_rows) > len(event_rows) else event_rows
        best = min(rows, key=lambda x: x["val_loss"]) if rows else None
        runs.append(
            {
                **run,
                "abs_run_dir": str(run_root),
                "n_val_event": len(event_rows),
                "n_val_log": len(log_rows),
                "n_val": len(rows),
                "best_val": best,
                "last_val": rows[-1] if rows else None,
                "eval": load_eval(run_root),
            }
        )

    summary = {"manifest": str(MANIFEST), "runs": runs}
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    by_depth: dict[int, dict[str, dict]] = {}
    for run in runs:
        by_depth.setdefault(int(run["depth"]), {})[run["condition"]] = run

    lines = [
        "# 20260730 TFGridNet Depth Sweep Summary",
        "",
        "Fixed settings: `num_epochs=50`, `fusion_layer=range(n_layers)`, 5000-item eval after train.",
        "",
        "## Validation Best",
        "",
        "| Depth | Condition | n_val | Best val SI-SDRi | Best val SI-SDR | Best step | Eval SI-SDRi | Eval SI-SDR | Eval SNRi |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in sorted(by_depth):
        for condition in ("baseline", "mococo"):
            run = by_depth[depth].get(condition)
            if not run:
                continue
            best = run["best_val"] or {}
            ev = run["eval"] or {}
            lines.append(
                "| {depth} | {condition} | {n_val} | {best_sdri} | {best_sdr} | {best_step} | {eval_sdri} | {eval_sdr} | {eval_snri} |".format(
                    depth=depth,
                    condition=condition,
                    n_val=run["n_val"],
                    best_sdri=f"{best.get('val_si_sdri', float('nan')):.6f}" if best else "",
                    best_sdr=f"{best.get('val_si_sdr', float('nan')):.6f}" if best else "",
                    best_step=best.get("step", "") if best else "",
                    eval_sdri=f"{ev.get('si_sdri', float('nan')):.6f}" if ev else "",
                    eval_sdr=f"{ev.get('si_sdr', float('nan')):.6f}" if ev else "",
                    eval_snri=f"{ev.get('snri', float('nan')):.6f}" if ev else "",
                )
            )

    lines.extend(["", "## MoCo Minus Baseline", ""])
    lines.append("| Depth | Eval SI-SDRi delta | Eval SI-SDR delta | Best val SI-SDRi delta |")
    lines.append("|---:|---:|---:|---:|")
    for depth in sorted(by_depth):
        base = by_depth[depth].get("baseline")
        moco = by_depth[depth].get("mococo")
        if not base or not moco:
            continue
        base_eval = base["eval"]
        moco_eval = moco["eval"]
        base_best = base["best_val"]
        moco_best = moco["best_val"]
        eval_sdri_delta = ""
        eval_sdr_delta = ""
        if base_eval and moco_eval:
            eval_sdri_delta = f"{moco_eval['si_sdri'] - base_eval['si_sdri']:+.6f}"
            eval_sdr_delta = f"{moco_eval['si_sdr'] - base_eval['si_sdr']:+.6f}"
        best_delta = ""
        if base_best and moco_best:
            best_delta = f"{moco_best['val_si_sdri'] - base_best['val_si_sdri']:+.6f}"
        lines.append(f"| {depth} | {eval_sdri_delta} | {eval_sdr_delta} | {best_delta} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
