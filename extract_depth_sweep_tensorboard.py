from __future__ import annotations

import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
RUNS = {
    "baseline_n3": ROOT / "Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
    "baseline_n4": ROOT / "Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
    "baseline_n6": ROOT / "Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
    "mococo_n3": ROOT / "Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
    "mococo_n4": ROOT / "Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
    "mococo_n6": ROOT / "Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
}


def read_run(name: str, path: Path):
    event = sorted(path.glob("events.out.tfevents.*"))[-1]
    acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
    acc.Reload()
    rows = []
    for tag in sorted(acc.Tags().get("scalars", [])):
        for item in acc.Scalars(tag):
            rows.append({"run": name, "tag": tag, "step": item.step, "wall_time": item.wall_time, "value": item.value})
    return rows


def main():
    out = ROOT / "exp/20260801_tensorboard_depth_sweep"
    out.mkdir(parents=True, exist_ok=True)
    rows = [row for name, path in RUNS.items() for row in read_run(name, path)]
    with open(out / "scalars.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)
    latest = {}
    for row in rows:
        key = (row["run"], row["tag"])
        if key not in latest or row["step"] >= latest[key]["step"]:
            latest[key] = row
    lines = ["# 50-epoch TensorBoard Scalar Summary", "", "Source: 20260730 TFGridNet depth sweep event files.", "", "| Run | Tag | Last step | Last value |", "|---|---|---:|---:|"]
    for (run, tag), row in sorted(latest.items()):
        lines.append(f"| {run} | {tag} | {row['step']} | {row['value']:.6f} |")
    lines += ["", "## TensorBoard", "", "Run from the repository root:", "", "```bash", "tensorboard --logdir=Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n4_fullfusion_50ep/stage1_tfgrid_sweep_mococo_mocobest_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0", "```"]
    (out / "scalars_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

