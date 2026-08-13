from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
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
TAGS = ["train_loss", "val_loss", "train_si_sdr", "val_si_sdr", "train_si_sdri", "val_si_sdri", "val_snr", "val_snri"]


def read(path: Path, tag: str):
    event = sorted(path.glob("events.out.tfevents.*"))[-1]
    acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return []
    return [(x.step, x.value) for x in acc.Scalars(tag)]


def render(tag: str, output: Path):
    width, height = 1800, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 90, 80, 1750, 790
    series = {name: read(path, tag) for name, path in RUNS.items()}
    values = [v for points in series.values() for _, v in points]
    if not values:
        return
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.08, 0.05)
    lo, hi = lo - pad, hi + pad
    xmax = max((p[-1][0] for p in series.values() if p), default=1)
    draw.text((left, 25), f"20260730 Depth Sweep - {tag}", fill="black")
    for i in range(6):
        y = top + (bottom - top) * i / 5
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        value = hi - (hi - lo) * i / 5
        draw.text((8, y - 7), f"{value:.3f}", fill="#444444")
    colors = {"baseline": "#1f77b4", "mococo": "#ff7f0e"}
    for name, points in series.items():
        if len(points) < 2:
            continue
        condition = name.split("_")[0]
        depth = name.split("_")[-1]
        coords = []
        for step, value in points:
            x = left + (right - left) * step / max(1, xmax)
            y = bottom - (bottom - top) * (value - lo) / (hi - lo)
            coords.append((x, y))
        draw.line(coords, fill=colors[condition], width=3)
        draw.text((right - 210, top + 25 * (int(depth) - 3) + (0 if condition == "baseline" else 12)), f"{condition} depth={depth}", fill=colors[condition])
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.text((right - 140, bottom + 20), f"step 0-{xmax}", fill="#444444")
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main():
    output = ROOT / "exp/20260801_tensorboard_depth_sweep/png"
    for tag in TAGS:
        render(tag, output / f"{tag}.png")


if __name__ == "__main__":
    main()

