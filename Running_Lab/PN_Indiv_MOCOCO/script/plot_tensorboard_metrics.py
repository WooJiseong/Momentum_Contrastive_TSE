#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    tag: str,
    values: list[float],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#8a8f98", width=1)
    draw.text((left + 10, top + 8), tag, fill="#111827", font=font)

    plot_left, plot_top = left + 56, top + 34
    plot_right, plot_bottom = right - 18, bottom - 34
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#6b7280", width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#6b7280", width=1)

    lower, upper = min(values), max(values)
    spread = upper - lower
    margin = max(spread * 0.08, 1e-6)
    lower -= margin
    upper += margin
    draw.text((left + 8, plot_top), f"{upper:.3f}", fill="#4b5563", font=font)
    draw.text((left + 8, plot_bottom - 10), f"{lower:.3f}", fill="#4b5563", font=font)
    draw.text((plot_left, bottom - 22), "0", fill="#4b5563", font=font)
    draw.text((plot_right - 32, bottom - 22), str(len(values) - 1), fill="#4b5563", font=font)

    width = max(plot_right - plot_left, 1)
    height = max(plot_bottom - plot_top, 1)
    points = []
    for index, value in enumerate(values):
        x = plot_left + width * index / max(len(values) - 1, 1)
        y = plot_bottom - height * (value - lower) / (upper - lower)
        points.append((round(x), round(y)))
    if len(points) > 1:
        draw.line(points, fill="#2563eb", width=2)
    for point in points:
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="#2563eb")
    draw.text((plot_right - 118, top + 8), f"latest {values[-1]:.4f}", fill="#111827", font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TensorBoard scalar histories to a PNG file.")
    parser.add_argument("--logdir", required=True, help="TensorBoard run directory containing events.out.tfevents.*")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["val_loss", "val_acc", "val_pos_sim", "val_neg_sim"],
        help="Scalar tags to plot.",
    )
    args = parser.parse_args()

    logdir = Path(args.logdir)
    events = sorted(logdir.glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file found in {logdir}")

    accumulator = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    tags = [tag for tag in args.tags if tag in available and accumulator.Scalars(tag)]
    if not tags:
        raise ValueError(f"None of the requested tags are available: {args.tags}")

    image = Image.new("RGB", (1280, 820), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((28, 20), "TensorBoard validation metrics", fill="#111827", font=font)
    draw.text((28, 38), "x-axis: validation epoch", fill="#4b5563", font=font)
    boxes = [
        (28, 72, 626, 430),
        (654, 72, 1252, 430),
        (28, 450, 626, 808),
        (654, 450, 1252, 808),
    ]
    for box, tag in zip(boxes, tags):
        draw_panel(draw, box, tag, [value.value for value in accumulator.Scalars(tag)], font)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
