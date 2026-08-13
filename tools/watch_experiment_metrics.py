#!/usr/bin/env python
"""Render TensorBoard validation metrics for one experiment to an in-place PNG."""

from __future__ import annotations

import argparse
import time
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
PREFERRED_TAGS = (
    "val_loss",
    "val_acc",
    "val_si_sdr",
    "val_si_sdri",
    "val_snr",
    "val_snri",
    "val_pos_sim",
    "val_neg_sim",
    "val_orthogonal_leakage_loss",
)


def resolve_experiment(experiment: str, project: str | None) -> Path:
    labs = ROOT / "Running_Lab"
    if project:
        path = labs / project / "exp" / experiment
        if not path.is_dir():
            raise FileNotFoundError(f"Experiment not found: {path}")
        return path

    candidates = sorted(labs.glob(f"*/exp/{experiment}"))
    if not candidates:
        raise FileNotFoundError(f"No experiment named '{experiment}' under {labs}")
    if len(candidates) > 1:
        names = ", ".join(str(path.relative_to(labs)) for path in candidates)
        raise ValueError(f"Experiment name is ambiguous ({names}). Pass --project.")
    return candidates[0]


def newest_event(stage_dir: Path) -> Path:
    events = sorted(
        stage_dir.glob("lightning_logs/**/events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file under {stage_dir / 'lightning_logs'}")
    return events[-1]


def resolve_event(event_arg: str) -> Path:
    event = Path(event_arg)
    if event.is_file():
        return event.resolve()
    if event.parent == Path("."):
        candidates = sorted(ROOT.rglob(event.name))
        candidates = [candidate for candidate in candidates if candidate.is_file()]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Event filename is ambiguous. Pass its full path: {event_arg}")
    raise FileNotFoundError(f"TensorBoard event file not found: {event_arg}")


def event_context(event: Path) -> tuple[Path, Path, str]:
    lightning_dir = next((parent for parent in event.parents if parent.name == "lightning_logs"), None)
    stage_dir = lightning_dir.parent if lightning_dir else event.parent
    experiment_dir = next((parent for parent in stage_dir.parents if parent.parent.name == "exp"), None)
    if experiment_dir:
        label = f"{experiment_dir.name} / {stage_dir.name}"
        return experiment_dir, stage_dir, label
    return stage_dir, stage_dir, stage_dir.name


def selected_tags(accumulator: EventAccumulator, requested_tags: list[str] | None) -> list[str]:
    available = accumulator.Tags().get("scalars", [])
    available_set = set(available)
    if requested_tags:
        return [tag for tag in requested_tags if tag in available_set and accumulator.Scalars(tag)]

    preferred = [tag for tag in PREFERRED_TAGS if tag in available_set and accumulator.Scalars(tag)]
    if preferred:
        return preferred[:4]
    return [
        tag
        for tag in available
        if tag.startswith(("val_", "val/", "validation_", "validation/")) and accumulator.Scalars(tag)
    ][:4]


def metric_event(event: Path, requested_tags: list[str] | None) -> Path:
    """Prefer a sibling event containing validation scalars over a setup-only event."""
    candidates = [event] + sorted(
        (candidate for candidate in event.parent.glob("events.out.tfevents.*") if candidate != event),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        accumulator = EventAccumulator(str(candidate), size_guidance={"scalars": 0})
        accumulator.Reload()
        if selected_tags(accumulator, requested_tags):
            return candidate
    return event


def draw_panel(draw, box: tuple[int, int, int, int], tag: str, values: list[float], font) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#8a8f98", width=1)
    draw.text((left + 10, top + 8), tag, fill="#111827", font=font)

    plot_left, plot_top = left + 56, top + 34
    plot_right, plot_bottom = right - 18, bottom - 34
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#6b7280", width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#6b7280", width=1)

    lower, upper = min(values), max(values)
    margin = max((upper - lower) * 0.08, 1e-6)
    lower -= margin
    upper += margin
    draw.text((left + 8, plot_top), f"{upper:.3f}", fill="#4b5563", font=font)
    draw.text((left + 8, plot_bottom - 10), f"{lower:.3f}", fill="#4b5563", font=font)
    draw.text((plot_left, bottom - 22), "0", fill="#4b5563", font=font)
    draw.text((plot_right - 36, bottom - 22), str(len(values) - 1), fill="#4b5563", font=font)

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


def render(event: Path, label: str, requested_tags: list[str] | None, out: Path) -> None:
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = selected_tags(accumulator, requested_tags)
    if not tags:
        raise ValueError("No requested validation scalar is available in the event file.")

    columns = 2
    rows = ceil(len(tags) / columns)
    width = 1280
    header = 72
    panel_height = 358
    image = Image.new("RGB", (width, header + rows * 378 + 12), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((28, 20), f"Validation metrics: {label}", fill="#111827", font=font)
    draw.text((28, 38), "x-axis: validation epoch", fill="#4b5563", font=font)

    panel_width = 598
    for index, tag in enumerate(tags):
        row, column = divmod(index, columns)
        left = 28 + column * 626
        top = header + row * 378
        draw_panel(
            draw,
            (left, top, left + panel_width, top + panel_height),
            tag,
            [value.value for value in accumulator.Scalars(tag)],
            font,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"event={event}")
    print(f"graph={out}")
    for tag in tags:
        latest = accumulator.Scalars(tag)[-1]
        print(f"{tag}: step={latest.step}, value={latest.value:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and optionally watch TensorBoard validation metrics.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", help="Experiment directory name under Running_Lab/*/exp/")
    source.add_argument("--event", help="TensorBoard event path, or a unique events.out.tfevents.* filename")
    parser.add_argument("--project", help="Running_Lab directory name. Required when --experiment is ambiguous.")
    parser.add_argument("--stage", default="stage0_moco", help="Stage directory under the experiment.")
    parser.add_argument("--tags", nargs="+", help="Validation scalar tags to plot. Default: up to four known tags.")
    parser.add_argument(
        "--out",
        help="Output PNG path. Default: an experiment/stage-specific filename beside the event data.",
    )
    parser.add_argument("--watch-seconds", type=float, default=0.0, help="Refresh interval. Zero renders once.")
    args = parser.parse_args()
    if args.watch_seconds < 0:
        parser.error("--watch-seconds must be zero or positive")

    if args.event:
        event = resolve_event(args.event)
        experiment_dir, stage_dir, label = event_context(event)
        stage_filename = stage_dir.name.replace("/", "_")
        default_out = stage_dir / f"{experiment_dir.name}_{stage_filename}_validation_metrics.png"
    else:
        experiment_dir = resolve_experiment(args.experiment, args.project)
        stage_dir = experiment_dir / args.stage
        if not stage_dir.is_dir():
            raise FileNotFoundError(f"Stage directory not found: {stage_dir}")
        event = metric_event(newest_event(stage_dir), args.tags)
        label = f"{experiment_dir.name} / {args.stage}"
        stage_filename = args.stage.replace("/", "_")
        default_out = stage_dir / f"{experiment_dir.name}_{stage_filename}_validation_metrics.png"
    out = Path(args.out) if args.out else default_out
    while True:
        try:
            if args.event:
                event = metric_event(resolve_event(args.event), args.tags)
            else:
                event = metric_event(newest_event(stage_dir), args.tags)
            render(event, label, args.tags, out)
        except Exception as error:
            if args.watch_seconds == 0:
                raise
            print(f"monitor_error={error}")
        if args.watch_seconds == 0:
            return
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
