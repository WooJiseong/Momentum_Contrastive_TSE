#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the latest TensorBoard scalar values.")
    parser.add_argument("--logdir", required=True, help="TensorBoard run directory containing events.out.tfevents.*")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["val_loss", "val_acc", "val_pos_sim", "val_neg_sim"],
        help="Scalar tags to print.",
    )
    args = parser.parse_args()

    logdir = Path(args.logdir)
    events = sorted(logdir.glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file found in {logdir}")

    accumulator = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    for tag in args.tags:
        if tag not in available:
            print(f"{tag}: unavailable")
            continue
        value = accumulator.Scalars(tag)[-1]
        print(f"{tag}: step={value.step}, value={value.value:.6f}")


if __name__ == "__main__":
    main()
