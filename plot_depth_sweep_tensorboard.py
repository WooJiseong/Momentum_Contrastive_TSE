from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
RUNS = {
    "baseline": {
        3: ROOT / "Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
        4: ROOT / "Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
        6: ROOT / "Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
    },
    "mococo": {
        3: ROOT / "Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
        4: ROOT / "Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
        6: ROOT / "Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0",
    },
}
TAGS = ["train_loss", "val_loss", "train_si_sdr", "val_si_sdr", "train_si_sdri", "val_si_sdri", "val_snr", "val_snri"]


def load_scalars(log_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file in {log_dir}")
    acc = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    return {
        tag: ([x.step for x in acc.Scalars(tag)], [x.value for x in acc.Scalars(tag)])
        for tag in TAGS if tag in available
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="exp/20260801_tensorboard_depth_sweep")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    data = {(condition, depth): load_scalars(path) for condition, depths in RUNS.items() for depth, path in depths.items()}

    for tag in TAGS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
        for ax, depth in zip(axes, (3, 4, 6)):
            plotted = False
            for condition, color in (("baseline", "tab:blue"), ("mococo", "tab:orange")):
                values = data[(condition, depth)].get(tag)
                if values:
                    ax.plot(values[0], values[1], label=condition, color=color, linewidth=1.8)
                    plotted = True
            ax.set_title(f"TFGridNet depth={depth}")
            ax.set_xlabel("optimizer step")
            ax.grid(alpha=0.25)
            if plotted:
                ax.legend()
        fig.suptitle(tag)
        fig.tight_layout()
        fig.savefig(out / f"{tag}.png", dpi=160)
        plt.close(fig)

    (out / "README.md").write_text(
        "# 20260730 Depth Sweep TensorBoard Summary\n\n"
        "Original event files are compared for Baseline and PN_MOCOCO at decoder depths 3, 4, and 6.\n\n"
        "Run TensorBoard from the contrastive_momentum root:\n\n"
        "```bash\n"
        "tensorboard --logdir=Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0\n"
        "```\n\n"
        "Generated PNGs contain the same scalar curves in a compact depth-by-depth comparison.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

