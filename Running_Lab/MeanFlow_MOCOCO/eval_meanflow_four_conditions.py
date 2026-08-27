#!/usr/bin/env python3
"""Evaluate a MeanFlow checkpoint under the four PN-Enroll paper conditions.

The data mixer and flow model are imported from the read-only ``sia_fm_tse``
repository.  This file only owns the experiment-specific evaluation harness.
It keeps one fixed, reproducible test stream per condition and writes both
per-item values and aggregate mean/std/95%-CI values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# librosa/numba is imported lazily by the online mixer.  Set a writable cache
# before that import so the evaluator also works outside Slurm shells.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/meanflow_mococo_eval_numba_cache")

import numpy as np
import torch
import yaml
from scipy.stats import t as student_t
from torchmetrics.functional import (
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    signal_noise_ratio,
)


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
PNFLOW_ROOT = PROJECT_ROOT.parent
SIA_REPO = PNFLOW_ROOT / "sia_fm_tse"
sys.path.insert(0, str(LAB_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SIA_REPO))
sys.path.insert(0, str(PNFLOW_ROOT))

from train_meanflow_mococo import (  # noqa: E402
    _load_sia_trainer,
    _patch_rectified_flow_goal_time,
)


CONDITIONS = (
    (2, 2, "2mix_2enroll"),
    (2, 3, "2mix_3enroll"),
    (3, 2, "3mix_2enroll"),
    (3, 3, "3mix_3enroll"),
)

# Values published by the reference README.  These are included for context;
# they do not replace the per-item evaluation performed by this script.
BASELINE_REFERENCE = {
    "snri": [12.34, 12.13, 12.32, 12.19],
    "snri_std": [3.77, 4.06, 3.59, 3.79],
    "si_snri": [11.76, 11.40, 11.22, 10.91],
    "si_snri_std": [5.56, 6.17, 5.77, 6.40],
    "si_sdri": [12.51, 12.16, 11.88, 11.62],
    "si_sdri_std": [5.04, 5.67, 5.35, 5.86],
    "pesq": [2.61, 2.58, 2.30, 2.27],
    "pesq_std": [0.42, 0.46, 0.49, 0.51],
    "stoi": [0.83, 0.83, 0.74, 0.73],
    "stoi_std": [0.15, 0.16, 0.19, 0.19],
    "dnsmos": [2.80, 2.79, 2.64, 2.62],
    "dnsmos_std": [0.33, 0.35, 0.35, 0.36],
    "wer": [0.24, 0.25, 0.38, 0.40],
    "wer_std": [0.26, 0.28, 0.30, 0.30],
}

REQUESTED_METRICS = ("snri", "si_snri", "si_sdri", "pesq", "stoi", "dnsmos", "wer")
CONDITION_LABELS = tuple(label for _, _, label in CONDITIONS)


def resolve_path(value: str | os.PathLike[str], *, relative_to: Path = PNFLOW_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (relative_to / path).resolve()


def build_dataset(
    config: dict[str, Any],
    mixture_speakers: int,
    enroll_speakers: int,
    split: str,
    *,
    mixture_seconds: int,
    enroll_seconds: int,
    snr_db_range: list[float] | None,
    neg_partial_range: list[float] | None,
):
    """Build the same online mixer as the MeanFlow/PN evaluation path.

    ``source_num`` and ``enroll_num`` are deliberately overridden separately
    for the four paper columns.  The fixed ``idx`` seeding in the mixer makes
    every condition reproducible across reruns.
    """
    from pndata.online_mixer import NoisyFlowTSEDataset

    dcfg = config["dataset"]
    sr = int(dcfg["sample_rate"])
    root_key = "test_root" if split == "test" else "val_root"
    noise_key = "test_noise" if split == "test" else "val_noise"
    root = resolve_path(dcfg[root_key], relative_to=SIA_REPO / "data")
    snr_range = list(snr_db_range if snr_db_range is not None else (dcfg.get("snr_db_range", []) or []))
    noise_dir = resolve_path(dcfg[noise_key], relative_to=SIA_REPO / "data")
    noise_dir = str(noise_dir) + "/" if snr_range else ""

    return NoisyFlowTSEDataset(
        root_dir=str(root),
        noise_dir=noise_dir,
        sample_rate=sr,
        wave_length=sr * int(mixture_seconds),
        pos_example_length=sr * int(enroll_seconds),
        neg_example_length=sr * int(enroll_seconds),
        source_num=int(mixture_speakers),
        enroll_num=int(enroll_speakers),
        active_num=tuple(dcfg.get("active_num", (-1, 1, -1))),
        snr_db_range=snr_range,
        special_spk=tuple(dcfg.get("special_spk", ()) or ()),
        partial_range=tuple(dcfg.get("partial_range", (0.33, 0.66))),
        neg_partial_range=tuple(
            neg_partial_range
            if neg_partial_range is not None
            else dcfg.get("neg_partial_range", (1.0, 1.0))
        ),
        clean_enroll=bool(dcfg.get("clean_enroll", False)),
        reproducable=True,
        filling_pattern=dcfg.get("filling_pattern", "repeat"),
        # This is the setting used by the existing apples-to-apples evaluator.
        enroll_exclude_mixture_utt=False,
    )


def _as_wave(wave: torch.Tensor) -> torch.Tensor:
    wave = wave.detach().float().cpu()
    if wave.ndim == 1:
        return wave[None]
    if wave.ndim == 2 and wave.shape[0] == 1:
        return wave
    return wave.reshape(1, -1)


def _safe_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _optional_quality_metrics(target: torch.Tensor, estimate: torch.Tensor, sr: int) -> dict[str, float | None]:
    """Calculate PESQ/STOI and optional DNSMOS/WER without making them hard deps."""
    target_np = target.numpy().reshape(-1)
    estimate_np = estimate.numpy().reshape(-1)
    result: dict[str, float | None] = {"pesq": None, "stoi": None, "dnsmos": None, "wer": None}

    try:
        from pypesq import pesq

        result["pesq"] = _safe_float(pesq(target_np, estimate_np, sr))
    except Exception:
        try:
            from pystoi import stoi

            # STOI is still useful when PESQ's backend rejects an item.
            result["stoi"] = _safe_float(stoi(target_np, estimate_np, sr, extended=False))
        except Exception:
            pass

    try:
        from pystoi import stoi

        result["stoi"] = _safe_float(stoi(target_np, estimate_np, sr, extended=False))
    except Exception:
        pass

    try:
        from speechmos import dnsmos

        scaled = estimate_np
        peak = np.max(np.abs(scaled))
        if peak > 1.0:
            scaled = scaled / peak
        result["dnsmos"] = _safe_float(dnsmos.run(scaled, sr=sr)["ovrl_mos"])
    except Exception:
        # speechmos is optional and is not installed in the current pnflowtse env.
        pass

    # WER needs a locally available ASR model and reference transcript.  Do not
    # download one implicitly inside a Slurm evaluation job.
    return result


@torch.no_grad()
def waveform_metrics(estimate: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor, sr: int):
    """Reference polarity correction plus output/input improvement metrics."""
    estimate = _as_wave(estimate)
    target = _as_wave(target)
    mixture = _as_wave(mixture)

    mse_pos = (estimate - target).square().sum(dim=-1)
    mse_neg = (-estimate - target).square().sum(dim=-1)
    flipped = mse_neg < mse_pos
    estimate = torch.where(flipped[:, None], -estimate, estimate)

    out_snr = signal_noise_ratio(estimate, target).reshape(-1)[0]
    in_snr = signal_noise_ratio(mixture, target).reshape(-1)[0]
    out_si_snr = scale_invariant_signal_noise_ratio(estimate, target).reshape(-1)[0]
    in_si_snr = scale_invariant_signal_noise_ratio(mixture, target).reshape(-1)[0]
    out_si_sdr = scale_invariant_signal_distortion_ratio(
        estimate, target, zero_mean=True
    ).reshape(-1)[0]
    in_si_sdr = scale_invariant_signal_distortion_ratio(
        mixture, target, zero_mean=True
    ).reshape(-1)[0]

    # Keep the legacy Base evaluator variant as a diagnostic for exact
    # cross-checks; the requested SISDRi column uses the scale-invariant call.
    out_sdr = signal_distortion_ratio(
        estimate, target, zero_mean=True, load_diag=False
    ).reshape(-1)[0]
    in_sdr = signal_distortion_ratio(
        mixture, target, zero_mean=True, load_diag=False
    ).reshape(-1)[0]

    quality = _optional_quality_metrics(target[0], estimate[0], sr)
    metrics = {
        "snri": _safe_float(out_snr - in_snr),
        "si_snri": _safe_float(out_si_snr - in_si_snr),
        "si_sdri": _safe_float(out_si_sdr - in_si_sdr),
        "pesq": quality["pesq"],
        "stoi": quality["stoi"],
        "dnsmos": quality["dnsmos"],
        "wer": quality["wer"],
        "legacy_sdri": _safe_float(out_sdr - in_sdr),
        "input_snr": _safe_float(in_snr),
        "input_si_snri": _safe_float(in_si_snr),
        "flipped": bool(flipped[0].item()),
    }
    return metrics


def _flow_batch(ds, indices: list[int], dcfg: dict[str, Any]):
    from utils.transforms import stft_torch

    n_fft = int(dcfg["n_fft"])
    hop = int(dcfg["hop_length"])
    win = int(dcfg["win_length"])
    scale = float(dcfg.get("stft_scale", 1.0))
    eps = 1e-8
    mix_specs, mix_ratios, pos_waves, neg_waves, sources, mixtures = [], [], [], [], [], []

    for idx in indices:
        sample, pos, neg = ds[idx]
        mixture = sample.sum(dim=0).squeeze(0).float()
        source = sample[0].squeeze(0).float()
        background = mixture - source
        source_rms = source.square().mean().sqrt().clamp_min(eps)
        background_rms = background.square().mean().sqrt().clamp_min(eps)
        ratio = (source_rms / (source_rms + background_rms)).clamp(1e-3, 1.0 - 1e-3)

        mix_specs.append((stft_torch(mixture, n_fft=n_fft, hop_length=hop, win_length=win) / scale).float())
        mix_ratios.append(ratio)
        pos_waves.append(pos.sum(dim=0).float())
        neg_waves.append(neg.sum(dim=0).float())
        sources.append(source)
        mixtures.append(mixture)

    return {
        "mixture_spec": torch.stack(mix_specs),
        "mixing_ratio": torch.stack(mix_ratios),
        "pos_wave": torch.stack(pos_waves),
        "neg_wave": torch.stack(neg_waves),
        "source": torch.stack(sources),
        "mixture": torch.stack(mixtures),
    }


def _load_tpredicter(path: str | None):
    if not path:
        return None
    from models.t_predicter import TPredicterPN

    checkpoint = torch.load(resolve_path(path), map_location="cpu")
    model_cfg = (checkpoint.get("hyper_parameters", {}) or {}).get("model", {"C": 1024})
    predictor = TPredicterPN(**model_cfg)
    state = {
        key[len("model."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    predictor.load_state_dict(state, strict=True)
    return predictor.eval().to("cuda")


@torch.no_grad()
def evaluate_condition(
    model,
    trainer_module,
    config: dict[str, Any],
    ds,
    n: int,
    sr: int,
    batch_size: int,
    t_mode: str,
    nfe: int,
    precision: str,
    tpred,
):
    from utils.transforms import istft_torch

    device = torch.device("cuda")
    dcfg = config["dataset"]
    n_fft, hop, win = int(dcfg["n_fft"]), int(dcfg["hop_length"]), int(dcfg["win_length"])
    stft_scale = float(dcfg.get("stft_scale", 1.0))
    if not math.isfinite(stft_scale) or stft_scale == 0.0:
        raise ValueError(f"dataset.stft_scale must be finite and non-zero, got {stft_scale}")
    use_bf16 = precision == "bf16"
    rows: list[dict[str, Any]] = []

    for start in range(0, n, batch_size):
        indices = list(range(start, min(start + batch_size, n)))
        batch = _flow_batch(ds, indices, dcfg)
        gpu = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        bsz = len(indices)
        ones = torch.ones(bsz, device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if t_mode == "predicted":
                enrollment_emb = model.pn_encoder.encode(gpu["pos_wave"], gpu["neg_wave"]).float()
                ratio = tpred(gpu["mixture"], enrollment_emb).reshape(bsz).float()
                ratio = ratio.clamp(1e-3, 1.0 - 1e-3)
            elif t_mode == "mean":
                ratio = gpu["mixing_ratio"].reshape(bsz).mean().expand(bsz)
            else:
                ratio = gpu["mixing_ratio"].reshape(bsz)

            enrollment = model._enrollment(gpu)
            state = gpu["mixture_spec"].clone()
            dt = (1.0 - ratio) / int(nfe)
            for step in range(int(nfe)):
                current_t = (ratio + step * dt).clamp(0.0, 1.0)
                velocity = model.model(state, current_t, ones, enrollment)
                state = state + dt.view(bsz, 1, 1) * velocity

            estimate = istft_torch(
                state.float(), n_fft=n_fft, hop_length=hop, win_length=win,
                length=gpu["source"].shape[-1],
            )
            # The STFT stream is divided by stft_scale and the UDiT endpoint is
            # source/m. Restore both factors before waveform metrics.
            restore = ratio * stft_scale
            estimate = estimate * restore.to(estimate.dtype).view(bsz, 1)

        estimate = estimate.float().cpu()
        target = batch["source"].float()
        mixture = batch["mixture"].float()
        for item, idx in enumerate(indices):
            row = waveform_metrics(estimate[item], target[item], mixture[item], sr)
            row["idx"] = idx
            row["used_t"] = _safe_float(ratio[item].cpu())
            row["true_m"] = _safe_float(batch["mixing_ratio"][item])
            rows.append(row)

        if (start // batch_size) % 20 == 0:
            print(f"[eval] {start + len(indices)}/{n}", flush=True)

    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    result: dict[str, Any] = {"count": count, "metrics": {}}
    for metric in REQUESTED_METRICS:
        values = np.asarray(
            [row[metric] for row in rows if row.get(metric) is not None], dtype=np.float64
        )
        if values.size == 0:
            result["metrics"][metric] = {
                "count": 0, "mean": None, "std": None,
                "ci95_low": None, "ci95_high": None,
            }
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        critical = float(student_t.ppf(0.975, values.size - 1)) if values.size > 1 else 0.0
        half_width = critical * std / math.sqrt(values.size)
        result["metrics"][metric] = {
            "count": int(values.size),
            "mean": mean,
            "std": std,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
        }
    return result


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_summary(out_dir: Path, metadata: dict[str, Any], summaries: dict[str, Any]) -> None:
    # The evaluator writes a checkpoint after each condition so an interrupted
    # 5k run can resume.  During that incremental write, later conditions are
    # intentionally absent from ``summaries``.  Keep the partial report valid
    # instead of indexing every condition unconditionally.
    requested_labels = tuple(metadata.get("requested_conditions", CONDITION_LABELS))
    completed_labels = [
        label for _, _, label in CONDITIONS
        if label in requested_labels and label in summaries
    ]
    condition_columns = {label: column for column, (_, _, label) in enumerate(CONDITIONS)}
    report_metadata = dict(metadata)
    report_metadata["completed_conditions"] = completed_labels
    report_metadata["complete"] = len(completed_labels) == len(requested_labels)

    payload = {"metadata": report_metadata, "conditions": summaries}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# MeanFlow four-condition evaluation",
        "",
        f"- checkpoint: `{metadata['checkpoint']}`",
        f"- split: `{metadata['split']}`",
        f"- samples per condition: `{metadata['n']}`",
        f"- inference: `NFE={metadata['nfe']}`, `t_mode={metadata['t_mode']}`, `precision={metadata['precision']}`",
        f"- completed conditions: `{len(completed_labels)}/{len(CONDITIONS)}`",
        "- CI: two-sided 95% Student-t interval from per-item values; `mean +/- std` is sample std (`ddof=1`).",
        "",
        "## MeanFlow result",
        "",
        "| Condition | Metric | n | Mean +/- Std | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in completed_labels:
        summary = summaries[label]
        for metric in REQUESTED_METRICS:
            item = summary["metrics"][metric]
            mean_std = "N/A" if item["mean"] is None else f"{item['mean']:.4f} +/- {item['std']:.4f}"
            ci = "N/A" if item["mean"] is None else f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]"
            lines.append(f"| {label} | {metric} | {item['count']} | {mean_std} | {ci} |")

    lines.extend([
        "",
        "## Published Baseline reference",
        "",
        "The values below are copied from `Base/TSE-through-Positive-Negative-Enroll/README.md`.",
        "The README provides only mean/std, so its interval is an approximate normal 95% CI using the evaluation `n`; it is not a raw-sample CI.",
        "",
        "| Condition | Metric | Published Mean +/- Std | Approx. 95% CI |",
        "|---|---:|---:|---:|",
    ])
    for label in completed_labels:
        column = condition_columns[label]
        for metric in REQUESTED_METRICS:
            if metric not in ("snri", "si_snri", "si_sdri", "pesq", "stoi", "dnsmos", "wer"):
                continue
            mean = BASELINE_REFERENCE[metric][column]
            std = BASELINE_REFERENCE[f"{metric}_std"][column]
            half = 1.96 * std / math.sqrt(max(1, int(metadata["n"])))
            lines.append(
                f"| {label} | {metric} | {mean:.2f} +/- {std:.2f} | [{mean - half:.4f}, {mean + half:.4f}] |"
            )

    lines.extend([
        "",
        "## Optional metrics",
        "",
        "- `DNSMOS`: N/A when `speechmos` is not installed.",
        "- `WER`: N/A unless a local ASR model and transcript-aligned evaluation path are explicitly configured; this job does not download ASR weights.",
        "- `legacy_sdri` is retained in each raw row for comparison with the legacy `signal_distortion_ratio` implementation.",
    ])
    (out_dir / "result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--tpred-ckpt",
        default=None,
        help="Optional t-predictor checkpoint override for evaluation only.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--t-mode", choices=("predicted", "oracle", "mean"), default="predicted")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument(
        "--mixture-seconds", type=int, default=6,
        help="Mixture duration. Baseline paper evaluation uses 6 seconds.",
    )
    parser.add_argument(
        "--enroll-seconds", type=int, default=3,
        help="Enrollment duration. Baseline paper evaluation uses 3 seconds.",
    )
    parser.add_argument(
        "--snr-db-range", type=float, nargs=2, default=None,
        metavar=("MIN", "MAX"),
        help="Override WHAM SNR range; Baseline uses -2.5 2.5.",
    )
    parser.add_argument(
        "--neg-partial-range", type=float, nargs=2, default=None,
        metavar=("MIN", "MAX"),
        help="Override negative enrollment partial range; Baseline uses 0.33 1.0.",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITION_LABELS, default=None,
        help="Evaluate only selected condition labels; useful for parallel evaluation.",
    )
    args = parser.parse_args()
    selected_labels = set(args.conditions or CONDITION_LABELS)
    selected_conditions = tuple(
        condition for condition in CONDITIONS if condition[2] in selected_labels
    )
    if not selected_conditions:
        raise ValueError("--conditions selected no evaluation conditions")

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation is configured for one CUDA GPU; CUDA is unavailable.")
    if args.n <= 0 or args.nfe <= 0:
        raise ValueError("--n and --nfe must be positive")

    config_path = resolve_path(args.config, relative_to=LAB_DIR)
    checkpoint_path = resolve_path(args.checkpoint, relative_to=PROJECT_ROOT)
    out_dir = resolve_path(args.out_dir, relative_to=PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.tpred_ckpt:
        tpred_path = resolve_path(args.tpred_ckpt, relative_to=PROJECT_ROOT)
        config.setdefault("paths", {})["t_predicter_ckpt"] = str(tpred_path)
        config.setdefault("enroll", {})["t_predicter_ckpt"] = str(tpred_path)

    print(f"[eval] loading checkpoint: {checkpoint_path}", flush=True)
    trainer_module = _load_sia_trainer()
    # Pure rectified-flow training uses r=t. The read-only reference trainer
    # applies this compatibility patch in its MOCOCO wrapper; evaluation must
    # apply the same patch or UDiT receives an unseen nonzero (r-t) embedding.
    _patch_rectified_flow_goal_time(trainer_module, config)
    model = trainer_module.LightningModule.load_from_checkpoint(
        str(checkpoint_path), config=config, strict=True, map_location="cpu"
    )
    model.eval().to("cuda")
    tpred_path = (config.get("paths", {}) or {}).get("t_predicter_ckpt")
    tpred = _load_tpredicter(tpred_path) if args.t_mode == "predicted" else None
    if args.t_mode == "predicted" and tpred is None:
        raise RuntimeError("predicted t-mode requires paths.t_predicter_ckpt")

    dcfg = config["dataset"]
    sr = int(dcfg["sample_rate"])
    metadata = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": args.split,
        "n": args.n,
        "batch_size": args.batch_size,
        "nfe": args.nfe,
        "t_mode": args.t_mode,
        "precision": args.precision,
        "mixture_seconds": args.mixture_seconds,
        "enroll_seconds": args.enroll_seconds,
        "snr_db_range": args.snr_db_range,
        "neg_partial_range": args.neg_partial_range,
        "metrics": list(REQUESTED_METRICS),
        "dnsmos_available": False,
        "wer_available": False,
        "requested_conditions": [label for _, _, label in selected_conditions],
    }
    summaries: dict[str, Any] = {}

    for mixture_speakers, enroll_speakers, label in selected_conditions:
        condition_json = raw_dir / f"{label}.json"
        if condition_json.exists():
            print(f"[eval] loading existing condition: {label}", flush=True)
            rows = json.loads(condition_json.read_text(encoding="utf-8"))
        else:
            print(f"[eval] condition={label} mixture={mixture_speakers} enroll={enroll_speakers}", flush=True)
            ds = build_dataset(
                config,
                mixture_speakers,
                enroll_speakers,
                args.split,
                mixture_seconds=args.mixture_seconds,
                enroll_seconds=args.enroll_seconds,
                snr_db_range=args.snr_db_range,
                neg_partial_range=args.neg_partial_range,
            )
            rows = evaluate_condition(
                model, trainer_module, config, ds, args.n, sr, args.batch_size,
                args.t_mode, args.nfe, args.precision, tpred,
            )
            condition_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        summaries[label] = aggregate(rows)
        write_summary(out_dir, metadata, summaries)

    write_summary(out_dir, metadata, summaries)
    print(f"[eval] completed: {out_dir / 'result_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
