"""Canonical waveform metrics shared by every contrastive_momentum Lab.

The reference PN-Enroll evaluator evaluates both output polarities and selects
the one with the lower target MSE before calculating metrics. It then reports
SNR, SI-SNR, and the TorchMetrics ``signal_distortion_ratio`` configuration
used by the reference code. This module keeps the existing ``si_sdr`` and
``snr`` keys for backward-compatible result files and adds the reference SDR
and SI-SNR variants explicitly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
from torchmetrics.functional import (
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    signal_noise_ratio,
)


REFERENCE_METRIC_KEYS = (
    "si_sdr",
    "si_sdri",
    "sdr",
    "sdri",
    "si_snr",
    "si_snri",
    "snr",
    "snri",
)


def _as_batch_waveform(wave: Tensor, name: str) -> Tensor:
    """Normalize [B,T] or [B,1,T] waveform tensors to [B,T]."""
    wave = wave.float()
    if wave.ndim == 3 and wave.shape[1] == 1:
        wave = wave[:, 0]
    elif wave.ndim != 2:
        raise ValueError(f"{name} must be [B,T] or [B,1,T], got {tuple(wave.shape)}")
    return wave


def align_output_polarity(
    estimate: Tensor,
    target: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Apply the reference evaluator's per-item +/- polarity selection.

    Returns:
        aligned estimate [B,T] and a boolean ``flipped`` mask [B].
    """
    estimate = _as_batch_waveform(estimate, "estimate")
    target = _as_batch_waveform(target, "target")
    if estimate.shape != target.shape:
        raise ValueError(
            f"estimate and target must have identical shapes, "
            f"got {tuple(estimate.shape)} and {tuple(target.shape)}"
        )

    positive_error = (estimate - target).square().sum(dim=-1)
    negative_error = (-estimate - target).square().sum(dim=-1)
    flipped = negative_error < positive_error
    aligned = torch.where(flipped[:, None], -estimate, estimate)
    return aligned, flipped


@torch.no_grad()
def reference_metric_batch(
    estimate: Tensor,
    target: Tensor,
    mixture: Tensor,
    *,
    sign_correction: bool = True,
) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
    """Calculate per-item metrics using the Base evaluator convention.

    ``si_sdr`` remains the conventional scale-invariant SDR used by existing
    result summaries. ``sdr`` follows Base exactly:
    ``signal_distortion_ratio(zero_mean=True, load_diag=False)``.
    ``si_snr`` is the Base evaluator's scale-invariant signal-to-noise ratio.

    Returns:
        ``(metrics, aligned_estimate, flipped_mask)``. Every metric value is
        a [B] tensor, while input metrics are included for reproducible gains.
    """
    estimate = _as_batch_waveform(estimate, "estimate")
    target = _as_batch_waveform(target, "target")
    mixture = _as_batch_waveform(mixture, "mixture")
    if estimate.shape != target.shape or mixture.shape != target.shape:
        raise ValueError(
            "estimate, target, and mixture must have identical [B,T] shapes: "
            f"estimate={tuple(estimate.shape)}, target={tuple(target.shape)}, "
            f"mixture={tuple(mixture.shape)}"
        )

    if sign_correction:
        estimate, flipped = align_output_polarity(estimate, target)
    else:
        flipped = torch.zeros(
            estimate.shape[0], dtype=torch.bool, device=estimate.device
        )

    out_si_sdr = scale_invariant_signal_distortion_ratio(
        estimate, target, zero_mean=True
    )
    in_si_sdr = scale_invariant_signal_distortion_ratio(
        mixture, target, zero_mean=True
    )

    # This is deliberately the exact reference call from
    # Base/TSE-through-Positive-Negative-Enroll/eval_monaural.py.
    out_sdr = signal_distortion_ratio(
        estimate, target, zero_mean=True, load_diag=False
    )
    in_sdr = signal_distortion_ratio(
        mixture, target, zero_mean=True, load_diag=False
    )

    out_si_snr = scale_invariant_signal_noise_ratio(estimate, target)
    in_si_snr = scale_invariant_signal_noise_ratio(mixture, target)
    out_snr = signal_noise_ratio(estimate, target)
    in_snr = signal_noise_ratio(mixture, target)

    metrics = {
        "si_sdr": out_si_sdr,
        "si_sdri": out_si_sdr - in_si_sdr,
        "sdr": out_sdr,
        "sdri": out_sdr - in_sdr,
        "si_snr": out_si_snr,
        "si_snri": out_si_snr - in_si_snr,
        "snr": out_snr,
        "snri": out_snr - in_snr,
        "input_si_sdr": in_si_sdr,
        "input_sdr": in_sdr,
        "input_si_snr": in_si_snr,
        "input_snr": in_snr,
    }
    return metrics, estimate, flipped


@torch.no_grad()
def reference_metrics(
    estimate: Tensor,
    target: Tensor,
    mixture: Tensor,
    *,
    sign_correction: bool = True,
) -> Dict[str, Tensor]:
    """Return batch-mean metrics for Lightning logging."""
    metrics, _, _ = reference_metric_batch(
        estimate,
        target,
        mixture,
        sign_correction=sign_correction,
    )
    return {key: value.mean() for key, value in metrics.items()}


@torch.no_grad()
def fast_training_metrics(
    estimate: Tensor,
    target: Tensor,
    mixture: Tensor,
) -> Dict[str, Tensor]:
    """Return the lightweight metrics used during optimization.

    The reference SDR requires a 512-tap double-precision Toeplitz solve for
    every waveform. That is appropriate for validation/evaluation, but running
    it on every training batch can monopolize the GPU and delay the optimizer
    step. Training retains the historical SI-SDR/SNR logging path; validation
    and evaluation continue to use ``reference_metrics``.
    """
    estimate = _as_batch_waveform(estimate, "estimate")
    target = _as_batch_waveform(target, "target")
    mixture = _as_batch_waveform(mixture, "mixture")
    if estimate.shape != target.shape or mixture.shape != target.shape:
        raise ValueError(
            "estimate, target, and mixture must have identical [B,T] shapes: "
            f"estimate={tuple(estimate.shape)}, target={tuple(target.shape)}, "
            f"mixture={tuple(mixture.shape)}"
        )

    out_si_sdr = scale_invariant_signal_distortion_ratio(
        estimate, target, zero_mean=True
    )
    in_si_sdr = scale_invariant_signal_distortion_ratio(
        mixture, target, zero_mean=True
    )
    out_snr = signal_noise_ratio(estimate, target)
    in_snr = signal_noise_ratio(mixture, target)
    return {
        "si_sdr": out_si_sdr.mean(),
        "si_sdri": (out_si_sdr - in_si_sdr).mean(),
        "snr": out_snr.mean(),
        "snri": (out_snr - in_snr).mean(),
    }


def negative_si_sdr_loss(estimate: Tensor, target: Tensor) -> Tensor:
    """Keep the existing SI-SDR training objective unchanged."""
    estimate = _as_batch_waveform(estimate, "estimate")
    target = _as_batch_waveform(target, "target")
    return -scale_invariant_signal_distortion_ratio(
        estimate, target, zero_mean=True
    ).mean()
