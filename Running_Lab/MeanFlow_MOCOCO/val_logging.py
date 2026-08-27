"""MeanFlow validation helpers backed by the repository-wide metric contract."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio
from pystoi import stoi as _stoi
from torchmetrics.functional.audio.pesq import (
    perceptual_evaluation_speech_quality as _pesq,
)

from Base.Code_Snippet.metrics_code import reference_metric_batch


_MEL = {}


def _mel_tf(device):
    key = str(device)
    if key not in _MEL:
        _MEL[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            power=2.0,
        ).to(device)
    return _MEL[key]


@torch.no_grad()
def batch_metrics(est: torch.Tensor, tgt: torch.Tensor, mix: torch.Tensor) -> dict:
    """Return Base-compatible waveform metrics plus PESQ/STOI means."""
    metrics, _, _ = reference_metric_batch(
        est,
        tgt,
        mix,
        sign_correction=False,
    )
    out = {key: value.mean() for key, value in metrics.items()}
    pesq_values = []
    stoi_values = []
    est_cpu = est.float().cpu()
    tgt_cpu = tgt.float().cpu()
    for index in range(est.shape[0]):
        try:
            pesq_values.append(
                float(_pesq(est_cpu[index], tgt_cpu[index], fs=16000, mode="nb"))
            )
        except Exception:
            pass
        try:
            stoi_values.append(
                float(_stoi(tgt_cpu[index].numpy(), est_cpu[index].numpy(), 16000, extended=False))
            )
        except Exception:
            pass
    out["pesq"] = torch.tensor(
        float(np.mean(pesq_values)) if pesq_values else float("nan"),
        device=est.device,
    )
    out["stoi"] = torch.tensor(
        float(np.mean(stoi_values)) if stoi_values else float("nan"),
        device=est.device,
    )
    return out


@torch.no_grad()
def mel_image(wave: torch.Tensor, device=None) -> torch.Tensor:
    device = device or wave.device
    mel = _mel_tf(device)(wave.to(device).float())
    logmel = torch.log(mel + 1e-6)
    lo, hi = logmel.min(), logmel.max()
    image = (logmel - lo) / (hi - lo + 1e-8)
    return torch.flip(image, dims=[0]).unsqueeze(0).cpu()


def peak_norm(wave: torch.Tensor) -> torch.Tensor:
    wave = wave.float()
    return wave / (wave.abs().max() + 1e-8)
