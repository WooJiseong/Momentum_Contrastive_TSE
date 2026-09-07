"""Validation metrics + TensorBoard visualization helpers for PNNoisyFlowTSE.

- batch_metrics: SI-SDR, SI-SDRi (== SI-SNRi), PESQ(nb), STOI on (est, target, mixture) waveforms,
  matching the metrics the PN-Enroll baseline reports so the flow's val curves are directly comparable.
- mel_image: a log-mel spectrogram rendered as a [1, n_mels, T] image in [0,1] for SummaryWriter.add_image.
Self-contained: torchaudio + torchmetrics + pystoi (all in the env).
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as _si_sdr
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as _pesq
from pystoi import stoi as _stoi

_MEL = {}


def _mel_tf(device):
    key = str(device)
    if key not in _MEL:
        _MEL[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80, power=2.0
        ).to(device)
    return _MEL[key]


@torch.no_grad()
def batch_metrics(est: torch.Tensor, tgt: torch.Tensor, mix: torch.Tensor) -> dict:
    """est/tgt/mix: [B, NW] waveforms. Returns scalar tensors: si_sdr, si_sdri, pesq, stoi (means).
    PESQ = narrowband, STOI = standard (extended=False) to match pnenroll's eval-metric definitions."""
    est, tgt, mix = est.float(), tgt.float(), mix.float()
    si = _si_sdr(est, tgt, zero_mean=True)            # [B]
    si_in = _si_sdr(mix, tgt, zero_mean=True)          # [B]
    out = {"si_sdr": si.mean(), "si_sdri": (si - si_in).mean()}
    pq, st = [], []
    ec, tc = est.cpu(), tgt.cpu()
    for b in range(est.shape[0]):
        try:
            pq.append(float(_pesq(est[b], tgt[b], fs=16000, mode="nb")))   # narrowband PESQ (== pnenroll eval)
        except Exception:
            pass
        try:
            st.append(float(_stoi(tc[b].numpy(), ec[b].numpy(), 16000, extended=False)))  # standard STOI (== pnenroll)
        except Exception:
            pass
    out["pesq"] = torch.tensor(float(np.mean(pq)) if pq else float("nan"))
    out["stoi"] = torch.tensor(float(np.mean(st)) if st else float("nan"))
    return out


@torch.no_grad()
def mel_image(wave: torch.Tensor, device=None) -> torch.Tensor:
    """wave: [NW] -> [1, n_mels, T] log-mel image in [0,1] (low freq at bottom) for TB add_image."""
    device = device or wave.device
    mel = _mel_tf(device)(wave.to(device).float())     # [n_mels, T]
    logmel = torch.log(mel + 1e-6)
    lo, hi = logmel.min(), logmel.max()
    img = (logmel - lo) / (hi - lo + 1e-8)
    return torch.flip(img, dims=[0]).unsqueeze(0).cpu()  # [1, n_mels, T]


def peak_norm(wave: torch.Tensor) -> torch.Tensor:
    """Peak-normalize a waveform to [-1, 1] for TB add_audio."""
    wave = wave.float()
    return wave / (wave.abs().max() + 1e-8)
