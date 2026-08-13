from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.datasets import _build_inner, _noise_dir, _worker_init


class IndividualPNDataset(Dataset):
    """Minimal PN adapter preserving each negative speaker/noise row separately."""

    def __init__(self, inners, dcfg: dict, length: int, train: bool):
        self.inners = list(inners)
        self.dcfg = dcfg
        self.length = int(length)
        self.train = bool(train)

    def __len__(self) -> int:
        return self.length

    def _pick(self, idx: int):
        if self.train:
            inner = random.choice(self.inners)
            return inner[random.randint(0, len(inner) - 1)]
        return self.inners[0][idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample, pos, neg = self._pick(idx)
        pos_wave = pos.sum(dim=0).float()
        neg_waves = neg.float()
        neg_valid = neg_waves.abs().flatten(1).amax(dim=1) > 1e-8
        if neg_valid.any():
            neg_wave = neg_waves[neg_valid].sum(dim=0).float()
        else:
            neg_wave = neg_waves.sum(dim=0).float()
        return {
            "pos_wave": pos_wave,
            "neg_wave": neg_wave,
            "neg_waves": neg_waves,
            "neg_valid": neg_valid,
            "utt_id": str(idx),
        }


def get_dataloaders(config: dict):
    dcfg = config["dataset"]
    tcfg = config["train"]
    has_noise = bool(dcfg.get("snr_db_range", []))
    train_roots = dcfg.get("train_roots", ["LibriSpeech/train-clean-360", "LibriSpeech/train-clean-100"])
    val_root = dcfg.get("val_root", "LibriSpeech/dev-clean")
    train_noise = _noise_dir(dcfg.get("train_noise", "wham_noise/tr"), has_noise)
    val_noise = _noise_dir(dcfg.get("val_noise", "wham_noise/cv"), has_noise)
    train_inners = [_build_inner(root, train_noise, dcfg, reproducable=False) for root in train_roots]
    val_inner = _build_inner(val_root, val_noise, dcfg, reproducable=True)
    train_ds = IndividualPNDataset(train_inners, dcfg, dcfg.get("samples_per_epoch", 10000), True)
    val_ds = IndividualPNDataset([val_inner], dcfg, dcfg.get("val_size", 200), False)
    workers = int(tcfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=True, worker_init_fn=_worker_init,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False,
        num_workers=min(2, workers), pin_memory=True, worker_init_fn=_worker_init,
    )
    return train_loader, val_loader
