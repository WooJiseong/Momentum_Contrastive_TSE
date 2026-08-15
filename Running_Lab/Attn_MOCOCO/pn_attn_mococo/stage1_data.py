"""Stage1 data adapter that preserves each physical nuisance source."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.datasets import _build_inner, _ddp_rank, _emit_posneg, _noise_dir, _worker_init


class AttnLeakageDataset(Dataset):
    """Emit source, mixture, enrollments, and individual mixture nuisances."""

    def __init__(self, inners, length: int, train: bool, emit_posneg: bool):
        self.inners = list(inners)
        self.length = int(length)
        self.train = bool(train)
        self.emit_posneg = bool(emit_posneg)

    def __len__(self) -> int:
        return self.length

    def _pick(self, idx: int):
        if self.train:
            inner = random.choice(self.inners)
            return inner[random.randint(0, len(inner) - 1)]
        return self.inners[0][idx]

    def __getitem__(self, idx: int) -> dict:
        sample, pos, neg = self._pick(idx)
        source = sample[0].squeeze(0)
        nuisance_sources = sample[1:].squeeze(1)
        if nuisance_sources.ndim != 2 or nuisance_sources.shape[0] == 0:
            raise RuntimeError("Leakage Stage1 requires sample[1:] as [J,T] sources")
        background = nuisance_sources.sum(dim=0)
        output = {
            "source": source.float(),
            "mixture": (source + background).float(),
            "background": background.float(),
            "nuisance_sources": nuisance_sources.float(),
            "utt_id": str(idx),
        }
        if self.emit_posneg:
            output["pos_wave"] = pos.sum(dim=0).float()
            output["neg_wave"] = neg.sum(dim=0).float()
        return output


def get_attn_leakage_dataloaders(config: dict, is_ddp=False, world_size=1, rank=0):
    del is_ddp, world_size, rank
    dcfg = config["dataset"]
    tcfg = config["train"]
    has_noise = bool(dcfg.get("snr_db_range", []))
    train_roots = dcfg.get(
        "train_roots", ["LibriSpeech/train-clean-360", "LibriSpeech/train-clean-100"]
    )
    train_noise = _noise_dir(dcfg.get("train_noise", "wham_noise/tr"), has_noise)
    val_noise = _noise_dir(dcfg.get("val_noise", "wham_noise/cv"), has_noise)
    train_inners = [
        _build_inner(root, train_noise, dcfg, reproducable=False) for root in train_roots
    ]
    val_inner = _build_inner(dcfg.get("val_root", "LibriSpeech/dev-clean"), val_noise, dcfg, reproducable=True)
    emit_posneg = _emit_posneg(config)
    train_ds = AttnLeakageDataset(
        train_inners,
        length=int(dcfg.get("samples_per_epoch", 129400)),
        train=True,
        emit_posneg=emit_posneg,
    )
    val_ds = AttnLeakageDataset(
        [val_inner],
        length=int(dcfg.get("val_size", 200)),
        train=False,
        emit_posneg=emit_posneg,
    )
    num_workers = int(tcfg["num_workers"])
    if num_workers == 0:
        seed = (torch.initial_seed() + 100003 * _ddp_rank()) % (2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=min(2, num_workers),
        pin_memory=True,
        worker_init_fn=_worker_init,
        persistent_workers=False,
    )
    return train_loader, val_loader
