"""Stage1 data adapter that keeps each physical mixture nuisance source."""
from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset

from pn_indiv_mococo.paths import add_repo_paths

add_repo_paths()

from data.datasets import _build_inner, _noise_dir, _worker_init


class IndividualLeakageStage1Dataset(Dataset):
    """Emit TFGridNet inputs and the individual waveforms used in each mixture."""

    def __init__(self, inners, length: int, train: bool):
        self.inners = list(inners)
        self.length = int(length)
        self.train = bool(train)

    def __len__(self) -> int:
        return self.length

    def _pick(self, idx: int):
        if self.train:
            inner = random.choice(self.inners)
            return inner[random.randint(0, len(inner) - 1)]
        return self.inners[0][idx]

    def __getitem__(self, idx: int) -> dict:
        # sample is [1 + J, 1, T]: target followed by all physical mixture
        # components, including WHAM noise when it is enabled.
        sample, pos, neg = self._pick(idx)
        source = sample[0].squeeze(0)
        nuisance_sources = sample[1:].squeeze(1)
        if nuisance_sources.ndim != 2 or nuisance_sources.shape[0] == 0:
            raise RuntimeError(
                "Stage1 leakage loss requires sample[1:] as [J, T] individual nuisance sources."
            )

        background = nuisance_sources.sum(dim=0)
        neg_waves = neg.float()
        neg_valid = neg_waves.abs().flatten(1).amax(dim=1) > 1e-8
        neg_wave = (
            neg_waves[neg_valid].sum(dim=0).float()
            if neg_valid.any()
            else neg_waves.sum(dim=0).float()
        )
        return {
            "source": source.float(),
            "mixture": (source + background).float(),
            "background": background.float(),
            "nuisance_sources": nuisance_sources.float(),
            "pos_wave": pos.sum(dim=0).float(),
            "neg_wave": neg_wave,
            "utt_id": str(idx),
        }


def get_stage1_dataloaders(config: dict, is_ddp=False, world_size=1, rank=0):
    """Build the standard Stage1 loaders while retaining ``nuisance_sources``."""
    del is_ddp, world_size, rank
    dcfg = config["dataset"]
    tcfg = config["train"]
    has_noise = bool(dcfg.get("snr_db_range", []))
    train_roots = dcfg.get(
        "train_roots", ["LibriSpeech/train-clean-360", "LibriSpeech/train-clean-100"]
    )
    val_root = dcfg.get("val_root", "LibriSpeech/dev-clean")
    train_noise = _noise_dir(dcfg.get("train_noise", "wham_noise/tr"), has_noise)
    val_noise = _noise_dir(dcfg.get("val_noise", "wham_noise/cv"), has_noise)
    train_inners = [
        _build_inner(root, train_noise, dcfg, reproducable=False) for root in train_roots
    ]
    val_inner = _build_inner(val_root, val_noise, dcfg, reproducable=True)
    train_ds = IndividualLeakageStage1Dataset(
        train_inners, int(dcfg.get("samples_per_epoch", 10000)), train=True
    )
    val_ds = IndividualLeakageStage1Dataset(
        [val_inner], int(dcfg.get("val_size", 200)), train=False
    )
    workers = int(tcfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=_worker_init,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=min(2, workers),
        pin_memory=True,
        worker_init_fn=_worker_init,
        persistent_workers=False,
    )
    return train_loader, val_loader
