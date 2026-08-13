from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset

from pn_soft_indiv_mococo.paths import add_repo_paths

add_repo_paths()

from data.datasets import _build_inner, _noise_dir, _worker_init


class _OnlineMixerDataset(Dataset):
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

    @staticmethod
    def _enrollment(pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_wave = pos.sum(dim=0).float()
        neg_waves = neg.float()
        neg_valid = neg_waves.abs().flatten(1).amax(dim=1) > 1e-8
        neg_wave = neg_waves[neg_valid].sum(dim=0).float() if neg_valid.any() else neg_waves.sum(dim=0).float()
        return pos_wave, neg_wave, neg_waves, neg_valid


class SoftIndividualStage0Dataset(_OnlineMixerDataset):
    """Preserve each negative enrollment waveform for individual-negative MoCo."""

    def __getitem__(self, idx: int) -> dict:
        _, pos, neg = self._pick(idx)
        pos_wave, neg_wave, neg_waves, neg_valid = self._enrollment(pos, neg)
        return {
            "pos_wave": pos_wave,
            "neg_wave": neg_wave,
            "neg_waves": neg_waves,
            "neg_valid": neg_valid,
            "utt_id": str(idx),
        }


class SoftIndividualStage1Dataset(_OnlineMixerDataset):
    """Keep the actual per-source mixture nuisance waveforms for leakage loss."""

    def __getitem__(self, idx: int) -> dict:
        # sample is [1 + J, 1, T]: target followed by all mixture components,
        # including WHAM noise when it is enabled.
        sample, pos, neg = self._pick(idx)
        source = sample[0].squeeze(0)
        nuisance_sources = sample[1:].squeeze(1)
        if nuisance_sources.ndim != 2 or nuisance_sources.shape[0] == 0:
            raise RuntimeError("Stage1 requires sample[1:] as [J, T] individual nuisance sources.")
        background = nuisance_sources.sum(dim=0)
        pos_wave, neg_wave, _, _ = self._enrollment(pos, neg)
        return {
            "source": source.float(),
            "mixture": (source + background).float(),
            "background": background.float(),
            "nuisance_sources": nuisance_sources.float(),
            "pos_wave": pos_wave,
            "neg_wave": neg_wave,
            "utt_id": str(idx),
        }


def _inners(config: dict):
    dcfg = config["dataset"]
    has_noise = bool(dcfg.get("snr_db_range", []))
    train_roots = dcfg.get("train_roots", ["LibriSpeech/train-clean-360", "LibriSpeech/train-clean-100"])
    val_root = dcfg.get("val_root", "LibriSpeech/dev-clean")
    train_noise = _noise_dir(dcfg.get("train_noise", "wham_noise/tr"), has_noise)
    val_noise = _noise_dir(dcfg.get("val_noise", "wham_noise/cv"), has_noise)
    train = [_build_inner(root, train_noise, dcfg, reproducable=False) for root in train_roots]
    val = _build_inner(val_root, val_noise, dcfg, reproducable=True)
    return train, val


def _dataloaders(config: dict, dataset_type: type[_OnlineMixerDataset]):
    dcfg = config["dataset"]
    tcfg = config["train"]
    train_inners, val_inner = _inners(config)
    train_ds = dataset_type(train_inners, int(dcfg.get("samples_per_epoch", 10000)), train=True)
    val_ds = dataset_type([val_inner], int(dcfg.get("val_size", 200)), train=False)
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


def get_stage0_dataloaders(config: dict):
    return _dataloaders(config, SoftIndividualStage0Dataset)


def get_stage1_dataloaders(config: dict, is_ddp=False, world_size=1, rank=0):
    del is_ddp, world_size, rank
    return _dataloaders(config, SoftIndividualStage1Dataset)
