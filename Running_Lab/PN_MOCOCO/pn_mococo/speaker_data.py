from __future__ import annotations

"""PN_MOCOCO data adapter that preserves target speaker IDs.

The original online mixer already knows the target speaker, but its public
batch adapter discarded that metadata.  This adapter keeps the existing
waveform construction and adds ``target_spk_id`` for queue masking.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.datasets import _build_inner, _ddp_rank, _noise_dir, _worker_init


class SpeakerAwarePNDataset(Dataset):
    """Return the original PN-MOCOCO waves plus a stable integer speaker ID."""

    def __init__(self, inners, speaker_to_id: dict[str, int], length: int, train: bool):
        super().__init__()
        self.inners = list(inners)
        self.speaker_to_id = speaker_to_id
        self.length = int(length)
        self.train = bool(train)

    def __len__(self) -> int:
        return self.length

    def _pick(self, idx: int):
        if self.train:
            inner = random.choice(self.inners)
            item_idx = random.randint(0, len(inner) - 1)
            return inner, item_idx
        return self.inners[0], idx

    @staticmethod
    def _target_pid(inner, item_idx: int, train: bool) -> str:
        if train:
            # reproducable=False uses person_ids[idx] in NoisyFlowTSEDataset.
            return inner.person_ids[item_idx]
        # reproducable=True starts __getitem__ with random.seed(idx) followed
        # by random.sample(person_ids, 1). Keep this lookup side-effect free.
        return random.Random(item_idx).sample(inner.person_ids, 1)[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        inner, item_idx = self._pick(idx)
        target_pid = self._target_pid(inner, item_idx, self.train)
        sample, pos, neg = inner[item_idx]

        return {
            "source": sample[0].squeeze(0).float(),
            "pos_wave": pos.sum(dim=0).float(),
            "neg_wave": neg.sum(dim=0).float(),
            "target_spk_id": torch.tensor(self.speaker_to_id[target_pid], dtype=torch.long),
            "utt_id": str(item_idx),
        }


def _speaker_to_id(inners) -> dict[str, int]:
    speakers = sorted({pid for inner in inners for pid in inner.person_ids})
    return {speaker: index for index, speaker in enumerate(speakers)}


def get_speaker_aware_dataloaders(config, is_ddp=False, world_size=1, rank=0):
    """Build the PN-MOCOCO loaders while retaining target speaker metadata."""
    del is_ddp, world_size, rank  # Lightning supplies its sampler as before.

    dcfg = config["dataset"]
    tcfg = config["train"]
    has_noise = bool(dcfg.get("snr_db_range", []))

    train_roots = dcfg.get(
        "train_roots",
        ["LibriSpeech/train-clean-360", "LibriSpeech/train-clean-100"],
    )
    val_root = dcfg.get("val_root", "LibriSpeech/dev-clean")
    train_noise = _noise_dir(dcfg.get("train_noise", "wham_noise/tr"), has_noise)
    val_noise = _noise_dir(dcfg.get("val_noise", "wham_noise/cv"), has_noise)

    train_inners = [_build_inner(root, train_noise, dcfg, reproducable=False) for root in train_roots]
    val_inner = _build_inner(val_root, val_noise, dcfg, reproducable=True)
    speaker_to_id = _speaker_to_id([*train_inners, val_inner])

    train_ds = SpeakerAwarePNDataset(
        train_inners,
        speaker_to_id,
        dcfg.get("samples_per_epoch", 129400),
        train=True,
    )
    val_ds = SpeakerAwarePNDataset(
        [val_inner],
        speaker_to_id,
        dcfg.get("val_size", 200),
        train=False,
    )

    workers = int(tcfg.get("num_workers", 0))
    if workers == 0:
        base = (torch.initial_seed() + 100003 * _ddp_rank()) % (2**31 - 1)
        random.seed(base)
        np.random.seed(base)

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
    )
    return train_loader, val_loader
