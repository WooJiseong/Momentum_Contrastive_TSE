"""Dataset-balanced sampling for Soft_MOCOCO.

Each configured training root is selected with equal probability, then a
speaker/item is sampled inside that root by the original online mixer. This
generalizes the old train-clean-360 + train-clean-100 behavior to any number
of roots while preserving the original mixer implementation.
"""

from __future__ import annotations

import os
import random
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.datasets import _build_inner, _ddp_rank, _noise_dir, _worker_init


def _speaker_key(scope: str, root_index: int, speaker_id: str) -> str:
    """Keep identical speaker IDs from different roots distinct."""
    return f"{scope}:root{root_index}:{speaker_id}"


class BalancedRootSpeakerDataset(Dataset):
    """Wrap the original mixer and sample dataset partitions uniformly.

    With two roots this is the 50:50 root split used by the historical
    360+100 run. A smaller root is therefore sampled more often per speaker.
    A single root is split into large/small speaker partitions before this
    class is constructed, so it also receives the same 50:50 policy.
    """

    def __init__(
        self,
        inners,
        speaker_to_id: dict[str, int],
        length: int,
        train: bool,
        scope: str,
    ):
        super().__init__()
        self.inners = list(inners)
        if not self.inners:
            raise ValueError("At least one training root is required.")
        self.speaker_to_id = speaker_to_id
        self.length = int(length)
        self.train = bool(train)
        self.scope = str(scope)

    def __len__(self) -> int:
        return self.length

    def _pick(self, idx: int):
        if self.train:
            root_index = random.randrange(len(self.inners))
            item_index = random.randrange(len(self.inners[root_index]))
            return root_index, item_index
        return 0, idx

    def _target_speaker(self, root_index: int, item_index: int) -> str:
        inner = self.inners[root_index]
        if self.train:
            # reproducable=False uses person_ids[item_index] as the target.
            return inner.person_ids[item_index]
        # reproducable=True seeds the global RNG by idx before choosing target.
        return random.Random(item_index).sample(inner.person_ids, 1)[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        root_index, item_index = self._pick(idx)
        inner = self.inners[root_index]
        target_speaker = self._target_speaker(root_index, item_index)
        sample, pos, neg = inner[item_index]
        key = _speaker_key(self.scope, root_index, target_speaker)
        return {
            "source": sample[0].squeeze(0).float(),
            "pos_wave": pos.sum(dim=0).float(),
            "neg_wave": neg.sum(dim=0).float(),
            "target_spk_id": torch.tensor(self.speaker_to_id[key], dtype=torch.long),
            "utt_id": f"{key}:{item_index}",
        }


def _log_sampling_policy(inners) -> None:
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return
    counts = [len(inner.person_ids) for inner in inners]
    total = sum(counts)
    root_prob = 1.0 / len(counts)
    factors = [total / (len(counts) * count) for count in counts]
    print(
        "[DSnOS] root_splits="
        f"{counts}, root_probability={root_prob:.6f}, "
        f"per_speaker_oversampling_factors={[round(x, 4) for x in factors]}",
        flush=True,
    )


def _restricted_inner(inner, speaker_ids):
    """Make a mixer view containing only the requested speakers."""
    view = copy.copy(inner)
    view.person_ids = list(speaker_ids)
    view.person_sound_map = {
        speaker_id: inner.person_sound_map[speaker_id] for speaker_id in view.person_ids
    }
    return view


def get_dataloaders(config, is_ddp=False, world_size=1, rank=0):
    """Build loaders with equal-probability sampling over all train roots."""
    del is_ddp, world_size, rank
    dcfg = config["dataset"]
    tcfg = config["train"]
    has_noise = bool(dcfg.get("snr_db_range", []))

    train_roots = list(
        dcfg.get(
            "train_roots",
            ["LibriSpeech/train-clean-360", "LibriSpeech/train-clean-100"],
        )
    )
    val_root = dcfg.get("val_root", "LibriSpeech/dev-clean")
    train_noise = _noise_dir(dcfg.get("train_noise", "wham_noise/tr"), has_noise)
    val_noise = _noise_dir(dcfg.get("val_noise", "wham_noise/cv"), has_noise)

    train_inners = [
        _build_inner(root, train_noise, dcfg, reproducable=False)
        for root in train_roots
    ]
    if len(train_inners) == 1:
        # Match the 360+100 speaker ratio: 251 / (921 + 251) ~= 21.4%.
        small_fraction = float(dcfg.get("dsnos_small_fraction", 251 / 1172))
        if not 0.0 < small_fraction < 1.0:
            raise ValueError("dataset.dsnos_small_fraction must be between 0 and 1.")
        speaker_ids = sorted(train_inners[0].person_ids)
        random.Random(int(dcfg.get("dsnos_split_seed", 42))).shuffle(speaker_ids)
        small_count = min(len(speaker_ids) - 1, max(1, round(len(speaker_ids) * small_fraction)))
        large_ids = speaker_ids[small_count:]
        small_ids = speaker_ids[:small_count]
        base_inner = train_inners[0]
        train_inners = [
            _restricted_inner(base_inner, large_ids),
            _restricted_inner(base_inner, small_ids),
        ]
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                "[DSnOS] one root split into large/small partitions: "
                f"{len(large_ids)}/{len(small_ids)} speakers, "
                f"small_fraction={small_fraction:.6f}, partition_probability=0.5",
                flush=True,
            )
    val_inner = _build_inner(val_root, val_noise, dcfg, reproducable=True)
    _log_sampling_policy(train_inners)

    keys = [
        _speaker_key("train", root_index, speaker_id)
        for root_index, inner in enumerate(train_inners)
        for speaker_id in inner.person_ids
    ] + [
        _speaker_key("val", 0, speaker_id)
        for speaker_id in val_inner.person_ids
    ]
    speaker_to_id = {key: index for index, key in enumerate(sorted(set(keys)))}

    train_ds = BalancedRootSpeakerDataset(
        train_inners,
        speaker_to_id,
        dcfg.get("samples_per_epoch", 129400),
        train=True,
        scope="train",
    )
    val_ds = BalancedRootSpeakerDataset(
        [val_inner],
        speaker_to_id,
        dcfg.get("val_size", 200),
        train=False,
        scope="val",
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
