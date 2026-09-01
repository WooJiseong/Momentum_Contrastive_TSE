"""Dataset adapter with one global speaker pool across all configured roots."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.datasets import _build_inner, _ddp_rank, _noise_dir, _worker_init


@dataclass(frozen=True)
class SpeakerRef:
    """A speaker identity together with the root that owns its audio files."""

    root_index: int
    speaker_id: str

    @property
    def key(self) -> str:
        # Namespacing also keeps the adapter correct if two roots reuse an ID.
        return f"root{self.root_index}:{self.speaker_id}"


class UnifiedNoisyFlowTSEDataset(Dataset):
    """Reuse the online mixer contract while merging speakers across roots.

    The upstream mixer stores one root per ``NoisyFlowTSEDataset`` instance.
    This class keeps those instances for file loading and noise handling, but
    samples target, interference, and enrollment speakers from one global pool.
    """

    def __init__(self, roots: list[str], noise_dir: str, dcfg: dict, reproducable: bool):
        super().__init__()
        self.reproducable = bool(reproducable)
        self.inners = [
            _build_inner(root, noise_dir, dcfg, reproducable=reproducable)
            for root in roots
        ]
        if not self.inners:
            raise ValueError("At least one dataset root is required.")

        self.speakers = [
            SpeakerRef(root_index=root_index, speaker_id=speaker_id)
            for root_index, inner in enumerate(self.inners)
            for speaker_id in inner.person_ids
        ]
        self.speakers.sort(key=lambda ref: ref.key)
        if len({ref.key for ref in self.speakers}) != len(self.speakers):
            raise RuntimeError("Unified speaker keys are not unique.")
        self._speaker_lookup = {
            ref: self.inners[ref.root_index].person_sound_map[ref.speaker_id]
            for ref in self.speakers
        }

        template = self.inners[0]
        self.wave_length = template.wave_length
        self.pos_example_length = template.pos_example_length
        self.neg_example_length = template.neg_example_length
        self.source_num = template.source_num
        self.min_source_num = template.min_source_num
        self.enroll_num = template.enroll_num
        self.min_enroll_num = template.min_enroll_num
        self.active_num = list(template.active_num)
        self.snr_db_range = list(template.snr_db_range)
        self.special_spk = list(template.special_spk)
        self.partial_range = list(template.partial_range)
        self.neg_partial_range = list(template.neg_partial_range)
        self.clean_enroll = template.clean_enroll
        self.filling_pattern = template.filling_pattern
        self.sample_rate = template.sample_rate
        self.enroll_exclude_mixture_utt = template.enroll_exclude_mixture_utt
        self.noise_dir = template.noise_dir
        self.noise_names = list(template.noise_names)

        if self.clean_enroll and len(self.speakers) < 2:
            raise ValueError("Clean enrollment needs at least two speakers.")
        if not self.clean_enroll and len(self.speakers) < self.source_num:
            raise ValueError("The global speaker pool is smaller than source_num.")

    def __len__(self) -> int:
        # Match the original PN semantics: one index per target speaker.
        return len(self.speakers)

    def target_ref_for_index(self, idx: int) -> SpeakerRef:
        if self.reproducable:
            return random.Random(idx).sample(self.speakers, 1)[0]
        return self.speakers[idx]

    def _inner_and_files(self, ref: SpeakerRef):
        return self.inners[ref.root_index], self._speaker_lookup[ref]

    def _load_person(self, ref: SpeakerRef, length: int, candidates=None):
        inner, files = self._inner_and_files(ref)
        choices = files if candidates is None else candidates
        sound_name, _ = random.sample(choices, 1)[0]
        sound, loaded_length = inner.load_and_repeat(
            os.path.join(inner.root_dir, ref.speaker_id, sound_name),
            length,
            filling_pattern=self.filling_pattern,
        )
        return sound, loaded_length, sound_name

    @staticmethod
    def _ref_sort_key(ref: SpeakerRef) -> str:
        return ref.key

    def __getitem__(self, idx: int):
        target = self.target_ref_for_index(idx)
        if self.reproducable:
            # The original validation mixer seeds the global RNG inside __getitem__.
            random.seed(idx)
            target = random.sample(self.speakers, 1)[0]

        source_num = random.randint(self.min_source_num, self.source_num)
        enroll_num = random.randint(self.min_enroll_num, self.enroll_num)
        other_speakers = [ref for ref in self.speakers if ref != target]

        enroll_noise_refs = random.sample(other_speakers, enroll_num - 1)
        enroll_noise_refs.sort(key=self._ref_sort_key)
        sample_noise_refs = random.sample(other_speakers, source_num - 1)
        split = self.active_num[1] - 1
        pos_noise_refs = sample_noise_refs[:split] + enroll_noise_refs[split:]
        neg_refs = enroll_noise_refs[split:]

        used_target_utts = set()
        sound, loaded_length, sound_name = self._load_person(
            target, self.wave_length
        )
        used_target_utts.add(sound_name)
        target_audio = [sound]
        acc_length = loaded_length
        while acc_length < self.wave_length:
            sound, loaded_length, sound_name = self._load_person(
                target, self.wave_length
            )
            used_target_utts.add(sound_name)
            acc_length += loaded_length
            target_audio.append(sound)
        sample = [torch.cat(target_audio, dim=-1)[..., : self.wave_length]]

        for ref in sample_noise_refs:
            acc_length = 0
            audios = []
            while acc_length < self.wave_length:
                sound, loaded_length, _ = self._load_person(ref, self.wave_length)
                acc_length += loaded_length
                audios.append(sound)
            sample.append(torch.cat(audios, dim=-1)[..., : self.wave_length])
        sample = torch.stack(sample)

        if self.clean_enroll:
            _, target_files = self._inner_and_files(target)
            enroll_files = [
                item for item in target_files if item[0] not in used_target_utts
            ]
            if not enroll_files:
                enroll_files = target_files
            sound, _, _ = self._load_person(
                target, self.pos_example_length, candidates=enroll_files
            )
            pos_cond = torch.stack([sound])

            neg_candidates = [
                ref for ref in other_speakers if ref not in sample_noise_refs
            ]
            neg_ref = random.sample(neg_candidates, 1)[0]
            sound, _, _ = self._load_person(neg_ref, self.neg_example_length)
            neg_cond = torch.stack([sound])
        else:
            acc_length = 0
            audios = []
            while acc_length < self.pos_example_length:
                sound, loaded_length, _ = self._load_person(
                    target, self.pos_example_length
                )
                acc_length += loaded_length
                audios.append(sound)
            pos_cond_list = [
                torch.cat(audios, dim=-1)[..., : self.pos_example_length]
            ]
            for ref in pos_noise_refs:
                acc_length = 0
                audios = []
                while acc_length < self.pos_example_length:
                    sound, loaded_length, _ = self._load_person(
                        ref, self.pos_example_length
                    )
                    acc_length += loaded_length
                    audios.append(sound)
                pos_cond_list.append(
                    torch.cat(audios, dim=-1)[..., : self.pos_example_length]
                )
            pos_cond = torch.stack(pos_cond_list)

            neg_cond_list = []
            for ref in neg_refs:
                acc_length = 0
                audios = []
                while acc_length < self.neg_example_length:
                    sound, loaded_length, _ = self._load_person(
                        ref, self.neg_example_length
                    )
                    acc_length += loaded_length
                    audios.append(sound)
                neg_cond_list.append(
                    torch.cat(audios, dim=-1)[..., : self.neg_example_length]
                )
            neg_cond = torch.stack(neg_cond_list)

        if self.special_spk and not self.clean_enroll:
            partial_pos_num = 0
            if "Partial_Pos" in self.special_spk:
                partial_pos_num = random.randint(0, enroll_num - self.active_num[1])
                for i in range(partial_pos_num):
                    active_len = int(
                        self.pos_example_length
                        * random.uniform(self.partial_range[0], self.partial_range[1])
                    )
                    active_len = min(
                        active_len, self.pos_example_length - self.sample_rate // 2
                    )
                    start = random.randint(0, self.pos_example_length - active_len)
                    end = start + active_len
                    pos_cond[self.active_num[1] + i, :, :start] = 0
                    pos_cond[self.active_num[1] + i, :, end:] = 0
                    neg_cond[i] = 0

            if "Partial_Neg" in self.special_spk:
                partial_neg_num = random.randint(
                    0, enroll_num - self.active_num[1] - partial_pos_num
                )
                for i in range(partial_neg_num):
                    active_len = int(
                        self.neg_example_length
                        * random.uniform(
                            self.neg_partial_range[0], self.neg_partial_range[1]
                        )
                    )
                    active_len = max(active_len, self.sample_rate // 2)
                    start = random.randint(0, self.neg_example_length - active_len)
                    end = start + active_len
                    neg_cond[partial_pos_num + i, :, :start] = 0
                    neg_cond[partial_pos_num + i, :, end:] = 0

        if source_num < self.source_num:
            sample = F.pad(
                sample,
                (0, 0, 0, 0, 0, self.source_num - source_num),
                mode="constant",
                value=0,
            )

        if enroll_num < self.enroll_num and not self.clean_enroll:
            pad = self.enroll_num - enroll_num
            pos_cond = F.pad(pos_cond, (0, 0, 0, 0, 0, pad), mode="constant", value=0)
            neg_cond = F.pad(neg_cond, (0, 0, 0, 0, 0, pad), mode="constant", value=0)

        if self.snr_db_range:
            sample = F.pad(sample, (0, 0, 0, 0, 0, 1), mode="constant", value=0)
            if not self.clean_enroll:
                pos_cond = F.pad(pos_cond, (0, 0, 0, 0, 0, 1), mode="constant", value=0)
                neg_cond = F.pad(neg_cond, (0, 0, 0, 0, 0, 1), mode="constant", value=0)

            noise_name = random.sample(self.noise_names, 1)[0]
            template = self.inners[0]
            noise, _ = template.load_and_repeat(
                self.noise_dir + noise_name,
                self.wave_length + self.pos_example_length + self.neg_example_length,
                remove_zero=False,
                filling_pattern="repeat",
            )
            noise_scaling_factor = template.get_noise_ratio(sample[0], noise)
            noise = noise_scaling_factor * noise
            sample[-1] += noise[:, : sample.shape[-1]]
            if not self.clean_enroll:
                pos_start = sample.shape[-1]
                neg_start = pos_start + pos_cond.shape[-1]
                pos_cond[-1] += noise[:, pos_start:neg_start]
                neg_cond[-1] += noise[:, neg_start:]

        return sample, pos_cond, neg_cond


class UnifiedSpeakerAwareDataset(Dataset):
    """Expose the same batch contract as PN_MOCOCO with global speaker IDs."""

    def __init__(self, mixer: UnifiedNoisyFlowTSEDataset, speaker_to_id: dict[str, int], length: int, train: bool):
        super().__init__()
        self.mixer = mixer
        self.speaker_to_id = speaker_to_id
        self.length = int(length)
        self.train = bool(train)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if self.train:
            mixer_idx = random.randint(0, len(self.mixer) - 1)
        else:
            mixer_idx = idx
        target_ref = self.mixer.target_ref_for_index(mixer_idx)
        sample, pos, neg = self.mixer[mixer_idx]
        return {
            "source": sample[0].squeeze(0).float(),
            "pos_wave": pos.sum(dim=0).float(),
            "neg_wave": neg.sum(dim=0).float(),
            "target_spk_id": torch.tensor(
                self.speaker_to_id[target_ref.key], dtype=torch.long
            ),
            "utt_id": f"{target_ref.key}:{mixer_idx}",
        }


def get_dataloaders(config, is_ddp=False, world_size=1, rank=0):
    """Build train/validation loaders using one global speaker pool per split."""
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

    train_mixer = UnifiedNoisyFlowTSEDataset(
        train_roots, train_noise, dcfg, reproducable=False
    )
    val_mixer = UnifiedNoisyFlowTSEDataset(
        [val_root], val_noise, dcfg, reproducable=True
    )
    all_speaker_keys = sorted(
        {ref.key for ref in train_mixer.speakers + val_mixer.speakers}
    )
    speaker_to_id = {key: index for index, key in enumerate(all_speaker_keys)}

    train_ds = UnifiedSpeakerAwareDataset(
        train_mixer,
        speaker_to_id,
        dcfg.get("samples_per_epoch", 129400),
        train=True,
    )
    val_ds = UnifiedSpeakerAwareDataset(
        val_mixer,
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
