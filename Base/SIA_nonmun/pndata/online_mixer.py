# Online LibriSpeech+WHAM TSE dataset: emits mixture + clean-target waveforms + PN pos/neg enroll.
"""Dataset for NoisyFlowTSE.

COPY-ADAPT of sia_fm_tse/pnenroll/dataset/LibriSpeech_single_emb.py, MONAURAL-ONLY and stripped:
  REMOVED: binaural path, all RIR/HRTF simulators (CIPIC/RRBRIR/ASH/CATT) + the `sofa` import,
           resemblyzer VoiceEncoder / d-vector path (return_dvec/return_clean_dvec), speed perturb.
  KEPT VERBATIM: the online mixing, per-speaker source stacking, WHAM-noise add, partial-enroll
           masking (Partial_Pos/Partial_Neg), the CLEAN_ENROLL branch, and the reproducable seeding.

The item layout (UNCHANGED from the PN dataset) is the contract train.py relies on:
  sample             : float[N, 1, NW]   row 0 = clean target; rows 1..K = interferers;
                                          last row = WHAM noise (present iff snr_db_range != []).
  pos_cond_separated : float[P, 1, NW]   positive enrollment stack (target-bearing).
  neg_cond           : float[Ng, 1, NW]  negative enrollment stack.
  => mixture wave  = sample.sum(dim=0)              float[1, NW]
  => clean target  = sample[: active_num[1]].sum(0) float[1, NW]   (active_num[1] = 1)
  => PN encode in  = pos.sum(dim=1), neg.sum(dim=1) (sum the enroll stack to mono [B,1,NW]).

NORMALIZATION (mirror reference/AD-FlowTSE/data/datasets.py gain convention): the PN dataset
does NOT apply LibriMix per-source gain rescaling (normalize=False). To keep the STFT inputs
scaled like the reference (whose mixture = mixing_ratio*source + (1-mixing_ratio)*background with
source/background each rescaled by their LibriMix gain), the WHAM noise is RMS-scaled to the target
via get_noise_ratio (snr_db sampled in snr_db_range) — identical to the reference's SNR-driven
mixing-ratio in spirit. NO global amplitude normalization is applied (matches PN stage-2). See
BLUEPRINT "REFERENCE-DERIVED SETTINGS" for the exact gain/RMS formula.

REAL  variant: source_num=3 (target + 2 interferers) + WHAM noise at the configured snr_db_range
               (e.g. [-3,3] dB in the shipped configs), partial pos/neg enrollment.
EASY  variant: source_num=2 (target + 1 interferer), NO noise (snr_db_range=[]), clean enrollment.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import librosa
import torch
from torch.utils.data import Dataset
# VAD silence trim ONLY (NOT the VoiceEncoder/d-vector path, which the blueprint removes). This is the
# proven PN dataset's `remove_zero=True` behavior (env package; preserves the training data distribution).
from resemblyzer import trim_long_silences


class NoisyFlowTSEDataset(Dataset):
    """Online target-speaker-extraction dataset (monaural).

    Args mirror the PN dataset's monaural-relevant subset. The two project variants are
    selected purely by (source_num, snr_db_range, clean_enroll, special_spk).

    Args:
        root_dir: a LibriSpeech split dir whose immediate children are speaker-id folders
                  (e.g. data/LibriSpeech/train-clean-360).
        noise_dir: a WHAM split dir WITH trailing slash (e.g. data/wham_noise/tr/); "" disables noise.
        sample_rate: 16000.
        wave_length / pos_example_length / neg_example_length: lengths in SAMPLES.
        source_num: #speech sources in the mixture (REAL=3, EASY=2). min_source_num == source_num.
        enroll_num: #enrollment speakers; tracks source_num. min_enroll_num == enroll_num.
        active_num: split tuple; only active_num[1] is indexed (the target=row0 split). Use (-1,1,-1).
        snr_db_range: WHAM SNR window in dB; [] => no noise (EASY), non-empty => add noise sampled in that
                      window (REAL; shipped configs use [-3,3]).
        special_spk / partial_range / neg_partial_range: REAL partial-enrollment masking knobs.
        clean_enroll: EASY => pos = ONE clean target utt, neg = ONE clean unrelated-speaker utt.
        reproducable: True => seed RNG by idx (val/test); False => index speakers directly (train).
        filling_pattern: "repeat" (loop short clips to length).
    """

    def __init__(
        self,
        root_dir: str,
        noise_dir: str = "",
        sample_rate: int = 16000,
        wave_length: int = 48000,
        pos_example_length: int = 48000,
        neg_example_length: int = 48000,
        source_num: int = 3,
        enroll_num: int = 3,
        active_num: Tuple[int, int, int] = (-1, 1, -1),
        snr_db_range: List[float] | None = None,
        special_spk: Tuple[str, ...] = (),
        partial_range: Tuple[float, float] = (0.33, 0.66),
        neg_partial_range: Tuple[float, float] = (1.0, 1.0),
        clean_enroll: bool = False,
        reproducable: bool = True,
        filling_pattern: str = "repeat",
        enroll_exclude_mixture_utt: bool = True,
        min_utts_per_speaker: int = 2,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.wave_length = wave_length
        self.pos_example_length = pos_example_length
        self.neg_example_length = neg_example_length
        self.source_num = source_num
        self.min_source_num = source_num
        self.enroll_num = enroll_num
        self.min_enroll_num = enroll_num
        self.active_num = list(active_num)
        self.snr_db_range = [] if snr_db_range is None else list(snr_db_range)
        self.special_spk = list(special_spk)
        self.partial_range = list(partial_range)
        self.neg_partial_range = list(neg_partial_range)
        self.clean_enroll = clean_enroll
        self.reproducable = reproducable
        self.filling_pattern = filling_pattern
        # No-overlap enrollment: draw the enrollment utterance from a DIFFERENT clip than the one used
        # in the mixture (matches MeanFlow-TSE, whose mixture2enrollment.csv curates distinct utts).
        # Requires >=2 utterances/speaker so a distinct enrollment always exists; speakers below that
        # are dropped (~0 for LibriSpeech — every speaker has many clips).
        self.enroll_exclude_mixture_utt = enroll_exclude_mixture_utt
        self.min_utts_per_speaker = max(2, int(min_utts_per_speaker)) if enroll_exclude_mixture_utt else 1

        # build {speaker_id: [(chapter_id/utt.flac, transcript), ...]} from the split tree.
        # Mirrors LibriSpeech_single_emb.py:302-321 (monaural; RIR/sofa/dvec/speed-perturb removed).
        self.person_ids: List[str] = [f for f in os.listdir(root_dir)]
        self.person_ids.sort()
        self.person_sound_map: dict = {}
        for pid in self.person_ids:
            files = []
            for chapter_id in os.listdir(root_dir + "/" + pid):
                summary = open(
                    f"{root_dir}/{pid}/{chapter_id}/{pid}-{chapter_id}.trans.txt"
                )
                files.extend(
                    [
                        (
                            f"{chapter_id}/{line.split(' ')[0]}.flac",
                            " ".join(line.split(" ")[1:]),
                        )
                        for line in summary
                    ]
                )
            files.sort()
            self.person_sound_map[pid] = files

        # Drop speakers with too few clips so a no-overlap enrollment always exists (see __init__).
        if self.min_utts_per_speaker > 1:
            kept = [p for p in self.person_ids
                    if len(self.person_sound_map[p]) >= self.min_utts_per_speaker]
            dropped = len(self.person_ids) - len(kept)
            if dropped:
                print(f"[NoisyFlowTSEDataset] dropped {dropped}/{len(self.person_ids)} speakers "
                      f"with < {self.min_utts_per_speaker} utterances (no-overlap enrollment).")
            self.person_ids = kept
            self.person_sound_map = {p: self.person_sound_map[p] for p in kept}

        # noise file list (WHAM split dir, trailing slash); "" -> no noise (EASY).
        self.noise_names: List[str] = []
        if self.noise_dir != "":
            self.noise_names = os.listdir(self.noise_dir)

    def __len__(self) -> int:
        """Number of speakers in the split (the PN dataset semantics)."""
        return len(self.person_ids)

    # ----- helpers (copied verbatim from LibriSpeech_single_emb.py) -----
    def rms(self, audio: torch.Tensor) -> torch.Tensor:
        """Root-mean-square of a waveform tensor."""
        return torch.sqrt(torch.mean(audio**2))

    def repeat_or_cut_waveform(self, waveform: torch.Tensor, desired_length: int) -> torch.Tensor:
        """Loop (repeat) or crop a waveform to exactly desired_length samples."""
        current_length = waveform.shape[-1]
        if current_length < desired_length:
            repeat_count = (desired_length + current_length - 1) // current_length
            waveform = waveform.repeat((1, repeat_count))
        cut_tensor = waveform[..., :desired_length]
        return cut_tensor

    def load_and_repeat(self, sound_name: str, length: int, remove_zero: bool = True,
                        filling_pattern: str = "repeat") -> Tuple[torch.Tensor, int]:
        """Load a flac, resample to sample_rate, optionally trim silence, fill to length. -> ([1, length], length)."""
        audio, _ = librosa.load(sound_name, sr=self.sample_rate)  # [length]
        if remove_zero:
            audio = trim_long_silences(audio)  # WebRTC-VAD trim (proven PN data distribution)
        audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        if filling_pattern == "repeat":
            audio = self.repeat_or_cut_waveform(audio, length)
            l = length
        else:
            raise NotImplementedError(filling_pattern)
        return audio, l

    def get_noise_ratio(self, wave: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """RMS scale factor for `noise` so SNR(wave, noise) falls in snr_db_range (reference-style)."""
        # LibriSpeech_single_emb.py:419-426. snr_db ~ U(snr_db_range); noise scaled to that SNR vs wave.
        snr_db = random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        snr = 10 ** (snr_db / 10)
        audio_power = torch.mean(wave**2)
        noise_power = torch.mean(noise**2)
        desired_noise_power = audio_power / snr
        noise_scaling_factor = torch.sqrt(desired_noise_power / noise_power)
        return noise_scaling_factor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one online mixture + enrollment item.

        Returns:
            sample             : float[N, 1, NW]   (target row0, interferers, optional noise row).
            pos_cond_separated : float[P, 1, NW]   positive enrollment stack.
            neg_cond           : float[Ng, 1, NW]  negative enrollment stack.
        Contract:
            - reproducable=True seeds python `random` by idx so val/test items are fixed across runs.
            - REAL: special_spk partial masking applied; WHAM noise appended (at configured snr_db_range, e.g. [-3,3]).
            - EASY (clean_enroll): pos/neg are single clean utterances; partial/noise on enroll skipped;
              snr_db_range=[] makes the mixture WHAM block a no-op (only speech sources).
        """
        # ----- pick target speaker (reproducable => seed by idx for fixed val/test) -----
        if self.reproducable:
            random.seed(idx)
            tgt_pid = random.sample(self.person_ids, 1)[0]
        else:
            tgt_pid = self.person_ids[idx]

        source_num = random.randint(self.min_source_num, self.source_num)
        enroll_num = random.randint(self.min_enroll_num, self.enroll_num)
        other_person_ids = list(self.person_sound_map.keys())
        other_person_ids.remove(tgt_pid)
        enroll_noise_pids = random.sample(other_person_ids, enroll_num - 1)
        enroll_noise_pids.sort()
        sample_noise_pids = random.sample(other_person_ids, source_num - 1)
        pos_noise_pids = (
            sample_noise_pids[: self.active_num[1] - 1]
            + enroll_noise_pids[self.active_num[1] - 1 :]
        )
        neg_pids = enroll_noise_pids[self.active_num[1] - 1 :]

        # ----- mixture: clean target row0 + (source_num-1) interferer rows -----
        used_tgt_utts = set()  # clips spent on the target's mixture row -> excluded from enrollment
        acc_l = 0
        audios = []
        while acc_l < self.wave_length:
            sound_name_, _ = random.sample(self.person_sound_map[tgt_pid], 1)[0]
            used_tgt_utts.add(sound_name_)
            sound, l = self.load_and_repeat(
                os.path.join(self.root_dir, tgt_pid + "/" + sound_name_),
                self.wave_length,
                filling_pattern=self.filling_pattern,
            )
            acc_l += l
            audios.append(sound)
        sound = torch.concat(audios, dim=-1)[..., : self.wave_length]
        sample = [sound]
        for i in sample_noise_pids:
            acc_l = 0
            audios = []
            while acc_l < self.wave_length:
                sound_name_, _ = random.sample(self.person_sound_map[i], 1)[0]
                sound, l = self.load_and_repeat(
                    os.path.join(self.root_dir, i + "/" + sound_name_),
                    self.wave_length,
                    filling_pattern=self.filling_pattern,
                )
                acc_l += l
                audios.append(sound)
            sound = torch.concat(audios, dim=-1)[..., : self.wave_length]
            sample.append(sound)
        sample = torch.stack(sample)

        # ----- enrollment (CLEAN_ENROLL=EASY vs partial=REAL) -----
        if self.clean_enroll:
            # EASY: pos = ONE clean target utt (a DIFFERENT clip than the mixture's), neg = ONE clean
            # unrelated-speaker utt; [1,1,L] each.
            enroll_cands = self.person_sound_map[tgt_pid]
            if self.enroll_exclude_mixture_utt:
                filtered = [x for x in enroll_cands if x[0] not in used_tgt_utts]
                if filtered:  # guaranteed non-empty by the >=2-utts speaker filter; fall back if not
                    enroll_cands = filtered
            enroll_name = random.sample(enroll_cands, 1)[0][0]
            sound, _ = self.load_and_repeat(
                os.path.join(self.root_dir, tgt_pid + "/" + enroll_name),
                self.pos_example_length,
                filling_pattern=self.filling_pattern,
            )
            pos_cond_separated = torch.stack([sound])

            neg_candidate_pids = [p for p in other_person_ids if p not in sample_noise_pids]
            neg_pid = random.sample(neg_candidate_pids, 1)[0]
            neg_sound, _ = self.load_and_repeat(
                os.path.join(
                    self.root_dir,
                    neg_pid + "/" + random.sample(self.person_sound_map[neg_pid], 1)[0][0],
                ),
                self.neg_example_length,
                filling_pattern=self.filling_pattern,
            )
            neg_cond = torch.stack([neg_sound])
        else:
            # REAL: pos = target utt + interferer utts; neg = the negative interferer utts.
            acc_l = 0
            audios = []
            while acc_l < self.pos_example_length:
                sound_name_, _ = random.sample(self.person_sound_map[tgt_pid], 1)[0]
                sound, l = self.load_and_repeat(
                    os.path.join(self.root_dir, tgt_pid + "/" + sound_name_),
                    self.pos_example_length,
                    filling_pattern=self.filling_pattern,
                )
                acc_l += l
                audios.append(sound)
            sound = torch.concat(audios, dim=-1)[..., : self.pos_example_length]
            pos_cond_separated = [sound]
            for i in pos_noise_pids:
                acc_l = 0
                audios = []
                while acc_l < self.pos_example_length:
                    sound_name_, _ = random.sample(self.person_sound_map[i], 1)[0]
                    sound, l = self.load_and_repeat(
                        os.path.join(self.root_dir, i + "/" + sound_name_),
                        self.pos_example_length,
                        filling_pattern=self.filling_pattern,
                    )
                    acc_l += l
                    audios.append(sound)
                sound = torch.concat(audios, dim=-1)[..., : self.pos_example_length]
                pos_cond_separated.append(sound)
            pos_cond_separated = torch.stack(pos_cond_separated)

            neg_cond = []
            for i in neg_pids:
                acc_l = 0
                negs = []
                while acc_l < self.neg_example_length:
                    sound_name_, _ = random.sample(self.person_sound_map[i], 1)[0]
                    sound, l = self.load_and_repeat(
                        os.path.join(self.root_dir, i + "/" + sound_name_),
                        self.neg_example_length,
                        filling_pattern=self.filling_pattern,
                    )
                    acc_l += l
                    negs.append(sound)
                sound = torch.concat(negs, dim=-1)[..., : self.neg_example_length]
                neg_cond.append(sound)
            neg_cond = torch.stack(neg_cond)

        # ----- REAL partial masking (skipped for clean_enroll: [1,1,L] would index out of range) -----
        if len(self.special_spk) != 0 and not self.clean_enroll:
            partial_pos_num = 0
            if "Partial_Pos" in self.special_spk:
                partial_pos_num = random.randint(0, enroll_num - self.active_num[1])
                for i in range(partial_pos_num):
                    active_len = int(
                        self.pos_example_length
                        * random.uniform(self.partial_range[0], self.partial_range[1])
                    )
                    active_len = int(
                        min(active_len, self.pos_example_length - self.sample_rate // 2)
                    )
                    start = random.randint(0, self.pos_example_length - active_len)
                    end = start + active_len
                    pos_cond_separated[self.active_num[1] + i, :, :start] = 0
                    pos_cond_separated[self.active_num[1] + i, :, end:] = 0
                    neg_cond[i] = 0

            if "Partial_Neg" in self.special_spk:
                partial_neg_num = random.randint(
                    0, enroll_num - self.active_num[1] - partial_pos_num
                )
                for i in range(partial_neg_num):
                    active_len = int(
                        self.neg_example_length
                        * random.uniform(self.neg_partial_range[0], self.neg_partial_range[1])
                    )
                    active_len = int(max(active_len, self.sample_rate // 2))
                    start = random.randint(0, self.neg_example_length - active_len)
                    end = start + active_len
                    neg_cond[partial_pos_num + i, :, :start] = 0
                    neg_cond[partial_pos_num + i, :, end:] = 0

        # ----- pad short mixtures up to source_num rows -----
        if source_num < self.source_num:
            sample = torch.nn.functional.pad(
                sample, (0, 0, 0, 0, 0, self.source_num - source_num), mode="constant", value=0
            )

        # ----- pad short enrollments up to enroll_num (skipped for clean_enroll: stays [1,1,L]) -----
        if enroll_num < self.enroll_num and not self.clean_enroll:
            pos_cond_separated = torch.nn.functional.pad(
                pos_cond_separated, (0, 0, 0, 0, 0, self.enroll_num - enroll_num),
                mode="constant", value=0,
            )
            neg_cond = torch.nn.functional.pad(
                neg_cond, (0, 0, 0, 0, 0, self.enroll_num - enroll_num),
                mode="constant", value=0,
            )

        # ----- WHAM noise: always added as a row of the MIXTURE; enrollment stays clean if clean_enroll -----
        if len(self.snr_db_range) > 0:
            sample = torch.nn.functional.pad(sample, (0, 0, 0, 0, 0, 1), mode="constant", value=0)
            if not self.clean_enroll:
                pos_cond_separated = torch.nn.functional.pad(
                    pos_cond_separated, (0, 0, 0, 0, 0, 1), mode="constant", value=0
                )
                neg_cond = torch.nn.functional.pad(
                    neg_cond, (0, 0, 0, 0, 0, 1), mode="constant", value=0
                )

            noise_name = random.sample(self.noise_names, 1)[0]
            noise, _ = self.load_and_repeat(
                self.noise_dir + noise_name,
                self.wave_length + self.pos_example_length + self.neg_example_length,
                remove_zero=False,
                filling_pattern="repeat",
            )
            noise_scaling_factor = self.get_noise_ratio(sample[0], noise)
            noise = noise_scaling_factor * noise

            sample[-1] += noise[:, : sample.shape[-1]]
            if not self.clean_enroll:
                pos_cond_separated[-1] += noise[
                    :, sample.shape[-1] : (sample.shape[-1] + pos_cond_separated.shape[-1])
                ]
                neg_cond[-1] += noise[
                    :, (sample.shape[-1] + pos_cond_separated.shape[-1]) :
                ]

        return sample, pos_cond_separated, neg_cond
