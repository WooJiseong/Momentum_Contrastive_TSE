from __future__ import annotations

"""Speaker metadata for the individual-negative online mixer adapter."""

import random


UNKNOWN_SPEAKER_ID = -1


def build_speaker_to_id(inners) -> dict[str, int]:
    speakers = sorted({pid for inner in inners for pid in inner.person_ids})
    return {speaker: index for index, speaker in enumerate(speakers)}


def negative_speaker_ids_for_item(
    inner,
    item_idx: int,
    speaker_to_id: dict[str, int],
) -> tuple[int, list[int]]:
    """Return metadata without changing the RNG state seen by the mixer."""
    state = random.getstate()
    try:
        if inner.reproducable:
            random.seed(item_idx)
        else:
            # The train adapter has already selected the inner and item index;
            # its __getitem__ starts directly at source/enrollment sampling.
            pass

        target_pid = (
            random.sample(inner.person_ids, 1)[0]
            if inner.reproducable
            else inner.person_ids[item_idx]
        )
        source_num = random.randint(inner.min_source_num, inner.source_num)
        enroll_num = random.randint(inner.min_enroll_num, inner.enroll_num)
        other_person_ids = list(inner.person_sound_map.keys())
        other_person_ids.remove(target_pid)
        enroll_noise_pids = random.sample(other_person_ids, enroll_num - 1)
        enroll_noise_pids.sort()
        sample_noise_pids = random.sample(other_person_ids, source_num - 1)
        negative_pids = enroll_noise_pids[inner.active_num[1] - 1 :]

        if inner.clean_enroll:
            # The clean branch still draws the target/source waveform names
            # before selecting its one unrelated negative speaker. With the
            # configured repeat filling pattern each loop consumes one draw.
            used_target_utts = set()
            for _ in range(0, inner.wave_length, inner.wave_length):
                sound_name = random.sample(inner.person_sound_map[target_pid], 1)[0][0]
                used_target_utts.add(sound_name)
            for pid in sample_noise_pids:
                for _ in range(0, inner.wave_length, inner.wave_length):
                    random.sample(inner.person_sound_map[pid], 1)

            enroll_candidates = inner.person_sound_map[target_pid]
            if inner.enroll_exclude_mixture_utt:
                filtered = [item for item in enroll_candidates if item[0] not in used_target_utts]
                if filtered:
                    enroll_candidates = filtered
            random.sample(enroll_candidates, 1)
            negative_candidates = [pid for pid in other_person_ids if pid not in sample_noise_pids]
            negative_pids = [random.sample(negative_candidates, 1)[0]]
            labels = [speaker_to_id[negative_pids[0]]]
        else:
            labels = [speaker_to_id[pid] for pid in negative_pids]
            padded_count = inner.enroll_num - inner.active_num[1]
            labels.extend([UNKNOWN_SPEAKER_ID] * (padded_count - len(labels)))
            if inner.snr_db_range:
                labels.append(UNKNOWN_SPEAKER_ID)
        target_id = speaker_to_id[target_pid]
    finally:
        random.setstate(state)
    return target_id, labels
