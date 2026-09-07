# =====================================================================================
# [이 파일이 하는 일 — 친절한 한국어 설명]
# 이 파일은 "데이터 공급소(adapter)"예요. LibriSpeech(영어 음성 모음)과 WHAM(잡음 모음)을
# 그 자리에서(online) 섞어서, flow 모델이 그대로 먹을 수 있는 batch(한 번에 학습에 쓰는 묶음)를
# 만들어 줍니다. 즉 "원본 음성 + 잡음 → 섞은 소리 + 정답 음성 + enrollment(목표 화자 참고 음성)"
# 를 한 덩어리로 포장하는 곳이에요.
#
# ⚠️ 데이터 경로가 막히거나(파일을 못 찾음) 데이터 모양이 이상하면 *1순위로* 이 파일을 보세요!
#    - 경로 규칙: 아래 _abs() 함수가 핵심. config에 짧게 적은 경로(예: "LibriSpeech/dev-clean")는
#      전부 이 data/ 폴더를 기준(base)으로 풀립니다. (절대경로면 그대로 사용)
#    - samples_per_epoch: "1 epoch(전체 데이터 1바퀴)을 몇 개의 섞은 소리로 칠지" 정하는 가상 길이.
#      실제 파일 개수가 아니라, 우리가 정한 '한 바퀴의 크기'예요 (아래 get_dataloaders 참고).
#
# ✏️ 바꿔도 되는 곳: config의 dataset 항목(경로, sample_rate, segment, snr_db_range,
#    samples_per_epoch, val_size 등). 이런 값들은 yaml에서 조절하세요.
# ⚠️ 만지지 마세요: convex anchor(볼록 결합 기준점) 수식, m 계산, STFT 스케일 처리.
#    여기 숫자/로직을 바꾸면 z(m)==mix(섞은 소리) 가 깨져서 학습/평가가 전부 어긋납니다.
# =====================================================================================
"""data/datasets.py — THE ADAPTER (heart of PNNoisyFlowTSE V1).

Presents MeanFlow-TSE's data API (``get_dataloaders`` + a Dataset whose ``__getitem__`` returns the
exact MeanFlow batch dict), but synthesizes every key from OUR online LibriSpeech+WHAM mixer
(``pndata.online_mixer.NoisyFlowTSEDataset``). The reference ``LibriMix``/asteroid reader is GONE —
we have no Libri2Mix wav tree or gain CSVs, only raw LibriSpeech + raw WHAM. With this file in place,
``train_meanflow.py`` reads its batch keys unchanged.

THE MIXING-RATIO / CONVEX-ANCHOR FIX (the load-bearing data-contract detail)
---------------------------------------------------------------------------
MeanFlow trains on the straight path ``z(t) = (1-t)*background_rescaled + t*source_rescaled`` and, at
validation/eval, STARTS the ODE from ``mixture_spec`` at ``t = mixing_ratio``. For that start to sit
ON the path we MUST have ``z(mixing_ratio) == mixture_spec``. Our physical mixture is a plain sum
``mix = clean + background`` (background = interferers + WHAM), which is NOT on the (background, clean)
convex segment for any single ``t`` with un-rescaled signals. So, exactly like MeanFlow's own gain
rescale, we pick a per-item ``m`` and rescale:

    m               = clean_rms / (clean_rms + background_rms)          # in (0, 1)
    source_rescaled = clean      / m
    background_rescaled = background / (1 - m)
    =>  z(m) = (1-m)*background_rescaled + m*source_rescaled
            = (1-m)*background/(1-m) + m*clean/m
            = background + clean = mix                                   # EXACT physical mixture

so ``mixture_spec = STFT(mix)`` lies exactly at ``t = m`` (proved per item; smoke asserts allclose).
This choice of ``m`` ALSO balances the two rescaled signals to a common RMS = clean_rms+background_rms,
keeping the STFT inputs well-scaled. Crucially the ANCHOR IS THE REAL MIXTURE (we do NOT re-randomize
the SNR into a synthetic convex mixture), so "our mixture" is preserved — only the (clean, background)
split is rescaled, which SI-SDR (scale-invariant) is blind to at the metric.

NORMALIZATION: one global ``stft_scale`` (default 1.0 = faithful to MeanFlow, which applies none) is
divided into EVERY spectrogram identically, so the velocity ``v = source_rescaled_spec -
background_rescaled_spec`` and the anchor identity are preserved. Tune it only if the smoke shows the
STFT std drifting far from O(1). Waveform passthrough keys stay UNSCALED (eval iSTFT/metrics use them).

ENROLLMENT: V1 (``enroll.provider = v1_raw``) uses ONE clean target utterance (MeanFlow-native) — the
mixer's ``clean_enroll`` branch already yields exactly that as ``pos[0]``. For V2 the adapter also
emits pos/neg waveforms+specs (consumed by the EnrollmentConditioner / frozen PN encoder upstream).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pndata.online_mixer import NoisyFlowTSEDataset
from utils.transforms import stft_torch

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # the data/ dir (LibriSpeech/wham symlinks live here)
_EPS = 1e-8


def _abs(rel: str) -> str:
    """Resolve a path relative to this data/ dir (so configs can stay short, e.g. 'LibriSpeech/dev-clean')."""
    # [경로 규칙 — 가장 자주 사고 나는 부분]
    # config에 적은 짧은 경로(예: "LibriSpeech/dev-clean")는 *항상 이 data/ 폴더 기준*으로 바뀝니다.
    # 즉 "LibriSpeech/dev-clean" → ".../PNFlowTSE/data/LibriSpeech/dev-clean".
    # 이미 절대경로("/로 시작")면 그대로 둡니다. 데이터를 못 찾으면 여기서 만들어진 최종 경로부터 확인!
    return rel if os.path.isabs(rel) else os.path.join(_DATA_DIR, rel)


def _build_inner(root_dir: str, noise_dir: str, dcfg: dict, reproducable: bool) -> NoisyFlowTSEDataset:
    """Instantiate OUR online mixer for one LibriSpeech split. ``noise_dir=''`` disables WHAM."""
    sr = int(dcfg["sample_rate"])
    return NoisyFlowTSEDataset(
        root_dir=_abs(root_dir),
        noise_dir=noise_dir,  # "" -> no noise (clean variant); otherwise an absolute dir WITH trailing slash
        sample_rate=sr,
        wave_length=sr * int(dcfg["segment"]),
        pos_example_length=sr * int(dcfg["segment_aux"]),
        neg_example_length=sr * int(dcfg["segment_aux"]),
        source_num=int(dcfg["source_num"]),
        enroll_num=int(dcfg["source_num"]),
        active_num=tuple(dcfg.get("active_num", (-1, 1, -1))),
        snr_db_range=list(dcfg.get("snr_db_range", []) or []),
        special_spk=tuple(dcfg.get("special_spk", ()) or ()),  # () for V1 clean enrollment (no partial masking)
        partial_range=tuple(dcfg.get("partial_range", (0.33, 0.66))),
        neg_partial_range=tuple(dcfg.get("neg_partial_range", (1.0, 1.0))),
        clean_enroll=bool(dcfg.get("clean_enroll", True)),
        reproducable=reproducable,
        filling_pattern=dcfg.get("filling_pattern", "repeat"),
        enroll_exclude_mixture_utt=bool(dcfg.get("enroll_exclude_mixture_utt", True)),
        min_utts_per_speaker=int(dcfg.get("min_utts_per_speaker", 2)),
    )


def _noise_dir(rel_or_empty: str, has_noise: bool) -> str:
    """Return an absolute WHAM dir WITH trailing slash, or '' when the variant has no noise."""
    if not has_noise:
        return ""
    p = _abs(rel_or_empty)
    return p if p.endswith("/") else p + "/"


def _ddp_rank() -> int:
    """지금 프로세스의 DDP rank (분산 아니면 0). ⚠️ 왜 필요한가: seed_everything 은 모든 rank 에
    '같은' 시드를 주고, 학습 모드 샘플링은 sampler 의 idx 를 무시하고 전역 RNG 로 뽑기 때문에,
    rank 성분이 없으면 멀티 GPU 학습 시 모든 rank 가 '완전히 똑같은 배치'를 합성합니다(실증 확인)."""
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is not None and v.isdigit():
            return int(v)
    return 0


def _worker_init(worker_id: int) -> None:
    """DataLoader 워커 '그리고' DDP rank 별로 python/numpy RNG 를 분리
    (학습 샘플링이 전역 RNG 를 쓰므로 두 차원 모두 달라야 함)."""
    base = (torch.initial_seed() + worker_id + 100003 * _ddp_rank()) % (2 ** 31 - 1)
    random.seed(base)
    np.random.seed(base)


class PNLibriMixInformed(Dataset):
    """Wrap OUR online mixer and emit the EXACT MeanFlow batch dict (see module docstring for the math).

    Args:
        inners: list of NoisyFlowTSEDataset (train may span train-clean-360 + train-clean-100).
        dcfg: the config['dataset'] block (stft params, stft_scale, ...).
        length: virtual epoch length (train) or #fixed items (val/test).
        train: True -> sample a random inner + random speaker each call (reproducable=False inners);
               False -> deterministic ``inners[0][idx]`` (reproducable=True seeds by idx -> fixed items).
        emit_posneg: also emit pos/neg waveforms+specs (V2a/V2b). V1 leaves them out.
    """

    def __init__(self, inners, dcfg: dict, length: int, train: bool, emit_posneg: bool = False):
        super().__init__()
        self.inners = list(inners)
        self.length = int(length)
        self.train = bool(train)
        self.n_fft = int(dcfg["n_fft"])
        self.hop = int(dcfg["hop_length"])
        self.win = int(dcfg["win_length"])
        self.stft_scale = float(dcfg.get("stft_scale", 1.0))
        self.emit_posneg = bool(emit_posneg)

    def __len__(self) -> int:
        return self.length

    def _pick(self, idx: int):
        if self.train:
            inner = random.choice(self.inners)
            return inner[random.randint(0, len(inner) - 1)]  # reproducable=False -> fresh random mixture
        return self.inners[0][idx]  # reproducable=True -> idx seeds a fixed, distinct item

    def _S(self, wave: torch.Tensor) -> torch.Tensor:
        """STFT (real|imag -> 512 ch) of a 1-D waveform, divided by the single global stft_scale."""
        return (stft_torch(wave, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win)
                / self.stft_scale).float()

    def __getitem__(self, idx: int) -> dict:
        sample, pos, neg = self._pick(idx)  # sample[N,1,NW], pos[P,1,NW], neg[Ng,1,NW]
        mix = sample.sum(dim=0).squeeze(0)              # [NW]  clean + interferers + (WHAM)
        clean = sample[0].squeeze(0)                    # [NW]  clean target (active_num[1]=1 -> row 0)
        nuisance_sources = sample[1:].squeeze(1)         # [J, NW] individual interferers + WHAM
        background = mix - clean                        # [NW]  interferers + WHAM (== mix - clean)

        # --- convex anchor: m balances rescaled amplitudes and makes z(m) == mix exactly ---
        clean_rms = torch.sqrt(torch.mean(clean ** 2)).clamp_min(_EPS)
        bg_rms = torch.sqrt(torch.mean(background ** 2)).clamp_min(_EPS)
        m = (clean_rms / (clean_rms + bg_rms)).clamp(1e-3, 1.0 - 1e-3)
        source_rescaled = clean / m                     # rms = clean_rms + bg_rms
        background_rescaled = background / (1.0 - m)     # rms = clean_rms + bg_rms

        enroll = pos[0].squeeze(0)                       # V1 enrollment: ONE clean target utt [NW]

        out = {
            # ---- training keys (train_meanflow.py training_step) ----
            "source_rescaled_spec": self._S(source_rescaled),       # t=1 target
            "background_rescaled_spec": self._S(background_rescaled),  # t=0 start
            "enroll_spec": self._S(enroll),                          # reference prefix
            # ---- validation keys (train_meanflow.py validation_step) ----
            "mixture_spec": self._S(mix),                            # ODE x_init at t=mixing_ratio
            "mixing_ratio": m.float(),                               # scalar -> collates to [B]
            "source": clean.float(),                                 # waveform, for iSTFT length + SI-SDR
            "nuisance_sources": nuisance_sources.float(),             # [J, NW] for leakage loss
            # ---- eval keys (used by eval/eval_benchmark.py) ----
            "mixture": mix.float(),
            "background": background.float(),
            "enroll": enroll.float(),
            "source_rescaled": source_rescaled.float(),              # metric ref (model targets this up to scale)
            "background_rescaled": background_rescaled.float(),      # t=0 waveform (t-predicter convex-path synth)
            "mixture_rescaled": mix.float(),                         # metric mixture ref (input SI-SDR baseline)
            "utt_id": str(idx),
            "mixture_filename": f"{idx}.wav",
            "alpha": m.float(),                                      # parity with MeanFlow (train ignores)
        }
        if self.emit_posneg:
            pos_wave = pos.sum(dim=0)   # [1, NW]  (sum the positive stack to mono; == clean target for V1 data)
            neg_wave = neg.sum(dim=0)   # [1, NW]
            out["pos_wave"] = pos_wave.float()
            out["neg_wave"] = neg_wave.float()
            out["pos_spec"] = self._S(pos_wave.squeeze(0))
            out["neg_spec"] = self._S(neg_wave.squeeze(0))
        return out


def _emit_posneg(config: dict) -> bool:
    return (config or {}).get("enroll", {}).get("provider", "v1_raw") in (
        "v2a_posneg_concat", "v2b_pn_encoder")


def get_dataloaders(config, is_ddp=False, world_size=1, rank=0):
    """Build (train_loader, val_loader). Same signature MeanFlow's DataModule calls.

    Lightning's DDPStrategy injects its own DistributedSampler, so we return plain shuffled loaders
    (``is_ddp`` is accepted for API-compatibility but unused, mirroring MeanFlow's own DataModule).
    """
    dcfg = config["dataset"]
    tcfg = config["train"]
    has_noise = bool(dcfg.get("snr_db_range", []))

    train_roots = dcfg.get("train_roots", ["LibriSpeech/train-clean-360", "LibriSpeech/train-clean-100"])
    val_root = dcfg.get("val_root", "LibriSpeech/dev-clean")
    tr_noise = _noise_dir(dcfg.get("train_noise", "wham_noise/tr"), has_noise)
    cv_noise = _noise_dir(dcfg.get("val_noise", "wham_noise/cv"), has_noise)

    inners_train = [_build_inner(r, tr_noise, dcfg, reproducable=False) for r in train_roots]
    inner_val = _build_inner(val_root, cv_noise, dcfg, reproducable=True)

    # One virtual epoch = `samples_per_epoch` global mixtures, set to MeanFlow's EXACT Libri2Mix train
    # ConcatDataset length = 2*(50800 train-360 + 13900 train-100) = 129400 (LibriMixInformed counts
    # (mixture, target_speaker) pairs, i.e. 2 per 2-spk mixture; cross-checked by their log_interval
    # 1011 = ceil(129400/(4 GPU * 32 batch))). On 8 GPUs x batch 32 x accum 1 this is ~505 optimizer
    # steps/epoch == MeanFlow's 1011 microbatches/4GPU/accum2, so the epoch-based LR cosine (t_max=50),
    # warmup (5) and alpha curriculum (end=2000) all align with MeanFlow in step space.
    # [samples_per_epoch 의미] = "1 epoch을 섞은 소리 몇 개로 칠지" 정하는 가상 길이(virtual length).
    # 실제 파일 개수가 아니라, 우리가 '한 바퀴 = 이만큼'이라고 약속한 수예요. 학습할 때는 매번 새로
    # 무작위로 섞으므로(아래 _pick) 같은 파일이 또 나와도 상관없습니다. 이 값이 epoch 길이/LR 스케줄
    # 길이를 결정하니, 빠르게 돌려보고 싶으면 config에서 작게 줄여도 됩니다(예: 2000).
    train_len = int(dcfg.get("samples_per_epoch", 129400))
    val_len = int(dcfg.get("val_size", 200))
    emit = _emit_posneg(config)

    train_ds = PNLibriMixInformed(inners_train, dcfg, train_len, train=True, emit_posneg=emit)
    val_ds = PNLibriMixInformed([inner_val], dcfg, val_len, train=False, emit_posneg=emit)

    if int(tcfg["num_workers"]) == 0:
        # num_workers=0 이면 _worker_init 이 아예 안 불림 -> 메인 프로세스 RNG 를 rank 별로 여기서 분리.
        # (val 은 영향 없음: reproducable=True 아이템은 idx 로 시드되므로 전역 RNG 와 무관.)
        base = (torch.initial_seed() + 100003 * _ddp_rank()) % (2 ** 31 - 1)
        random.seed(base)
        np.random.seed(base)
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True,
        num_workers=int(tcfg["num_workers"]), pin_memory=True, worker_init_fn=_worker_init,
        persistent_workers=int(tcfg["num_workers"]) > 0,
    )
    val_workers = min(2, int(tcfg["num_workers"]))   # val is tiny (val_size items); keep its pool small
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False,
        num_workers=val_workers, pin_memory=True, worker_init_fn=_worker_init,
        persistent_workers=False,   # don't keep a 2nd worker pool alive (avoids A+B CPU oversubscription)
    )
    return train_loader, val_loader


def build_test_dataset(config) -> PNLibriMixInformed:
    """Fixed test set for offline evaluation (reproducable items on test-clean). batch_size=1 downstream."""
    dcfg = config["dataset"]
    has_noise = bool(dcfg.get("snr_db_range", []))
    test_root = dcfg.get("test_root", "LibriSpeech/test-clean")
    tt_noise = _noise_dir(dcfg.get("test_noise", "wham_noise/tt"), has_noise)
    inner = _build_inner(test_root, tt_noise, dcfg, reproducable=True)
    test_n = int(config.get("eval", {}).get("test_n", 100))
    return PNLibriMixInformed([inner], dcfg, test_n, train=False, emit_posneg=_emit_posneg(config))
