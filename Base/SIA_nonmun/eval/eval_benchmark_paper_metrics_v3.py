# =====================================================================================
# [이 파일이 하는 일 — 친절한 한국어 설명]
# 학습이 끝난 flow 모델(또는 비교용 pnenroll baseline)을 *똑같은 고정 테스트 음성*으로 평가해서
# 점수(SI-SDR / PESQ / STOI = 음질·명료도 지표)를 매기는 채점 프로그램이에요.
#
# [꼭 알아야 할 두 가지 설정]
#  1) NFE (Number of Function Evaluations = 모델을 몇 번 호출해 한 번에 정답에 다가갈지):
#     이 프로젝트의 표준은 NFE=1 (1-step, 단 한 번 호출). 그래서 아래 --nfe-list 기본값을 "1"로 둡니다.
#     숫자를 늘리면(예: 1,4,16) 더 여러 번 나눠 가지만, 기본은 빠른 1-step 추론입니다.
#  2) t-mode (시작 시각 t 를 어디서 가져올지):
#     - "oracle"    : 정답에서 계산한 진짜 m(이상적 상한값). "최고로 잘 됐을 때" 측정용.
#     - "mean"      : batch 평균 t (예전 검증 방식, 참고용 교차확인).
#     - "predicted" : 학습해 둔 t-predicter(섞임 비율 예측기)가 추정한 t. ← 실제로 쓰는 현실적 방식.
#     기본값을 "predicted"로 둡니다 (정답을 모르는 진짜 상황과 같게).
#  => 정리: 아무 옵션 안 주면 "NFE=1 + predicted-t" 로 1-step 추론을 합니다.
#
# ✏️ 바꿔도 되는 곳: 명령줄 옵션들(--ckpt, --n, --shard/--nshards, --nfe-list, --t-mode, --split 등).
# ⚠️ 만지지 마세요: 지표 계산(per_item_metrics)과 ODE 적분 루프 — baseline과 '완전히 똑같이' 맞춰
#    공정 비교를 보장하는 부분이라, 손대면 비교가 무의미해집니다.
# =====================================================================================
"""eval/eval_benchmark.py — RIGOROUS, apples-to-apples benchmark vs the PN-Enroll baseline.

Both the flow model AND the discriminative pnenroll baseline are evaluated on the IDENTICAL fixed
dev set (same NoisyFlowTSEDataset construction as baseline_pnenroll.py, reproducable=True -> the same
mixture at every index), with the IDENTICAL metric implementations (torchmetrics SI-SDR zero_mean,
narrowband PESQ, pystoi standard STOI). So:

  * running --model pnenroll here MUST reproduce the baseline (SI-SDR ~1.716) -> proves the harness,
  * running --model flow --ckpt <best|last> measures the flow model on the SAME 200 mixtures.

Flow sampling = the SAME ODE the validation_step uses (Euler, x_init=mixture_spec, integrate t->1,
r=1.0), but with the PER-SAMPLE ORACLE start time t = mixing_ratio m = clean_rms/(clean+bg) (the
"optimal t known" assumption). NFE=1 is a single Euler step:  x1 = mix + (1-m) * v(mix, t=m, r=1).

Sharding: --shard i --nshards K assigns indices {idx : idx % K == i}, so K processes (one per GPU)
cover the set with balanced, reproducible load. Each writes per-item metrics to --out (JSON);
eval/aggregate.py merges the shards into the final table.

Usage (one shard):
  CUDA_VISIBLE_DEVICES=0 python eval/eval_benchmark.py --model pnenroll --n 200 --shard 0 --nshards 8 \
      --out eval/out/pnenroll.s0.json
  CUDA_VISIBLE_DEVICES=0 python eval/eval_benchmark.py --model flow \
      --ckpt checkpoints/flow_best.ckpt --tpred-ckpt checkpoints/t_predicter_best.ckpt \
      --nfe-list 1 --t-mode predicted --split test \
      --n 200 --shard 0 --nshards 8 --out eval/out/best.s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# repo root = parent of eval/
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from torchmetrics.functional import (
    scale_invariant_signal_distortion_ratio as _si_sdr,
    scale_invariant_signal_noise_ratio as _si_snr,
    signal_noise_ratio as _snr,
)
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as _pesq
from pystoi import stoi as _stoi

_dnsmos = None  # intentionally disabled

from pndata.online_mixer import NoisyFlowTSEDataset
from utils.transforms import stft_torch, istft_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data (== baseline)
def build_dataset(config, n, split="dev"):
    """EXACT same NoisyFlowTSEDataset construction as baseline_pnenroll.py (so pnenroll reproduces
    1.716 and flow sees identical mixtures). Pulls split/snr/source_num from config to stay in sync.
    split='dev' -> val_root/val_noise (dev-clean, monitoring); split='test' -> test_root/test_noise
    (test-clean, the paper's held-out split for the final N=5000 report)."""
    d = config["dataset"]
    sr = int(d["sample_rate"])
    nw = sr * int(d["segment"])
    snr = list(d.get("snr_db_range", []) or [])
    if split == "test":
        root_rel, noise_rel = d.get("test_root", "LibriSpeech/test-clean"), d.get("test_noise", "wham_noise/tt")
    else:
        root_rel, noise_rel = d.get("val_root", "LibriSpeech/dev-clean"), d.get("val_noise", "wham_noise/cv")
    val_root = os.path.join(REPO, "data", root_rel)
    noise_dir = (os.path.join(REPO, "data", noise_rel) + "/") if snr else ""
    ds = NoisyFlowTSEDataset(
        root_dir=val_root,
        noise_dir=noise_dir,
        sample_rate=sr, wave_length=nw, pos_example_length=nw, neg_example_length=nw,
        source_num=int(d["source_num"]), enroll_num=int(d["source_num"]),
        active_num=tuple(d.get("active_num", (-1, 1, -1))),
        snr_db_range=snr,
        special_spk=tuple(d.get("special_spk", ("Partial_Pos", "Partial_Neg"))),
        partial_range=tuple(d.get("partial_range", (0.33, 0.66))),
        neg_partial_range=tuple(d.get("neg_partial_range", (1.0, 1.0))),
        clean_enroll=bool(d.get("clean_enroll", False)),
        reproducable=True,                 # <- fixed item per index (matches baseline)
        filling_pattern="repeat",
        enroll_exclude_mixture_utt=False,  # <- baseline_pnenroll.py value (NOT the train default True)
    )
    return ds, sr, nw


# ----------------------------------------------------------------------------- metrics (== baseline)
def per_item_metrics(est, tgt, mix):
    """Extended PN-paper-compatible metrics.

    Saves:
      - input/output SNR and SNRi
      - input/output SI-SNR and SI-SNRi
      - existing SI-SDR and SI-SDRi
      - PESQ / STOI
      - DNSMOS OVRL
    """
    e = est.detach().float().cpu().reshape(1, -1)
    t = tgt.detach().float().cpu().reshape(1, -1)
    m = mix.detach().float().cpu().reshape(1, -1)

    # Official PN evaluator polarity correction:
    # compare +est / -est and keep lower target-MSE polarity.
    mse_pos = ( e - t).square().sum(dim=-1)
    mse_neg = (-e - t).square().sum(dim=-1)
    flip = mse_neg < mse_pos
    e = torch.where(flip.unsqueeze(-1), -e, e)

    # Actual waveform input SNR:
    # target energy / total interfering-background energy.
    bg = m - t
    eps = 1e-8
    target_power = t.square().mean(dim=-1)
    bg_power = bg.square().mean(dim=-1)
    actual_input_snr_db = float(
        (
            10.0
            * torch.log10(
                (target_power + eps)
                / (bg_power + eps)
            )
        ).mean()
    )

    # --------------------------------------------------------------
    # Official PN evaluator polarity correction
    #
    # Evaluate +estimate and -estimate, then keep the polarity
    # with the lower target MSE.
    # normalize=False in the official evaluator, so amplitude is
    # NOT rescaled here.
    # --------------------------------------------------------------
    mse_pos = (e - t).square().sum(dim=-1)
    mse_neg = (-e - t).square().sum(dim=-1)

    flip = mse_neg < mse_pos

    e = torch.where(
        flip.unsqueeze(-1),
        -e,
        e,
    )

    # --------------------------------------------------------------
    # PN paper metrics
    # Official evaluator:
    # snr(out,target) - snr(mix,target)
    # si_snr(out,target) - si_snr(mix,target)
    # --------------------------------------------------------------
    out_snr = float(_snr(e, t).mean())
    in_snr = float(_snr(m, t).mean())

    out_si_snr = float(_si_snr(e, t).mean())
    in_si_snr = float(_si_snr(m, t).mean())

    # --------------------------------------------------------------
    # Existing project metric: keep for historical comparison
    # --------------------------------------------------------------
    out_si_sdr = float(
        _si_sdr(e, t, zero_mean=True).mean()
    )
    in_si_sdr = float(
        _si_sdr(m, t, zero_mean=True).mean()
    )

    mse = float(
        torch.nn.functional.mse_loss(e, t).mean()
    )

    # --------------------------------------------------------------
    # PESQ
    # --------------------------------------------------------------
    try:
        pesq = float(
            _pesq(
                e,
                t,
                fs=16000,
                mode="nb",
            )
        )
    except Exception:
        pesq = float("nan")

    # --------------------------------------------------------------
    # STOI
    # --------------------------------------------------------------
    try:
        stoi = float(
            _stoi(
                t.numpy().squeeze(),
                e.numpy().squeeze(),
                16000,
                extended=False,
            )
        )
    except Exception:
        stoi = float("nan")

    # --------------------------------------------------------------
    # DNSMOS
    # Official PN evaluator scales only when output peak > 1.
    # --------------------------------------------------------------
    # DNSMOS intentionally disabled in this project.
    # DNSMOS intentionally disabled.
    dnsmos = float("nan")

    return {
        "mse": mse,
        "actual_input_snr_db": actual_input_snr_db,

        "input_snr": in_snr,
        "snr": out_snr,
        "snri": out_snr - in_snr,

        "input_si_snr": in_si_snr,
        "si_snr": out_si_snr,
        "si_snri": out_si_snr - in_si_snr,

        "in_si": in_si_sdr,
        "si_sdr": out_si_sdr,
        "si_sdri": out_si_sdr - in_si_sdr,

        "pesq": pesq,
        "stoi": stoi,
        "dnsmos": dnsmos,
    }


# ----------------------------------------------------------------------------- pnenroll baseline
def run_pnenroll(ds, indices):
    """Reproduce baseline_pnenroll.estimate() exactly, per-item."""
    from baseline_pnenroll import load_pn_full, estimate
    from pn_encoder import load_pn_encoder
    ckpt = os.path.join(REPO, "checkpoints", "proposed-monaural.pt")
    pn_encoder = load_pn_encoder(ckpt, DEVICE)
    pn_full = load_pn_full(DEVICE)
    rows = []
    for s in indices:
        sample, pos, neg = ds[s]
        sample = sample.unsqueeze(0).to(DEVICE)
        pos, neg = pos.unsqueeze(0).to(DEVICE), neg.unsqueeze(0).to(DEVICE)
        mix = sample.sum(dim=1).squeeze(1)            # [1,NW]
        tgt = sample[:, :1].sum(dim=1).squeeze(1)     # [1,NW] active target (row 0)
        est = estimate(pn_encoder, pn_full, mix, pos, neg, tgt)
        r = per_item_metrics(est, tgt, mix)
        r["idx"] = int(s)
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- flow model
def _flow_batch(ds, idx_list, dcfg):
    """Replicate data/datasets.py __getitem__ for a list of indices -> batched flow tensors."""
    n_fft, hop, win = int(dcfg["n_fft"]), int(dcfg["hop_length"]), int(dcfg["win_length"])
    scale = float(dcfg.get("stft_scale", 1.0))
    eps = 1e-8
    mix_specs, ms, pos_w, neg_w, srcs, mixes = [], [], [], [], [], []
    for s in idx_list:
        sample, pos, neg = ds[s]
        mix = sample.sum(dim=0).squeeze(0)            # [NW]
        clean = sample[0].squeeze(0)                  # [NW]
        bg = mix - clean
        crms = torch.sqrt(torch.mean(clean ** 2)).clamp_min(eps)
        brms = torch.sqrt(torch.mean(bg ** 2)).clamp_min(eps)
        m = (crms / (crms + brms)).clamp(1e-3, 1.0 - 1e-3)
        mix_specs.append((stft_torch(mix, n_fft=n_fft, hop_length=hop, win_length=win) / scale).float())
        ms.append(m.float())
        pos_w.append(pos.sum(dim=0).float())          # [1,NW]
        neg_w.append(neg.sum(dim=0).float())          # [1,NW]
        srcs.append(clean.float())
        mixes.append(mix.float())
    return {
        "mixture_spec": torch.stack(mix_specs),       # [B,512,T]
        "mixing_ratio": torch.stack(ms),              # [B]
        "pos_wave": torch.stack(pos_w),               # [B,1,NW]
        "neg_wave": torch.stack(neg_w),               # [B,1,NW]
        "source": torch.stack(srcs),                  # [B,NW]
        "mixture": torch.stack(mixes),                # [B,NW]
    }


def _load_tpredicter(tpred_ckpt):
    """Load a trained TPredicterPN from a train_t_predicter_pn.py Lightning checkpoint (model.* keys)."""
    from models.t_predicter import TPredicterPN
    ck = torch.load(tpred_ckpt if os.path.isabs(tpred_ckpt) else os.path.join(REPO, tpred_ckpt), map_location="cpu")
    mcfg = (ck.get("hyper_parameters", {}) or {}).get("model", {"C": 1024})
    tp = TPredicterPN(**mcfg)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    tp.load_state_dict(sd, strict=True)
    return tp.eval().to(DEVICE)


@torch.no_grad()
def run_flow(ds, indices, config, ckpt_path, nfe_list, t_mode, precision, batch_size, tpred_ckpt=None):
    from train_meanflow import LightningModule
    dcfg = config["dataset"]
    model = LightningModule.load_from_checkpoint(ckpt_path, config=config, strict=True)
    model.eval().to(DEVICE)
    tpred = _load_tpredicter(tpred_ckpt) if t_mode == "predicted" else None
    n_fft, hop, win = int(dcfg["n_fft"]), int(dcfg["hop_length"]), int(dcfg["win_length"])
    autocast = (precision == "bf16")

    results = {nfe: [] for nfe in nfe_list}
    for i in range(0, len(indices), batch_size):
        idx_list = indices[i:i + batch_size]
        b = _flow_batch(ds, idx_list, dcfg)
        bgpu = {k: v.to(DEVICE) for k, v in b.items()}
        B = bgpu["mixture_spec"].shape[0]
        ones = torch.ones(B, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            # [t(시작 시각) 고르기 — 위 3가지 t-mode] 정답을 알면 oracle, 모르면 predicted를 씁니다.
            if t_mode == "predicted":                                # 실제로 쓰는 방식: 학습된 t-predicter가 t 추정
                enroll_emb = model.pn_encoder.encode(bgpu["pos_wave"], bgpu["neg_wave"]).float()
                m = tpred(bgpu["mixture"], enroll_emb).reshape(B).float().clamp(1e-3, 1.0 - 1e-3)
            elif t_mode == "mean":                                   # 교차확인용: batch 평균 t (예전 검증 방식)
                m = bgpu["mixing_ratio"].reshape(B).mean().expand(B).contiguous()
            else:                                                    # oracle: 정답에서 계산한 진짜 m (이상적 상한)
                m = bgpu["mixing_ratio"].reshape(B)
            enrollment = model._enrollment(bgpu)                     # [B,512,M]
            # [NFE 만큼 나눠서 t=m -> 1 로 한 걸음씩 이동] nfe=1 이면 단 한 걸음(1-step)으로 끝.
            for nfe in nfe_list:
                x = bgpu["mixture_spec"].clone()
                dt = (1.0 - m) / nfe                                 # [B] 한 걸음 크기 = 남은 거리 / 걸음 수
                for k in range(nfe):
                    t_cur = (m + k * dt).clamp(0.0, 1.0)
                    v = model.model(x, t_cur, ones, enrollment)      # r=1.0 (== validation_step)
                    x = x + dt.view(B, 1, 1) * v
                src_hat = istft_torch(x.float(), n_fft=n_fft, hop_length=hop, win_length=win,
                                      length=bgpu["source"].shape[-1])

                # --------------------------------------------------
                # PHYSICAL INVERSE-RESCALE
                #
                # Flow endpoint is source_rescaled = clean / m.
                # Restore physical waveform amplitude before metrics.
                #
                # predicted mode -> deployable predicted m
                # oracle mode    -> true m
                # --------------------------------------------------
                src_hat = (
                    src_hat
                    * m.to(dtype=src_hat.dtype).view(B, 1)
                )

                for j, s in enumerate(idx_list):
                    r = per_item_metrics(src_hat[j], bgpu["source"][j], bgpu["mixture"][j])
                    r["idx"] = int(s)

                    # metric에 실제 적용된 deployable/oracle scale
                    r["output_scale_m"] = float(
                        m[j].detach().float().cpu()
                    )

                    # 진단용 true mixture ratio
                    r["true_m"] = float(
                        bgpu["mixing_ratio"]
                        .reshape(B)[j]
                        .detach()
                        .float()
                        .cpu()
                    )

                    results[nfe].append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_PNNoisyFlow_v2b_crossattn_jitter.yaml")
    ap.add_argument("--model", choices=["flow", "pnenroll"], required=True)
    ap.add_argument("--ckpt", default=None, help="flow checkpoint (required for --model flow)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    # [기본값] 표준 추론은 1-step 이라 nfe-list 기본을 "1" 로 둡니다(원하면 "1,4,16" 처럼 늘려도 됨).
    ap.add_argument("--nfe-list", default="1")
    # [기본값] 정답을 모르는 실제 상황과 같게 t-mode 기본을 "predicted"(t-predicter 추정) 로 둡니다.
    ap.add_argument("--t-mode", choices=["oracle", "mean", "predicted"], default="predicted")
    ap.add_argument("--tpred-ckpt", default=None, help="TPredicterPN checkpoint (required for --t-mode predicted)")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--split", choices=["dev", "test"], default="test")  # README의 'test-clean 자체 점검' 기본값과 일치
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import yaml
    config = yaml.safe_load(open(os.path.join(REPO, a.config)))
    ds, sr, nw = build_dataset(config, a.n, split=a.split)
    indices = [s for s in range(a.n) if s % a.nshards == a.shard]
    print(f"[eval] model={a.model} shard={a.shard}/{a.nshards} items={len(indices)} "
          f"device={DEVICE} t-mode={a.t_mode}", flush=True)

    payload = {"model": a.model, "shard": a.shard, "nshards": a.nshards, "n": a.n,
               "config": a.config, "indices": indices}
    if a.model == "pnenroll":
        payload["pnenroll"] = run_pnenroll(ds, indices)
    else:
        assert a.ckpt, "--ckpt required for --model flow"
        nfe_list = [int(x) for x in a.nfe_list.split(",") if x]
        payload["ckpt"] = a.ckpt
        payload["nfe_list"] = nfe_list
        payload["t_mode"] = a.t_mode
        payload["precision"] = a.precision
        assert a.t_mode != "predicted" or a.tpred_ckpt, "--tpred-ckpt required for --t-mode predicted"
        res = run_flow(ds, indices, config, os.path.join(REPO, a.ckpt) if not os.path.isabs(a.ckpt) else a.ckpt,
                       nfe_list, a.t_mode, a.precision, a.batch_size, tpred_ckpt=a.tpred_ckpt)
        payload["tpred_ckpt"] = a.tpred_ckpt
        payload["results"] = {str(k): v for k, v in res.items()}

    outp = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(payload, f)
    print(f"[eval] wrote {outp} ({len(indices)} items)", flush=True)


if __name__ == "__main__":
    main()
