"""eval_reference_protocol.py — 학습한 flow(t-predicter, NFE=1)를 '논문과 동일한 평가 프로토콜'로 측정합니다.

무엇을 재나: 혼합음에서 등록 화자를 1-step으로 뽑은 추정치를 GT(정답 음성)와 비교해
  SNRi / SI-SNRi (신호 개선량, dB) · PESQ(pypesq, 음질) · STOI(명료도) 로 점수화합니다.
프로토콜(논문 Table-1, 3-spk mixture / 3-spk enroll): 6초 혼합 / 3초 등록, WHAM 잡음 SNR[-2.5,2.5],
부분(partial) pos/neg 마스킹, oracle 부호선택. N=5000(=50×100).

데이터 배치(README 참고): data/_test_data/ (테스트 화자 split, LibriSpeech 형식 40화자) + data/wham_noise/tt/ (잡음).
  ※ _test_data 는 학습용 data/LibriSpeech(심볼릭) 와 독립된 폴더입니다 — 충돌 방지를 위해 data/ 바로 아래 둡니다.

실행 (1장):  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python eval_reference_protocol.py \
                --shard 0 --nshards 1 --n 200 --out out_eval/s0.json
여러 장 샤딩 + 집계는 run_eval.sh 참고.
"""
import argparse, os, sys, json, random
import torch
torch.set_num_threads(1)   # ⚠️ 필수: CPU 메트릭이 코어를 다 잡으면 여러 샤드가 경쟁해 매우 느려짐 (1로 고정)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset.LibriSpeech_single_emb import LibriDataset_single_emb   # vendored 논문 데이터셋
from pypesq import pesq as pypesq_pesq                               # 논문과 동일한 PESQ 라이브러리
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr_loss
from torchmetrics.functional import signal_noise_ratio as snr_loss
import yaml
from utils.transforms import stft_torch, istft_torch
from models.t_predicter import TPredicterPN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
# ===== 평가 프로토콜 상수 (논문 Table-1과 동일 — ⚠️ 바꾸지 마세요) =====
TEST_DIR = os.path.join(HERE, "data/_test_data/")              # ✏️ 테스트 화자 split 위치 (README) — LibriSpeech 형식 40화자
NOISE_DIR = os.path.join(HERE, "data/wham_noise/") + "tt/"      # WHAM 테스트 잡음
source_num = 3; enroll_num = 3                                  # 혼합 3명 / 등록 3명 (= 논문 3/3)
pos_example_length = 48000; neg_example_length = 48000          # 등록 3초
pos_num = [0, 2]                                                # partial positive interferer 개수 범위
active_num = [-1, 1, -1]                                        # 타겟 = row 0
wave_length = 6 * SR                                            # 혼합 6초
snr_db_range = [-2.5, 2.5]                                      # 잡음 SNR 범위
partial_range = [0.33, 0.66]; neg_partial_range = [0.33, 1.0]  # 마스킹 길이 범위
set_size = 50; repeat_num = 100                                # N = 5000
stoi_model = ShortTimeObjectiveIntelligibility(fs=SR, extended=False)


def build_dataset():
    return LibriDataset_single_emb(
        TEST_DIR, sample_rate=SR, wave_length=wave_length,
        pos_example_length=pos_example_length, neg_example_length=neg_example_length,
        snr_db_range=snr_db_range, source_num=source_num, min_source_num=source_num,
        enroll_num=enroll_num, min_enroll_num=enroll_num, active_num=active_num,
        reproducable=True, normalize=False, filling_pattern="repeat", return_dvec=False,
        dvec_rate=50, include_silent=False, special_spk=[], reverb="none", binaural=False,
        reverb_cond=False, zero_in_tgt=False, noise_dir=NOISE_DIR, same_disturb=False)


def apply_masking(pos, neg, gidx):
    """등록 음성에 부분 마스킹 (논문 평가와 동일). gidx로 시드 -> 재현 가능."""
    rng = random.Random(gidx)
    for i in range(rng.randint(pos_num[0], pos_num[1])):
        L = int(pos_example_length * rng.uniform(*partial_range)); s = rng.randint(0, pos_example_length - L)
        pos[:, active_num[1] + i, :, :s] = 0; pos[:, active_num[1] + i, :, s + L:] = 0; neg[:, i] = 0
    for i in range(enroll_num - 1):
        L = int(neg_example_length * rng.uniform(*neg_partial_range)); s = rng.randint(0, neg_example_length - L)
        neg[:, i, :, :s] = 0; neg[:, i, :, s + L:] = 0
    return pos, neg


def load_flow(flow_ckpt, tpred_ckpt, cfg_path):
    cfg = yaml.safe_load(open(os.path.join(HERE, cfg_path)))
    pk = cfg.get("paths", {}).get("pn_ckpt")
    if pk and not os.path.isabs(pk):
        cfg["paths"]["pn_ckpt"] = os.path.join(HERE, pk)
    from train_meanflow import LightningModule
    flow = LightningModule.load_from_checkpoint(os.path.join(HERE, flow_ckpt), config=cfg, strict=True).eval().to(DEVICE)
    ck = torch.load(os.path.join(HERE, tpred_ckpt), map_location="cpu")
    mcfg = (ck.get("hyper_parameters", {}) or {}).get("model", {"C": 1024})
    tp = TPredicterPN(**mcfg)
    tp.load_state_dict({k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True)
    tp.eval().to(DEVICE)
    return flow, tp, cfg


@torch.no_grad()
def flow_forward(flow, tpred, cfg, audio, pos, neg):
    """1-step 추출: t-predicter가 시작시점 m̂ 예측 -> mixture에서 한 발짝 Euler -> istft. (실사용 predicted-t)"""
    d = cfg["dataset"]; n_fft, hop, win = int(d["n_fft"]), int(d["hop_length"]), int(d["win_length"])
    pos_w, neg_w = pos.sum(dim=1), neg.sum(dim=1)
    mix_1d = audio.sum(dim=1).squeeze(1)                            # [B, NW]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        enrollment = flow._enrollment({"pos_wave": pos_w, "neg_wave": neg_w})
        pn_emb = flow.pn_encoder.encode(pos_w, neg_w).float()
        m_hat = tpred(mix_1d, pn_emb).reshape(mix_1d.shape[0]).float().clamp(1e-3, 1.0 - 1e-3)
        specs = torch.stack([stft_torch(mix_1d[b], n_fft=n_fft, hop_length=hop, win_length=win) for b in range(mix_1d.shape[0])])
        ones = torch.ones(mix_1d.shape[0], device=DEVICE)
        out_spec = specs + (1.0 - m_hat).view(-1, 1, 1) * flow.model(specs, m_hat, ones, enrollment)
    out = istft_torch(out_spec.float(), n_fft=n_fft, hop_length=hop, win_length=win, length=mix_1d.shape[-1])
    return (out * m_hat.view(-1, 1)).unsqueeze(1)                   # m̂로 자연 레벨 복원 (SNRi/PESQ 공정성용)


@torch.no_grad()
def metrics(out, gt, mix):
    out = torch.concat([out, -out], dim=1)                          # oracle 부호선택
    idx = torch.min((out - gt).pow(2).sum(dim=-1), dim=1)[1]
    out = torch.gather(out, 1, idx[:, None, None].repeat(1, 1, out.shape[-1]))
    snr = snr_loss(out, gt); si = si_snr_loss(out, gt)
    r = {"snr": float(snr.mean()), "si_snr": float(si.mean()),
         "snri": float((snr - snr_loss(mix, gt)).mean()), "si_snri": float((si - si_snr_loss(mix, gt)).mean())}
    try: r["pesq"] = float(pypesq_pesq(gt.cpu().squeeze().numpy(), out.cpu().squeeze().numpy(), SR))
    except Exception: r["pesq"] = float("nan")
    try: r["stoi"] = float(stoi_model(gt.cpu(), out.cpu()).item())
    except Exception: r["stoi"] = float("nan")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=set_size * repeat_num)
    ap.add_argument("--flow-ckpt", default="checkpoints/flow_best.ckpt")
    ap.add_argument("--tpred-ckpt", default="checkpoints/t_predicter_best.ckpt")
    ap.add_argument("--config", default="config/config_PNNoisyFlow_v2b_crossattn_jitter.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = build_dataset()
    flow, tp, cfg = load_flow(a.flow_ckpt, a.tpred_ckpt, a.config)
    indices = [g for g in range(a.n) if g % a.nshards == a.shard]
    rows = []
    for g in indices:
        audio, pos, neg = ds[g]
        audio = audio.to(DEVICE)[None]; pos = pos.to(DEVICE)[None]; neg = neg.to(DEVICE)[None]
        pos, neg = apply_masking(pos, neg, g)
        gt = audio[:, : active_num[1]].sum(dim=1); mix = audio.sum(dim=1)
        r = metrics(flow_forward(flow, tp, cfg, audio, pos, neg), gt, mix); r["idx"] = g
        rows.append(r)
    outp = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump({"shard": a.shard, "nshards": a.nshards, "n": a.n, "results": {"ours": rows}}, open(outp, "w"))
    print(f"[eval] shard {a.shard}: {len(indices)} items -> {outp}", flush=True)


if __name__ == "__main__":
    main()
