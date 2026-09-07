# PNFlowTSE — 등록 화자 한 명만 1-step으로 뽑아내는 흐름정합 TSE

## 1. 한 줄 소개

**여러 사람 목소리가 섞인 소리(mixture, 혼합음)에서, 미리 "등록"해 둔 한 사람의 목소리만 깨끗하게 뽑아내는 프로젝트입니다.** (TSE = Target Speaker Extraction, 목표 화자 추출)

핵심 아이디어는 이렇습니다.

- **흐름정합(flow matching)**: "섞인 소리 → 깨끗한 소리"로 가는 길(흐름)을 모델이 배웁니다. 보통 생성 모델은 이 길을 수십 번 나눠 걷지만, 우리는 **딱 한 발짝(1-step, NFE=1)** 만에 도착합니다. 그래서 빠릅니다.
- **등록(enrollment) 조건**: "이 사람을 뽑아줘"라는 정보를 두 종류로 줍니다. 원하는 사람의 예시 음성(positive, 양성)과 원하지 않는 사람의 예시 음성(negative, 음성). 이 둘을 **frozen(고정, 학습 안 함) PN-encoder(Positive/Negative encoder, 등록 화자 인코더)** 가 비교해서 화자 정보를 뽑아줍니다.
- **t-predicter(섞임비율 예측기)**: 한 발짝을 어디서 출발할지 정하려면 "지금 소리가 얼마나 섞여 있나(mixing ratio, 0~1)"를 알아야 합니다. 이걸 맞히는 작은 모델입니다.
- **MR-jitter(Mixing-Ratio jitter, 섞임비율 흔들기)**: t-predicter가 조금 틀려도 1-step 추론이 흔들리지 않도록, 학습할 때 출발점을 일부러 살짝 흔들어 **강건하게(robust)** 만듭니다.

구성품을 그림으로 보면 이렇습니다.

```
  pos 예시음성 ─┐
                ├─▶ [PN-encoder (frozen)] ─▶ 화자 조건 ─┐
  neg 예시음성 ─┘                                        │
                                                          ▼
  혼합음 ──▶ [t-predicter] ──▶ 출발 t ──▶ [ UDiT flow 모델 ] ──(1 step)──▶ 깨끗한 목표 화자 음성
```

> UDiT(U-shaped Diffusion Transformer, U자형 디퓨전 트랜스포머)가 실제로 소리를 밀어주는 본체 모델입니다.

---

## 2. 환경 셋업 — `bash setup.sh` 한 번이면 끝

conda(아나콘다) 환경 하나만 만들면 됩니다. **시스템에 깔린 CUDA 버전이 달라도 괜찮습니다** — conda가 PyTorch에 맞는 CUDA를 알아서 같이 깔아주기 때문입니다(시스템 CUDA를 건드리지 않습니다).

```bash
bash setup.sh
```

끝나면 이렇게 활성화해서 씁니다.

```bash
conda activate pnflowtse
```

`setup.sh`가 만들어 주는 것(검증된 버전 조합):

| 묶음 | 패키지 (버전) |
|---|---|
| 파이썬/딥러닝 | python 3.10, torch 2.8.0, torchaudio 2.8.0, pytorch-lightning 2.0.6 |
| 흐름정합 | flow_matching 1.0.10, einops 0.8.2 |
| 오디오/평가 | torchmetrics 0.11.4, pystoi 0.4.1, pesq 0.0.4, librosa 0.9.2, soundfile, scipy |
| 설정/로그 | omegaconf 2.3.0, numpy 1.23.5, tqdm, tensorboard |
| **reference 평가용** | **pypesq**(논문이 쓰는 PESQ 구현, `pesq`와 별개 패키지), **resemblyzer**(평가 데이터셋 무음 트림) — `eval_reference_protocol.py` 에 필요. 학습만 할 거면 없어도 동작합니다. |
| **음성 인코더(필수)** | **espnet==202412** — PN-encoder(`pn/encoder.py`)가 `espnet2`를 사용합니다. 없으면 임포트 단계에서 바로 에러납니다. |

> ⚠️ **주의 1**: `asteroid`는 **설치하지 마세요.** 옛 코드 흔적 때문에 충돌만 일으킵니다. 필요한 `AdamW`(옵티마이저)는 `utils/optim.py`에 자체 구현되어 있습니다.
>
> ⚠️ **주의 2**: `espnet` 설치는 의존성이 많아 **시간이 꽤 걸립니다(수 분~십수 분).** 멈춘 게 아니니 기다려 주세요.

---

## 3. 데이터 배치 — `data/` 아래에 두기

이 프로젝트는 두 가지 데이터를 씁니다.

- **LibriSpeech** : 깨끗한 사람 목소리 (목표 화자 + 방해 화자용)
- **WHAM!** (wham_noise) : 실제 환경 잡음 (소리를 더 어렵게 섞기 위함)

`data/` 폴더 아래에 아래 구조가 보이도록 두면 됩니다. **복사해도 되고, 용량을 아끼려면 심볼릭 링크(symbolic link, 바로가기)를 거는 것을 추천합니다.**

```
PNFlowTSE/
└── data/
    ├── LibriSpeech/            ← 학습/검증용 (아래 폴더 이름 그대로)
    │   ├── train-clean-360/    ← 학습용 (큰 셋)
    │   ├── train-clean-100/    ← 학습용 (작은 셋)
    │   ├── dev-clean/          ← 검증(validation)용
    │   └── test-clean/         ← (선택) 구 our-protocol 평가용
    ├── wham_noise/
    │   ├── tr/                 ← 학습용 잡음 (train)
    │   ├── cv/                 ← 검증용 잡음 (cross-validation)
    │   └── tt/                 ← 평가용 잡음 (test)  ← reference 평가에도 사용
    └── _test_data/             ← ★ reference 평가용 테스트 화자 split (LibriSpeech 형식 화자 폴더들)
```

> **`_test_data/` 는 무엇인가요?** 논문(arXiv:2502.16611)과 *완전히 똑같은 조건*으로 점수를 재기 위한 **고정 테스트 화자 split**입니다(논문 평가 코드가 쓰는 그 split). `run_eval.sh`(reference 평가)는 이 폴더를 씁니다. **학습용 `data/LibriSpeech` 와 충돌하지 않도록 `data/` 바로 아래 별도 폴더로 둡니다.** 이 split이 없으면 학습/구 평가는 되지만 reference 평가는 못 돕니다.

심볼릭 링크로 거는 예시(데이터가 다른 곳에 이미 있을 때):

```bash
# PNFlowTSE 폴더 안에서 실행
ln -s /실제/경로/LibriSpeech     data/LibriSpeech
ln -s /실제/경로/wham_noise      data/wham_noise
ln -s /실제/경로/_test_data      data/_test_data    # reference 평가용 (없으면 생략 가능)
```

> 폴더 이름이 위와 정확히 같아야 합니다(`train-clean-360`, `tr`, `_test_data` 등). 설정 파일(`config/*.yaml`)의 `dataset.train_roots`, `dataset.train_noise` 같은 키가 이 이름들을 그대로 가리킵니다.

---

## 4. 빠른 추론 — 동봉 체크포인트로 바로 (NFE=1)

**보통은 다시 학습할 필요 없이, 동봉된 체크포인트로 바로 평가만 하면 됩니다.** 필요한 체크포인트 3개가 `checkpoints/`에 이미 들어 있습니다.

| 파일 | 역할 |
|---|---|
| `checkpoints/proposed-monaural.pt` | frozen PN-encoder (등록 화자 인코더, 고정) |
| `checkpoints/flow_best.ckpt` | 학습 끝난 flow 본체 (UDiT) |
| `checkpoints/t_predicter_best.ckpt` | 학습 끝난 t-predicter (섞임비율 예측기) |

평가 실행 (reference 프로토콜):

```bash
bash run_eval.sh                 # 빠른 확인: N=200
N=5000 bash run_eval.sh          # 논문과 동일한 풀세트: N=5000
```

이 스크립트는 **논문(arXiv:2502.16611) Table-1 과 똑같은 조건**(`data/_test_data` + WHAM tt, 3-spk 혼합/3-spk 등록, 6초 혼합, 부분 등록 마스킹)에서 모델을 돌립니다. **모든 추론은 NFE=1(한 발짝)**, 출발점 t는 동봉된 t-predicter가 예측합니다(정답 m을 모르는 실제 추론 상황과 동일 = 실사용 가능한 평가).

> 내부적으로 `eval_reference_protocol.py`(추론+채점) → `aggregate_ref.py`(표 출력) 를 호출합니다. 위 4절의 `data/_test_data` 가 준비돼 있어야 합니다.

**무엇이 출력되나요?** 논문 Table-1 과 같은 형식으로, **논문 3-spk/3-spk 행과 나란히** mean±std 표가 나옵니다. 4지표 모두 클수록 좋습니다.

| 지표 | 뜻 | 좋은 값 |
|---|---|---|
| **SNRi** | Signal-to-Noise Ratio improvement. 입력 대비 SNR 개선량(dB). | 높을수록 ↑ |
| **SI-SNRi** | Scale-Invariant SNR improvement(= SI-SDRi). 스케일에 무관한 개선량(dB). | 높을수록 ↑ |
| **PESQ** | Perceptual Evaluation of Speech Quality(`pypesq`). 음질(1.0~4.5). | 높을수록 ↑ |
| **STOI** | Short-Time Objective Intelligibility. 명료도(0~1). | 높을수록 ↑ |

항목별 점수는 `eval/out/ref.s*.json` 으로 저장됩니다. 여러 GPU로 나눠 빠르게 N=5000 을 재는 방법은 `run_eval.sh` 하단 주석을 보세요.

**들어볼 수 있는 wav 샘플** 이 필요하면(점수 측정과 분리, 빠름):

```bash
CUDA_VISIBLE_DEVICES=0 python dump_samples_ref.py --n 20 --out-dir samples_ref
# idx별로 mixture / gt(정답) / est_ours(추출) wav 저장
```

> (참고) `data/_test_data` 가 없을 때는, 표준 test-clean 으로 도는 구 our-protocol 평가가 `eval/eval_benchmark.py` 에 남아 있습니다 — 논문과 직접 비교용은 아니고 빠른 자체 점검용입니다.

---

## 5. (선택) 재학습 — 순서가 중요합니다

동봉 체크포인트로 충분하면 이 절은 건너뛰어도 됩니다. **꼭 다시 학습한다면 반드시 아래 ① → ② 순서를 지키세요.**

```bash
# ① 먼저 t-predicter(섞임비율 예측기)를 학습
bash run_train.sh tpred

# ② 그다음 flow 본체를 MR-jitter 켜고 학습
bash run_train.sh flow
```

**왜 이 순서인가요?** flow 학습의 검증(validation)과 추론은 "출발 t"를 t-predicter가 정해 줘야 하기 때문입니다. 즉 **t-predicter가 먼저 준비돼 있어야 flow를 제대로 검증·튜닝**할 수 있습니다.

각 단계가 쓰는 설정 파일:

| 단계 | 명령 | config 파일 |
|---|---|---|
| ① t-predicter | `run_train.sh tpred` | `config/config_TPredicter_v2b.yaml` |
| ② flow (MR-jitter) | `run_train.sh flow` | `config/config_PNNoisyFlow_v2b_crossattn_jitter.yaml` |

학습 결과(체크포인트, 텐서보드 로그)는 config의 `train.log_dir` / `checkpoint.dir`이 가리키는 `exp/...` 폴더에 쌓입니다. 진행은 `tensorboard --logdir exp` 로 볼 수 있습니다.

---

## 6. GPU 1/2/4장(48GB, A6000) 튜닝표

A6000 48GB 기준으로, **전체 배치(global batch)를 일정하게 유지**(flow는 **256**, 더 가벼운 t-predicter는 **512**)하면서 GPU 수에 맞게 아래 키만 바꾸면 됩니다. (global batch = `batch_size × accumulation_steps × num_gpus`)

### flow 학습 (`config/config_PNNoisyFlow_v2b_crossattn_jitter.yaml`)

| GPU 수 | `train.batch_size` | `train.accumulation_steps` | `ddp.num_gpus` | global batch |
|:---:|:---:|:---:|:---:|:---:|
| 1장 | 16 | 16 | 1 | 256 |
| 2장 | 16 | 8 | 2 | 256 |
| 4장 | 16 | 4 | 4 | 256 |

공통 권장값: `train.num_workers` 4~6, `train.precision: bf16-mixed`.

### t-predicter 학습 (`config/config_TPredicter_v2b.yaml`)

t-predicter는 가벼워서 배치를 더 키울 수 있습니다.

| GPU 수 | `train.batch_size` | `train.accumulation_steps` | `ddp.num_gpus` |
|:---:|:---:|:---:|:---:|
| 1장 | 64 | 8 | 1 |
| 2장 | 64 | 4 | 2 |
| 4장 | 64 | 2 | 4 |

> 어떤 GPU를 쓸지는 `CUDA_VISIBLE_DEVICES`로 고르면 됩니다. 예: `CUDA_VISIBLE_DEVICES=0,1 bash run_train.sh flow` (이때 config의 `ddp.num_gpus`도 2로 맞춰 주세요).

---

## 7. 자주 막히는 곳 (트러블슈팅)

| 증상 | 원인 / 해결 |
|---|---|
| `setup.sh`가 한참 멈춘 듯함 | **espnet 설치가 원래 오래 걸립니다(수 분~십수 분).** 의존성이 많아서 그래요. 멈춘 게 아니니 기다리세요. |
| `ModuleNotFoundError: espnet2` | espnet이 안 깔린 것. `conda activate pnflowtse` 후 `pip install espnet==202412` 로 다시 설치. |
| `FileNotFoundError` (데이터 없음) | `data/LibriSpeech/`, `data/wham_noise/` 아래 폴더 이름이 4절 구조와 **정확히** 같은지 확인. 심볼릭 링크가 깨졌는지도 `ls -l data/` 로 확인. |
| `run_eval.sh` 가 `_test_data` 못 찾음 | reference 평가는 `data/_test_data/`(논문용 테스트 화자 split)가 필요합니다(4절 참고). 없으면 `eval/eval_benchmark.py`(test-clean) 로 자체 점검만 가능. |
| `ModuleNotFoundError: pypesq` / `resemblyzer` | reference 평가 전용 의존성입니다. `pip install pypesq resemblyzer` (setup.sh 최신본엔 포함). 학습만 할 거면 무시해도 됩니다. |
| **OOM** (CUDA out of memory) | `batch_size`를 **절반**으로 줄이고 `accumulation_steps`를 **2배**로 늘리세요(global batch는 그대로 유지). 그래도 안 되면 `num_workers`도 낮춰 보세요. |
| 학습이 멀티 GPU에서 멈춤(deadlock) | `ddp.num_gpus`와 실제 보이는 GPU 수(`CUDA_VISIBLE_DEVICES`)가 **일치**하는지 확인. DDP 관련 코드는 건드리지 마세요. |
| 점수가 이상하게 낮음 | 추론·평가는 **전부 NFE=1**이 맞는지, 그리고 t-predicter 체크포인트를 제대로 불러왔는지 확인. (재학습했다면 ① t-predicter → ② flow 순서를 지켰는지 점검) |
| `asteroid` 관련 충돌 | **asteroid는 설치하지 마세요.** 필요한 옵티마이저는 `utils/optim.py`에 들어 있습니다. |

---

## 8. 학습이 정확히 어떻게 이뤄지나 (deep dive)

> "내부에서 무슨 일이 일어나는지" 궁금한 분을 위한 상세 설명입니다. 그냥 쓰기만 할 거면 건너뛰어도 됩니다.

### 8-1. 데이터 — "가우시안 노이즈"는 넣지 않습니다 ⭐

보통의 생성 모델(디퓨전 · 일반적인 flow matching)은 **무작위 가우시안 노이즈 `N(0, I)`에서 출발해** 데이터를 만들어 갑니다. **우리는 그렇게 하지 않습니다.**

- 학습 샘플 한 개 = 실제 혼합음
  `mixture = clean(목표 화자) + 간섭 화자들 + WHAM 잡음`
  - **WHAM** = 실제 환경 잡음 *녹음*입니다(가우시안 아님). SNR은 `snr_db_range = [-3, 3] dB`에서 뽑아, 목표 화자(clean) 파워 기준으로 스케일해 더합니다(power-domain: `snr = 10^(snr_db/10)`).
- `source`(목표) = clean 목표 화자, `background` = `mixture − clean` = 간섭 화자 + WHAM.
- **핵심**: 흐름의 출발점(t=0)은 가우시안이 아니라 **혼합음/배경 그 자체**입니다. 즉 "노이즈→데이터 생성"이 아니라 **"혼합음 → 깨끗한 목표 화자"로 직접 흐릅니다.**
- 코드에서 난수는 **시간/섞임비율 샘플링**(예: flow의 `t = sigmoid(randn)`, t-predictor의 `alpha ~ U(0,1)`)과 **MR-jitter의 출발시각 흔들기**에만 쓰입니다. **신호/스펙트로그램에 가우시안 노이즈를 더하는 곳은 어디에도 없습니다.**

### 8-2. 스케일 — 혼합음을 "흐름 경로 위"에 정확히 올리는 트릭

흐름은 직선 경로 `z(t) = (1−t)·background + t·source` (t=0 배경 → t=1 깨끗한 소스) 위에서 배웁니다. 그런데 그냥 `mixture = clean + background`는 이 (배경, 소스) 직선 위에 안 놓입니다. 그래서 **샘플마다 비율 m으로 재스케일**합니다:

```
m                   = clean_rms / (clean_rms + bg_rms)      # 0~1 사이
source_rescaled     = clean      / m
background_rescaled = background  / (1 − m)
```

그러면:

```
z(m) = (1−m)·background_rescaled + m·source_rescaled
     = (1−m)·background/(1−m) + m·clean/m
     = background + clean = mixture          ← 정확히 t=m 에서 실제 혼합음과 일치!
```

- 효과 ①: 실제 혼합음이 경로의 **t = m 지점에 정확히** 놓입니다 → 추론 때 "혼합음에서 t=m으로 한 발짝(1-step) 출발"이 수학적으로 정당해집니다.
- 효과 ②: 두 rescaled 신호의 RMS가 `clean_rms+bg_rms`로 똑같아져 입력 스케일이 균형잡힙니다. (이 재스케일은 스케일 불변 지표 SI-SDR 점수엔 영향 없음.)
- 흐름은 **STFT(복소 스펙트로그램) 도메인**에서 동작합니다(파형 아님). `n_fft=510`(→ 주파수 256개)의 실수·허수부를 이어붙여 **512 채널**. `stft_scale=1.0`이라 사실상 추가 정규화 없음.

### 8-3. flow 손실 — "속도"를 배운다

- 정답 속도(직선이라 일정): `v = source_rescaled − background_rescaled`.
- 모델이 중간점 `z(t)`와 시간 `(t, r)`, 등록정보를 보고 속도 `u`를 예측 → 손실 = adaptive-L2(`u − v`).
- alpha 커리큘럼: 초반엔 "정직한 궤적 따라가기(alpha=1)", 후반엔 "한 방에 도착 연습(consistency, alpha→0)". **단, 우리 최종 모델은 아래 8-5의 MR-jitter 손실로 학습합니다**(config `mr_jitter.enabled=true`이면 이 손실로 대체).

### 8-4. t-predictor — 출발 시각 m을 맞히는 작은 모델

- 문제: 테스트 때는 정답(clean)을 모르니 `m = clean_rms/(clean_rms+bg_rms)`을 알 수 없습니다.
- 해결: **혼합음 + 등록(pos/neg)** 만 보고 `m̂`을 회귀로 예측합니다.
  - 입력 = (혼합음 → ECAPA-TDNN 분기) + (등록 pos/neg → frozen PN-encoder 임베딩 분기), 출력 = 스칼라 예측값.
  - **학습**: 매 스텝 `alpha ~ U(0,1)`를 무작위로 뽑아 합성 혼합음 `x_alpha = (1−alpha)·background + alpha·source`를 만들고, 모델이 그 `alpha`를 되맞히도록 MSE로 학습합니다. (정답 m이 아니라 무작위 alpha로 학습 → 모든 섞임 정도를 골고루 보는 증강 효과)
  - **검증·추론**: 실제 혼합음은 경로의 `t=m` 지점에 놓이므로(§8-2), 학습된 모델에 실제 혼합음을 넣으면 `m̂ ≈ m`을 내놓습니다. 검증은 이 `m̂`을 진짜 `m`(`batch['mixing_ratio']`)과 비교(MSE)해 정확도를 잽니다.
- 역할: 1-step 추론의 **출발 시각 `t = m̂`** 을 제공.
- ⚠️ 그래서 학습 순서가 **① t-predictor → ② flow** 입니다 (flow의 검증이 t-predictor의 `m̂`을 사용).

### 8-5. MR-jitter — t-predictor가 좀 틀려도 깨끗하게 착지

- 목표: 추론은 단 1-step. 출발 시각 `m̂`이 조금 틀려도 결과가 안 망가지게.
- 학습(`meanflow.loss_mr_jitter`, σ = `mr_jitter.sigma` = 0.25):

```
m̂     = clamp(m + N(0, σ))            # 일부러 '틀린 출발시각'을 흉내
target = (source − mixture) / (1 − m̂)  # m̂에서 끝까지 한 번에 가는 평균 속도
pred   = model(mixture, t=m̂, r=1)
loss   = adaptive_L2(pred − target)
```

- 의미: `x1 = mixture + (1−m̂)·(예측 속도)` 가 **어떤 m̂에서도** source에 착지하도록 배웁니다 → t-predictor 오차에 **강건(robust)**.
- σ→0이면 `target = source − background`로 정확히 8-3의 직선(rectified) 속도와 같아집니다(부드러운 일반화).

### 8-6. 1-step 추론 & 검증

- 추론(NFE=1): `x̂_source = mixture + (1 − m̂)·model(mixture, t=m̂, r=1)`, `m̂`은 t-predictor 예측.
  - 모델은 `source_rescaled = clean/m`(1/m배 증폭본)을 향해 학습되므로, 평가 시 출력에 `m̂`을 다시 곱해 자연 레벨로 되돌립니다(SNRi·PESQ 공정 비교용; SI-SNR/STOI는 스케일 불변이라 무관).
- 검증: 배치 평균 t가 아니라 **샘플마다 예측한 `m̂`** 로 1-step 복원해 `val_loss`를 잽니다(실제 추론과 똑같은 조건).

---

### 참고: 폴더 한눈에 보기

```
PNFlowTSE/
├── setup.sh                      # 환경 셋업 (conda env: pnflowtse)
├── run_eval.sh                   # 동봉 체크포인트로 reference 평가 (NFE=1)
├── run_train.sh                  # 재학습: tpred | flow
├── eval_reference_protocol.py    # ★ 논문과 동일한 평가 (추론+채점, 샤딩 지원)
├── aggregate_ref.py              # ★ 샤드 JSON → 논문 행과 나란히 최종 표
├── dump_samples_ref.py           # ★ 들어볼 wav 샘플 저장 (점수와 분리)
├── config/                       # YAML 설정 (flow / t-predicter)
├── checkpoints/                  # 동봉 체크포인트 3종
├── data/                         # LibriSpeech / wham_noise / _test_data (여기에 배치)
├── dataset/                      # reference 평가 데이터셋 (LibriSpeech_single_emb)
├── models/                       # UDiT, ECAPA, t-predicter 모델
├── pn/                           # PN-encoder (espnet2 사용)
├── pndata/ · data/               # 학습용 데이터 믹싱 / 로더
├── eval/                         # (구) our-protocol 평가·집계 스크립트
└── utils/                        # STFT, loss, optimizer 등
```
