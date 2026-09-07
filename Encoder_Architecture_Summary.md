# Soft_MOCOCO Stage0 Encoder Architecture Summary

이 문서는 현재 저장소에서 구현되어 실행되는 `Soft_MOCOCO`의 Stage0 인코더와
학습 목적함수를 논문 작성에 사용할 수 있는 수준으로 정리한 것이다.

## 1. Stage0의 목적

Stage0는 혼합 음성에서 화자 조건을 추출하는 Positive-Negative(`PN`) encoder를
학습하는 단계다. positive와 negative enrollment를 함께 처리하여 target
화자와 방해 화자를 구분하는 시간-주파수 조건 표현을 만든다. 이 조건 표현은
후속 Stage1 음성 디코더가 사용하며, Stage0 자체는 음성을 복원하는 디코더가
아니다.

`Soft_MOCOCO`는 다음 세 요소를 결합한다.

```text
PN encoder와 PN fusion head
        +
MoCo 계열의 momentum key encoder와 negative queue
        +
초기 표현을 보존하는 fixed Teacher consistency loss
```

여기서 `Soft`는 Teacher 표현을 hard target으로 복사하는 것이 아니라, 초기
표현과의 cosine consistency 항을 전체 목적함수에 가중하여 추가한다는 의미다.

## 2. 입력 데이터와 용어

현재 기본 설정은 `16 kHz`, `3 seconds` segment이므로 한 waveform segment는
일반적으로 `48000` sample이다. 온라인 mixer가 target source, positive/negative
enrollment 및 혼합 관련 음성을 생성한다.

| 기호 또는 코드 키 | 의미 |
|---|---|
| `source` | target 화자의 clean source waveform. 기본 설정에서 positive key로 사용된다. |
| `pos_wave` | query branch에 입력되는 positive waveform |
| `neg_wave` | query branch에 입력되는 negative waveform |
| `pos_key` | key branch의 positive 입력. `positive_key: source`이면 `source`다. |
| `neg_key` | key branch의 negative 입력. 현재 pair 생성에서는 `q_neg`가 사용된다. |
| enrollment | 화자 정보를 제공하는 음성 구간 |
| target/source | 추출 대상인 target 화자 음성 |
| positive | query와 같은 target 화자에 대응하는 조건 |
| negative | query가 구분해야 하는 다른 화자 또는 반대 PN pair |
| dense embedding | 시간 및 주파수 위치를 유지하는 encoder 출력 |
| global embedding | dense embedding을 pooling한 contrastive용 고정 길이 벡터 |
| condition | Stage1 디코더에 전달되는 PN encoder 표현 |

`pos_wave`와 `neg_wave`의 실제 화자 구성은 온라인 mixer와 `source_num`,
`enroll_num`, `active_num` 설정으로 결정된다. 그러므로 변수명만으로 실제
화자 수를 추정하지 말고 `train_moco_encoder.py`의 `_pairs`를 기준으로 해석해야
한다.

## 3. 전체 텐서 흐름

```text
pos_wave [B, 1, T] ──┐
                     ├─ shared TFGridNet encoder ──┐
neg_wave [B, 1, T] ──┘                             ├─ PN fusion head
                                                   └─ dense condition h_q

source [B, 1, T] ───┐
neg_key [B, 1, T] ──┘ ── momentum encoder/head ── dense key h_k+

h_q ── temporal pooling + projection MLP ── L2 normalize ── query q
h_k+ ── temporal pooling + projection MLP ── L2 normalize ── positive key k+

previous momentum keys ── queue ── normalized negative candidates
```

`q`와 `k+`는 contrastive loss를 위한 global vector다. `h_q`와 이에 대응하는
dense representation은 Teacher consistency loss 및 후속 Stage1 조건에 사용된다.
따라서 contrastive projection의 `256`차원 벡터가 Stage1 조건 전체를 대체하는
것은 아니다.

## 4. PN encoder 본체

### 4.1 공유 TFGridNet encoder

`PNEncodePath`는 `TFGridNet_encoder`와 `GridNetBlock_attnhead`로 구성된다.
positive와 negative 입력은 서로 다른 파라미터를 가진 두 encoder가 아니라 동일한
`self.encoder`를 두 번 통과한다. 이는 shared-weight Siamese 구조다.

입력 waveform은 내부적으로 `[B, T, 1]`로 변환되어 encoder에 전달된다. 현재
설정에서는 encoder feature가 일반적으로 `[B, 64, T_encoder, 65]` 형태다.
`64`는 channel 차원, `65`는 frequency feature 차원이며 `T_encoder`는 encoder의
stride를 거친 시간 위치 수다.

### 4.2 PN fusion head

encoder가 positive와 negative dense feature를 각각 만들면
`encoder_head(pos_emb, neg_emb)`가 두 feature를 결합해 PN condition을 만든다.
이 head의 attention fusion은 contrastive projection head에서 수행하는 temporal
mean pooling과 다른 연산이다.

shape 관점에서 `pos_emb`와 `neg_emb`가 각각
`[B, C, T_pos, F]`, `[B, C, T_neg, F]`라면 fusion head는 먼저 시간축으로
이어 붙여 `[B, C, T_pos + T_neg, F]`로 만든다. attention은 결합된 시간 위치들
사이에서 수행되고 주파수축 `F`는 유지된다. 이후 구현의
`cond[:, :, :pos_emb.shape[2]]`에 의해 앞쪽 `T_pos`개 시간 위치만 반환되므로
최종 dense PN condition은 `[B, C, T_pos, F]`다.

즉, 최종 condition은 시간축과 주파수축을 모두 보존한다. 이 PN encoder와
fusion head의 dense condition 생성 방식은 기존 Baseline의 `model.encode()`
경로에도 동일하게 존재한다. `Soft_MOCOCO`의 차별점은 이 backbone 출력에
projection head, momentum contrastive objective, queue 및 fixed Teacher
consistency를 추가하여 student encoder를 최적화한다는 점이다.

### 4.3 Stage1과의 관계

4.3은 새로운 Stage0 구조가 아니라 Stage1에서의 사용 방식을 설명한다. Stage0의
student encoder와 PN fusion head가 만드는 dense condition은 Stage1 디코더의
조건으로 사용된다. 반면 contrastive projection head는 Stage0의 화자 구분
학습을 위한 head다.

```text
dense PN condition  : 시간-주파수 구조를 보존하며 Stage1 조건으로 사용
256-D projection    : contrastive loss 계산을 위한 global speaker vector
```

## 5. Contrastive projection head

`ContrastiveProjectionHead`는 dense representation을 contrastive learning에
적합한 고정 길이 벡터로 변환한다. 현재 기본 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| 입력 channel | `64` |
| 입력 frequency | `65` |
| hidden dimension | `2048` |
| projection dimension | `256` |
| MLP layer 수 | `3` |
| BatchNorm | `False` |
| temporal pooling | 시간축 평균 |

코드상 연산은 다음과 같다.

```text
emb [B, C, T, F]
  → mean over T
  → [B, C, F] flatten
  → Linear(64×65, 2048)
  → GELU
  → Linear(2048, 2048)
  → GELU
  → Linear(2048, 256)
  → L2 normalization
```

dense feature를 `h ∈ R^(C×T×F)`라 하면

\[
u = \operatorname{vec}\left(\frac{1}{T}\sum_{t=1}^{T} h_{:,t,:}\right),
\]

\[
z = \frac{g_\theta(u)}{\|g_\theta(u)\|_2} \in \mathbb{R}^{256}.
\]

여기서 `g_theta`는 projection MLP다. 시간축 평균은 projection head 안에서만
수행되며, Stage1에 전달되는 dense condition 전체를 시간 평균으로 축약한다는
뜻은 아니다. L2 normalization으로 벡터 방향이 주요 contrastive 신호가 되며,
정규화된 벡터의 내적은 cosine similarity와 같다.

## 6. Momentum encoder와 MoCo objective

### 6.1 Student와 momentum branch

student branch는 gradient descent로 직접 업데이트된다. momentum branch는
gradient를 받지 않고 student를 지수이동평균(EMA)으로 추적한다.

\[
\phi \leftarrow m\phi + (1-m)\theta,
\]

여기서 `theta`는 student 파라미터, `phi`는 momentum encoder와 momentum
projection head 파라미터다. 기본 설정은 `ema_momentum_base=0.997`,
`ema_momentum_final=1.0`이며 학습 step에 따라 cosine schedule된다. momentum
branch의 buffer도 업데이트 시 student branch에서 복사된다.

momentum encoder/head는 `eval()` 상태이며 `requires_grad=False`다. student
encoder, PN fusion head, student projection head는 optimizer로 학습된다.

### 6.2 실제 positive와 explicit negative pair

기본 설정의 `positive_key`는 `source`다. `_pairs`가 만드는 pair는 다음과 같다.

```text
query pair       : (q_pos, q_neg) = (pos_wave, neg_wave)
positive key     : (pos_key, neg_key) = (source, q_neg)
explicit negative: (neg_key, pos_key) = (q_neg, source)
```

positive와 explicit negative는 단순 waveform 비교가 아니라 PN encoder에 입력되는
두 음성의 역할을 바꾼 pair다. 이 구조 때문에 코드에서
`negative_pos=[neg_key]`, `negative_neg=[pos_key]`가 사용된다.

positive key는 `detach()`되어 student gradient가 momentum branch로 흐르지 않는다.
query는 student branch에서 계산되어 gradient를 받는다.

### 6.3 InfoNCE loss

query projection을 `q`, positive key를 `k+`, 모든 negative 후보를 `{k_i^-}`라
하면 logits는

\[
\ell_+ = \frac{q^\top k_+}{\tau}, \qquad
\ell_i^- = \frac{q^\top k_i^-}{\tau},
\]

\[
\mathcal{L}_{\mathrm{MoCo}}
 = -\log
 \frac{\exp(\ell_+)}
 {\exp(\ell_+) + \sum_i \exp(\ell_i^-)}.
\]

구현에서는 positive logit을 첫 번째 열에 놓고 label을 항상 `0`으로 지정하여
`cross_entropy`를 계산한다. 기본 temperature는 `τ=0.10`이다.

### 6.4 Negative queue와 speaker-aware masking

기본 `queue_size=4096`이다. 각 batch의 momentum negative key를 queue에 넣고,
이전 iteration의 key를 다음 iteration의 negative 후보로 재사용한다. 이로써
매 step마다 대규모 negative를 새로 인코딩하지 않고도 많은 표현을 분모에 유지한다.

queue에는 현재 query와 같은 화자가 포함될 수 있다. 이런 항은 false negative다.
현재 구현은 `queue_speaker_ids`를 저장하고 `speaker_aware_queue`가 켜져 있으면
query와 같은 화자의 queue 항을 logits에서 제외한다. 이 masking은 학습 단계의
queue logits에 적용된다.

validation에서는 `train=False`로 계산하므로 queue를 사용하지 않고 갱신하지도
않는다. explicit negative pair의 loss는 계산하지만, validation 결과가 학습
batch의 queue 상태에 의존하지 않도록 구성되어 있다.

## 7. Soft Teacher Anchor

### 7.1 Fixed Teacher의 정의

`SoftMomentumContrastivePNLearner`는 초기 checkpoint에서 생성한 student encoder를
deep copy하여 `self.teacher`를 만든다. Teacher는 이후 학습에서 갱신되지 않으며
항상 `eval()` 및 `no_grad()`로 실행된다.

```text
Teacher 초기화 : 초기 proposed-monaural.pt에서 복사
학습 중 업데이트 : 없음
역할           : 초기 PN 표현을 보존하는 고정 Anchor
```

Teacher는 momentum encoder와 별개의 네트워크다.

```text
fixed Teacher    : 초기 checkpoint에 고정, consistency 기준
momentum encoder : student를 EMA로 추적, contrastive key 생성
student encoder  : gradient descent로 학습, 최종 Stage0 encoder
```

### 7.2 Teacher consistency loss

query pair의 student dense condition을 `h_s`, fixed Teacher condition을 `h_t`라
하면 channel 축(`dim=1`)으로 L2 normalize한 뒤 cosine distance를 평균낸다.

\[
\bar h_s = \frac{h_s}{\|h_s\|_2}, \qquad
\bar h_t = \frac{h_t}{\|h_t\|_2},
\]

\[
\mathcal{L}_{\mathrm{teacher}}
 = \operatorname{mean}\left(1 -
 \sum_c \bar h_{s,c}\bar h_{t,c}\right).
\]

전체 Stage0 objective는

\[
\mathcal{L}_{\mathrm{total}}
 = \mathcal{L}_{\mathrm{MoCo}}
 + \lambda_{\mathrm{teacher}}
   \mathcal{L}_{\mathrm{teacher}}.
\]

기본 `teacher_loss_weight`는 `0.1`이다. `0.03`, `0.1`, `0.3` ablation은 이
`lambda_teacher`만 변경한 비교다. Teacher loss에는 256차원 projection vector가
아니라 PN fusion head의 dense condition이 사용된다.

## 8. DINO 및 MoCo v3와의 관계

### DINO와의 유사점

- student와 teacher의 표현 일관성 loss를 사용한다.
- teacher 표현은 gradient target으로 취급하지 않는다.
- teacher-student alignment로 student 표현의 급격한 이동을 완화한다.

### DINO와 다른 점

- 현재 Teacher는 EMA Teacher가 아니라 초기 checkpoint에 고정된 Teacher다.
- DINO의 center/sharpening, teacher temperature, multi-crop global/local view를
  구현하지 않는다.

따라서 논문에서는 “DINO를 구현했다”보다 “DINO의 teacher-student consistency
개념에서 영감을 받은 fixed Teacher Anchor”라고 표현하는 것이 정확하다.

### MoCo 및 MoCo v3와의 유사점

- momentum encoder가 key representation을 생성한다.
- query/key projection head와 L2-normalized representation을 사용한다.
- temperature-scaled similarity 기반 contrastive objective를 사용한다.
- 다수의 negative representation을 사용한다.

### MoCo 계열 구현과 다른 점

- 현재 코드는 classic MoCo 방식의 queue를 사용한다. MoCo v3의 대표 설정은
  queue를 사용하지 않으므로 queue까지 “MoCo v3와 동일”하다고 쓰면 안 된다.
- positive/negative는 일반적인 image augmentation pair가 아니라 PN enrollment
  pair와 `source` 기반 key pair다.
- fixed Teacher consistency loss가 contrastive loss에 추가되어 있다.

따라서 `Soft_MOCOCO`는 MoCo 계열 momentum contrastive learning과 MoCo v3
계열 projection/normalized representation 설계를 PN speech encoder에 적용하고,
DINO-inspired fixed Teacher consistency를 추가한 방법으로 기술하는 것이
적절하다.

## 9. Optimization과 augmentation

학습 대상은 student PN encoder, PN fusion head, student projection head다.
momentum encoder/head와 fixed Teacher는 optimizer에 직접 포함되지 않는다.

기본 optimizer는 `AdamW`이며 encoder learning rate는 base learning rate에
`encoder_lr_scale`를 곱해 별도로 조정한다. 기본 설정은 `lr=1e-4`,
`encoder_lr_scale=0.03`, `weight_decay=0.04`와 warmup/cosine schedule이다.

waveform augmentation은 다음 요소를 포함할 수 있다.

- random gain
- 상대 RMS에 비례한 noise
- temporal zero mask
- circular time shift

기본 구현에서는 gain은 `_augment_wave` 호출 시 적용되고, strong 설정에 따라
noise, mask, shift의 강도가 달라진다. query에는 기본적으로 strong augmentation,
key에는 기본적으로 non-strong 설정을 적용하지만 key에도 gain이 적용될 수 있다.

## 10. Train과 validation의 의미

학습 시에는 query를 student encoder로 계산하고 positive/explicit negative key를
momentum encoder로 계산한다. queue와 speaker-aware mask로 MoCo loss를 계산한
뒤, fixed Teacher와 student dense condition의 Teacher loss를 더해 optimizer를
업데이트한다. optimizer update 후 momentum encoder/head를 EMA 갱신한다.

검증 시에는 동일한 objective의 수치 정의를 사용하지만 augmentation 없이
계산하며 queue를 사용하거나 갱신하지 않는다. TensorBoard의 `val_loss`,
`val_moco_loss`, `val_teacher_loss`는 Stage0 representation objective의
검증값이며 Stage1 음성의 `SI-SDR`, `SI-SDRi`, `SNR`와 직접 같은 지표가 아니다.

## 11. Checkpoint와 재현성

`last.ckpt`는 Lightning module 전체 상태를 저장하므로 optimizer, scheduler,
global step, epoch 등과 함께 학습을 재개하는 데 사용한다. `pn_encoder_best.pt`
같은 export checkpoint는 주로 student encoder와 PN fusion head를 Stage1에서
초기화하기 위한 가중치다.

논문 재현을 위해 초기 checkpoint hash, encoder depth, head depth, projection
차원, temperature, queue 설정, EMA schedule, teacher weight, masking, augmentation,
dataset, segment 길이, sample 수, global batch, accumulation, DDP world size,
seed와 resolved YAML을 함께 기록해야 한다.

`last.ckpt`에서 재개한 실험과 `pn_encoder_best.pt`로 새로 시작한 Stage1은
optimizer 상태와 학습 step이 다르므로 동일한 실험으로 취급하면 안 된다.

## 12. 논문용 요약 문단

> We train a Positive-Negative speaker encoder in Stage0 using a Soft_MOCOCO
> objective. Positive and negative enrollment waveforms are processed by a
> shared TFGridNet-based encoder and fused by a PN fusion head to produce a
> dense time-frequency conditioning representation. For contrastive learning,
> the dense representation is temporally pooled and projected to a normalized
> 256-dimensional vector. A momentum encoder, updated by exponential moving
> average, generates stable key representations, while a queue provides
> additional negative candidates. Same-speaker queue entries are excluded when
> speaker-aware masking is enabled. The contrastive InfoNCE loss is combined
> with a cosine consistency loss between the student dense representation and a
> frozen copy of the initial pretrained encoder. This frozen network acts as a
> representation anchor and is distinct from the EMA momentum encoder. The
> resulting student PN encoder and fusion head are exported for conditioning the
> Stage1 speech decoder.

## 13. 코드 기준 위치

- `Running_Lab/PN_MOCOCO/pn_mococo/moco_encoder.py`
  - PN encoder path, 입력 shape 변환, projection head, temporal pooling
  - momentum encoder, EMA, queue, InfoNCE loss
- `Running_Lab/PN_MOCOCO/train_moco_encoder.py`
  - waveform augmentation, query/key pair 생성, train/validation step, EMA 호출
- `Running_Lab/Soft_MOCOCO/pn_soft_mococo/soft_moco.py`
  - fixed Teacher Anchor와 Teacher consistency loss
- `Running_Lab/Soft_MOCOCO/train_moco_encoder.py`
  - MoCo loss와 Teacher loss 결합, Stage0 logging 및 validation
- `Running_Lab/Soft_MOCOCO/configs/config_soft_moco.yaml`
  - encoder, optimizer, data, augmentation, queue, Teacher 기본 설정

`positive`, `negative`, `key`, `teacher`, `momentum`은 코드에서 서로 다른
역할을 가질 수 있다. 변수명만으로 판단하지 말고 `_pairs`, `encode_fused_grad`,
`contrastive_loss`, `teacher_loss`의 호출 순서를 함께 확인해야 한다.
