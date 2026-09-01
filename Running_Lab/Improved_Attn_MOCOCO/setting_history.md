# Setting History

## 2026-08-26 논문 형평성용 360-only + MR-jitter 전체 파이프라인

기존 Improved Attn 실험은 `train-clean-360`과 `train-clean-100`을 함께 사용했기
때문에 원 논문 baseline과 직접 비교할 수 없다. 기존 관련 Slurm Job은 모두
취소하고, 아래의 새 실험만 최종 비교용으로 사용한다.

- 실험: `20260826_improved_attn_mococo_360only_teacher_decay_mrjitter`
- Stage0: Improved Attn_MOCOCO, fixed anchor Teacher weight `0.1 -> 0.03` cosine decay
- Stage0 학습: `LibriSpeech/train-clean-360`만 사용
- Stage1: 동일 Stage0 Best export를 사용한 Flow Generator, `mr_jitter.enabled=true`, `sigma=0.25`
- t-predictor: 동일 Stage0 Best export와 `train-clean-360`만 사용
- 병렬성: Stage0 성공 후 Flow Stage1과 t-predictor를 `afterok`로 병렬 제출
- 평가: 두 학습 완료 후 Oracle 5K와 t-predictor 5K를 각각 `test` split, 4조건, NFE=1로 실행

`mr_jitter`는 Flow Stage1의 학습 손실 분기이며, t-predictor는 실제 mixing ratio를
회귀하므로 별도 jitter 손실을 적용하지 않는다. 다만 t-predictor는 MR-jitter Flow와
동일한 360-only Stage0 조건에서 학습한다.

제출:

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
bash script/submit_360only_teacher_decay_mrjitter.sh
```

## 2026-08-26 논문 형평성용 train-clean-360-only Stage0

기존 `20260824_improved_attn_mococo_soft03_flow`는 Running_Lab의 초기 공통
설정을 따라 `train-clean-360`과 `train-clean-100`을 함께 사용했다. 그러나
원 논문 baseline README는 학습 split으로 `train-clean-360`만 명시하므로,
논문 수치와의 공정한 비교를 위해 새 실험을 별도로 구성한다.

- 실험: `20260826_improved_attn_mococo_soft03_360only`
- Stage0: Improved Attn_MOCOCO + fixed Teacher weight `0.03`
- 모델 구조, projection, queue, EMA, optimizer, augmentation 및 validation은 기존 Improved Attn Stage0와 동일
- 학습 음성: `LibriSpeech/train-clean-360`만 사용
- 검증/테스트: 기존과 동일하게 `dev-clean`/`test-clean`
- 기존 `360 + 100` 체크포인트는 덮어쓰지 않으며, 새 Stage0에서 export되는 Best와 50-epoch 체크포인트만 최종 공정 비교에 사용

실행:

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
sbatch script/slurm_train_stage0_360only.sh
```

## 2026-08-24 Improved 최종 비교 실험

### 질문
Hugging Face의 `improved-monaural.pt`를 사용하면서 기존 `Attn_MOCOCO +
Soft teacher weight=0.03` Stage0와 Flow Generator Stage1을 구성할 때, 기존
코드와 평가 조건을 어떻게 유지할 것인가?

### 판단
`improved-monaural.pt`는 기존 `proposed-monaural.pt`와 동일한 PN 인코더
구조가 아니다. 전체 USEF-TFGridNet checkpoint에서 `siamese`와
`encoder_head`만 추출하면 개선 모델의 PN enrollment 출력 계약
`[B, 64, T, 65]`를 유지할 수 있다.

### 적용

1. `improved_pn.py`에서 `num_blocks=1`, `layer_num=2`, `refine_layer_num=2`,
   `fusion_shortcut=[0, 1]`, `cut_pos=True`를 사용해 Improved PN 경로를 만들었다.
2. Stage0는 기존 Attn projection, MoCo queue, EMA, speaker-aware masking,
   fixed Teacher loss를 그대로 사용하고 Teacher weight만 `0.03`으로 설정했다.
3. `sia_fm_tse`는 읽기 전용으로 유지하고, `sia_runtime.py`에서 frozen PN
   loader만 모듈 주입 방식으로 교체했다.
4. Flow Decoder는 Improved separator checkpoint를 직접 초기화하지 않고,
   Improved PN embedding을 조건으로 새 Flow Generator를 학습한다. 서로 다른
   Decoder 구조의 weight를 억지로 매핑하지 않기 위한 결정이다.
5. `enroll_exclude_mixture_utt=false`를 설정하고 최종 평가는 변경하지 않은
   `eval_benchmark_paper_metrics_v3.py`를 사용한다. 이는 Base evaluator의
   동일 item 생성과 metric 계산을 보존하기 위한 설정이다.

## 2026-08-24 Teacher Weight Cosine Variant

Teacher weight `0.03`이 가장 좋은 Stage0 sweep 결과를 보였지만, 학습 초기의
anchor 안정화 효과를 버리지 않기 위해 별도 비교군을 추가했다. 새 variant는
optimizer step 기준으로 `0.1 -> 0.03` cosine decay를 사용한다. Encoder 깊이,
projection 크기, queue, EMA, optimizer 및 Stage1 decoder는 고정 실험과
동일하게 유지한다. Stage1은 새 Stage0 Best export를 사용하며, 로그와
checkpoint는 `20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03` 아래에
분리한다.
## 2026-08-27 Improved_Attn epoch-85 Best Stage1

현재 Stage0가 epoch 85까지 도달한 시점의 `improved_attn_360only_teacher_decay_best.ckpt`
및 exported `pn_encoder_best.pt`를 고정해 별도 Stage1을 구성한다. Stage0 학습이
이후 계속되어도 Stage1 입력 checkpoint는 변경되지 않는다.

- Stage0 source: `exp/20260826_improved_attn_mococo_360only_teacher_decay_mrjitter/stage0_moco/checkpoints/pn_encoder_best.pt`
- Frozen copy: `exp/20260827_improved_attn_85ep_best_mrjitter/stage0_input/pn_encoder_best_at_epoch85.pt`
- Stage1: `sia_fm_tse` Flow Generator, 360-only, `loss.gamma=0.0`
- MR-jitter: `enabled=true`, `sigma=0.25`
- DDP: one node, 4 A10 GPUs
