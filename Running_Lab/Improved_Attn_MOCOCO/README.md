# Improved_Attn_MOCOCO

`20260824_improved_attn_mococo_soft03_flow`는 Hugging Face의
`ShitongXu/TSE-Pos-Neg-Enroll` 저장소에서 제공하는 `improved-monaural.pt`를
초기 화자 인코더로 사용하는 최종 비교 실험이다.

## 실험 구성

- Stage0: Improved USEF-TFGridNet의 `siamese + encoder_head`를 strict 로드
- Stage0 학습: `Attn_MOCOCO`의 40-frame Lightweight Attention Pooling
- Soft anchor: 초기 Improved PN 경로를 고정 Teacher로 복사하고 `teacher_loss_weight=0.03`
- Queue: 기존 speaker-aware MoCo queue와 동일
- Stage1: `sia_fm_tse`의 Flow Generator와 기본 Rectified Flow Loss
- Stage1 Oracle: 실제 `mixing_ratio`로 `m -> 1` Euler 적분, validation NFE=1
- t-predictor: 같은 Best Stage0 체크포인트를 고정 조건으로 학습

## Teacher Weight Cosine Variant

`20260824_improved_attn_mococo_teacher_cosine_0p1_to_0p03`은 Teacher weight를
초기 `0.1`에서 최종 `0.03`까지 optimizer step 기준 cosine decay한다.

```text
w(p) = 0.03 + 0.5 * (0.1 - 0.03) * (1 + cos(pi * p))
```

여기서 `p`는 전체 optimizer step에 대한 진행률이다. 기존 `0.03` 고정
실험은 그대로 보존하며, 두 실험 모두 `train_teacher_weight`와
`val_teacher_weight`를 TensorBoard에 기록한다.

개선 모델 체크포인트는 전체 USEF separator이므로 Flow UDiT의 가중치로 직접
로드하지 않는다. 개선 체크포인트의 `siamese`와 `encoder_head`만
`improved_pn.py`에서 추출하여 Stage0와 Flow의 frozen enrollment encoder로
사용한다. Flow Decoder는 별도로 새로 학습한다.

## 경로 규칙

모든 로그, TensorBoard event, checkpoint, 평가 결과는 다음 실험 폴더 아래에
생성된다.

```text
exp/20260824_improved_attn_mococo_soft03_flow/
├── stage0_moco/
├── stage1_flow_oracle/
└── tpred/
```

Base와 `sia_fm_tse`는 수정하지 않는다. `sia_runtime.py`가 SIA의 원본 학습
코드를 import하면서 `pn_encoder` 이름만 Improved 전용 loader로 주입한다.

## 제출

새 Lab 디렉터리에서 실행한다.

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
bash script/submit_final_experiment.sh
```

이 명령은 Stage0를 먼저 제출하고, Stage0가 `afterok`로 종료된 뒤 Oracle Flow와
t-predictor를 각각 제출한다. Stage0 Best export는 다음 경로를 사용한다.

```text
exp/20260824_improved_attn_mococo_soft03_flow/stage0_moco/checkpoints/pn_encoder_best.pt
```

상태 확인:

```bash
squeue -u "$USER" -o "%.18i %.12P %.32j %.2t %.10M %.6D %R"
```

## Base 동일 평가

최종 5000 test 평가는 수정하지 않은
`sia_fm_tse/eval/eval_benchmark_paper_metrics_v3.py`를 호출한다. 이 평가기는
Base evaluator의 고정 test item 생성, polarity correction, SI-SDR/SI-SNR/SNR,
PESQ, STOI 계산 및 NFE=1 Euler 규칙을 그대로 사용한다.

```bash
python eval_improved_flow.py \
  --config configs/config_flow_oracle.yaml \
  --stage0-ckpt exp/20260824_improved_attn_mococo_soft03_flow/stage0_moco/checkpoints/pn_encoder_best.pt \
  --flow-ckpt exp/20260824_improved_attn_mococo_soft03_flow/stage1_flow_oracle/checkpoints/improved_attn_flow_oracle_best.ckpt \
  --n 5000 \
  --t-mode oracle \
  --out exp/20260824_improved_attn_mococo_soft03_flow/stage1_flow_oracle/evaluation/test5000_oracle.json
```

`eval_improved_flow.py`는 reference evaluator를 복사하지 않고 import하며, 실험
체크포인트에 맞는 Improved PN loader만 주입한다. 따라서 metric 계산 코드와
데이터 생성 규칙의 비교 가능성을 보존한다.

Scheduled Stage0와 종속 Flow/t-predictor job을 제출하려면:

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Improved_Attn_MOCOCO
bash script/submit_teacher_decay_experiment.sh
```

이 variant는 기존 `config_flow_oracle.yaml`과 `config_tpred.yaml`의 학습
규칙을 유지하고, `--run-root`로 로그와 checkpoint만 별도 experiment 폴더에
분리한다.
