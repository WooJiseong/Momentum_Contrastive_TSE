# Soft_MOCOCO

`Soft_MOCOCO`는 기존 PN_MOCOCO의 momentum contrastive 학습에 Teacher embedding
보존 loss를 추가한 안정화 실험이다.

## Objective

```text
L_total = L_MoCo + teacher_loss_weight * L_teacher
```

Teacher는 `proposed-monaural.pt`에서 초기화한 frozen PN encoder/head이다.
Student는 동일한 Teacher 초기점에서 시작해 MoCo objective로 업데이트된다.
`L_teacher`는 student와 frozen Teacher의 dense PN condition embedding을 cosine
distance로 맞추어, contrastive 학습이 기존 decoder-compatible representation을
갑자기 훼손하지 않도록 한다.

기본값 `teacher_loss_weight=0.1`은 MoCo update를 주 objective로 유지하면서 Teacher
representation에서 천천히 벗어나도록 하는 보수적인 시작점이다. 실제 sweep에서는
`0.0`, `0.01`, `0.1`, `1.0`을 비교하는 것을 권장한다.

Stage0는 PN_MOCOCO의 speaker-aware queue를 재사용한다. Online mixer가 보존한
`target_spk_id`를 queue embedding과 함께 저장하고, 알려진 동일 화자 항목은 false
negative가 되지 않도록 제외한다. 이 변경은 frozen Teacher imitation term과 독립적이다.

## Run

```bash
cd contrastive_momentum/Running_Lab/Soft_MOCOCO
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train.sh moco
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train.sh tfgridnet
```

Stage1은 기존 PN_MOCOCO supervised TFGridNet trainer를 사용하며 Soft_MOCOCO Stage0
export를 encoder override checkpoint로 사용한다.

## Stage0 Checkpoint Sweep

Stage0 50 epoch export를 Soft-leakage Stage1으로 독립 검증하는 실험 베이스는
`exp/20260803_stage0_50ep_softleakage_test/`에 준비되어 있다. Stage0 50 epoch 완료
직후의 `pn_encoder_last.pt`를 `pn_encoder_50ep.pt`로 복사한 뒤 다음 명령으로 Stage1과
고정 수 평가를 실행한다. Lightning 전체 checkpoint인 `last.ckpt`는 사용할 수 없다.

```bash
cd contrastive_momentum/Running_Lab/Soft_MOCOCO
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/train_stage1_softleakage_20260803.sh all
```

Slurm 제출:

```bash
sbatch script/slurm_stage1_softleakage_20260803.sh
```

이 Stage1 leakage loss는 negative enrollment나 합산 `background`를 사용하지 않는다.
online mixer가 만든 동일한 mixture의 `sample[1:]`을 `[B, J, T]` `nuisance_sources`로
보존해 source별로 계산하며, 첫 batch에서 이들의 합이 `mixture - source`인지 검증한다.

## Stage0 50 Epoch Without Leakage Loss

Soft-leakage loss가 Stage1 안정성에 미치는 영향을 분리하기 위해, 동일한 Stage0
50 epoch encoder export로 SI-SDR만 학습하는 대조 실험은
`exp/20260803_stage0_50ep_soft_wo_leakage_test/`에 둔다. 이 실험은 leakage loss를
weight 0으로 계산하는 방식이 아니라, 해당 loss를 호출하지 않는 표준 TFGridNet
trainer를 사용한다.

```bash
cd contrastive_momentum/Running_Lab/Soft_MOCOCO
sbatch script/slurm_stage1_wo_leakage_20260803.sh
```
