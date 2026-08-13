# Soft_Indiv_MOCOCO Negative `1/n` Ablation

이 실험은 `20260804_soft_indiv_mococo`와 동일한 Soft Teacher individual-negative
contrastive 학습에서 negative 집계의 `1/n` 평균화만 제거하는 비교군이다.

## Objective

기존 조건:

```text
logsumexp(negative_logits - log(n))
```

비교 조건:

```text
logsumexp(negative_logits)
```

Positive/negative embedding의 L2 normalization, frozen Teacher cosine term, EMA momentum
encoder, 데이터셋, seed, optimizer, batch size, precision 및 300 Epoch 설정은 동일하다.
Embedding normalization은 각 similarity의 범위를 제한하지만, 비교 조건에서는 negative
개수에 따른 `log(n)` 합산 효과가 남는다.

## Outputs

```text
stage0_moco/
├── config_stage0_runtime.yaml
├── train.log
├── checkpoints/
│   ├── soft_indiv_moco_no_1n_best.ckpt
│   ├── last.ckpt
│   ├── stage0_epoch_050.ckpt
│   ├── stage0_epoch_100.ckpt
│   ├── stage0_epoch_150.ckpt
│   ├── stage0_epoch_200.ckpt
│   ├── stage0_epoch_250.ckpt
│   ├── stage0_epoch_300.ckpt
│   ├── pn_encoder_best.pt
│   ├── pn_encoder_epoch_050.pt
│   └── ...
└── resume_state/
    ├── stage0_last_state.pt
    └── stage0_epoch_050_state.pt
```

각 `.ckpt`에는 Lightning optimizer/loop/RNG 상태가 포함되며, `resume_state/`에는 seed와
optimizer/RNG snapshot을 별도로 기록한다.

## Submit

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_Indiv_MOCOCO
sbatch script/slurm_train_no_1n_20260806.sh
```

재개 시에는 기존 실행 보호 규칙에 따라 다음처럼 명시한다.

```bash
REUSE_RUN=1 \
RESUME_STAGE0=exp/20260806_soft_indiv_mococo_no_1n/stage0_moco/checkpoints/stage0_epoch_050.ckpt \
bash script/train_eval_20260804.sh moco
```
