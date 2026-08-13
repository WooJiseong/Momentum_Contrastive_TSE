# Soft_Indiv_MOCOCO

## Hypothesis

Individual negative enrollment MoCo에 frozen initial-PN Teacher embedding 보존 항을 더하면,
개별 negative와의 분리력을 유지하면서 original PN decoder와 호환되는 condition embedding을
더 잘 보존할 수 있다.

## Baseline

- `Running_Lab/PN_Indiv_MOCOCO`: individual-negative MoCo CE만 사용한다.
- `Running_Lab/Soft_MOCOCO`: aggregate-negative MoCo에 frozen Teacher cosine term을 추가한다.

## Main Difference

- Stage0: individual-negative MoCo CE에 `0.1 * frozen Teacher cosine loss`를 추가한다.
- Stage1: `negative SI-SDR + 0.1 * target-orthogonal leakage loss`를 사용한다.
- Leakage loss는 negative enrollment가 아니라 online mixer의 실제 `sample[1:]` individual
  nuisance waveform `[B, J, T]`를 사용한다.

## Controlled Conditions

- PN_Indiv_MOCOCO와 동일한 online LibriSpeech + WHAM split 및 sample 설정
- batch size 2, gradient accumulation 4, AdamW, 4 GPU
- Stage0 300 epochs, Stage1 200 epochs, Stage1 fixed-count evaluation
- Stage0 teacher loss weight `0.1`, Stage1 leakage loss weight `0.1`

## Stage0 Checkpoint Sweep

Stage0는 매 50 completed epochs마다 다음 두 파일을 저장한다.

```text
stage0_moco/checkpoints/stage0_epoch_050.ckpt
stage0_moco/checkpoints/pn_encoder_epoch_050.pt
```

`.ckpt`는 Lightning model, optimizer, callback, loop 및 RNG 상태를 포함해 재개용으로
사용한다. `.pt`는 Stage1 encoder override 전용 export다. 별도로
`stage0_moco/resume_state/`에는 optimizer snapshot, seed, RNG snapshot을 보관한다.

## Run

새 전체 실행:

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_Indiv_MOCOCO
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29840 \
  bash script/train_eval_20260804.sh all
```

Epoch 50 encoder의 Stage1 및 evaluation sweep:

```bash
REUSE_RUN=1 STAGE0_SELECTOR=50 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash script/train_eval_20260804.sh stage1

REUSE_RUN=1 STAGE0_SELECTOR=50 \
  bash script/train_eval_20260804.sh eval
```

Stage0 재개:

```bash
REUSE_RUN=1 RESUME_STAGE0=exp/20260804_soft_indiv_mococo/stage0_moco/checkpoints/stage0_epoch_050.ckpt \
  NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash script/train_eval_20260804.sh moco
```

Stage1 재개:

```bash
REUSE_RUN=1 STAGE0_SELECTOR=50 \
RESUME_STAGE1=exp/20260804_soft_indiv_mococo/stage1_sweep/stage0_epoch_050/checkpoints/last.ckpt \
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash script/train_eval_20260804.sh stage1
```

## Slurm

Slurm이 output 경로를 열 수 있도록 제출 전 기본 exp 폴더를 만든다.

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_Indiv_MOCOCO
mkdir -p exp/20260804_soft_indiv_mococo
sbatch script/slurm_train_eval_20260804.sh
```

## Negative `1/n` Ablation

비교군은 Soft_Indiv_MOCOCO의 다른 설정을 유지하고, individual negative 집계에서만
`logsumexp(negative_logits - log(n))`의 `-log(n)` 항을 제거한다. 따라서 벡터 정규화는
그대로 적용되지만 negative 수에 따른 합산 효과는 남는다.

```text
exp/20260806_soft_indiv_mococo_no_1n/
└── stage0_moco/
    └── checkpoints/
        ├── soft_indiv_moco_no_1n_best.ckpt
        ├── stage0_epoch_050.ckpt
        ├── stage0_epoch_100.ckpt
        ├── stage0_epoch_150.ckpt
        ├── stage0_epoch_200.ckpt
        ├── stage0_epoch_250.ckpt
        └── stage0_epoch_300.ckpt
```

각 50 Epoch checkpoint에는 Lightning optimizer/loop/RNG 상태가 포함되고,
`resume_state/`에는 seed와 optimizer/RNG snapshot이 별도로 저장된다.

제출 명령:

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_Indiv_MOCOCO
mkdir -p exp/20260806_soft_indiv_mococo_no_1n
sbatch script/slurm_train_no_1n_20260806.sh
```
