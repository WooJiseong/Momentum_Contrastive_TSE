# 20260803 Stage0 50 Epoch Soft-leakage Test

이 폴더는 Stage0 50 epoch encoder checkpoint를 고정하고 Soft-leakage Stage1을
독립적으로 학습 및 평가하는 실험 베이스다.

## Required Input

Stage0 50 epoch 완료 직후 export된 `pn_encoder_last.pt`를 다음 파일명으로 복사한다.

```text
stage0_moco/checkpoints/pn_encoder_50ep.pt
```

`last.ckpt` 또는 `indiv_moco_best.ckpt` 같은 Lightning 전체 checkpoint는 이 경로에
사용하지 않는다. Stage1 encoder override는 `encoder.*`, `encoder_head.*`만 포함한
encoder export 형식을 요구한다.

Stage1은 `proposed-monaural.pt`에서 새로 초기화하며, 위 encoder만 override한다.
따라서 다른 Stage0 checkpoint와 비교할 때도 Stage1 학습량과 초기값이 섞이지 않는다.

## Run

```bash
cd contrastive_momentum/Running_Lab/Soft_MOCOCO
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/train_stage1_softleakage_20260803.sh all
```

Slurm 제출은 다음과 같다.

```bash
sbatch script/slurm_stage1_softleakage_20260803.sh
```

산출물은 `stage1_tfgridnet/`, `evaluation/`, `environment.txt`에 저장한다.
