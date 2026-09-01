# data_robust_debug

`Soft_MOCOCO` Stage0의 train/validation loss 변동 원인을 재현하기 위한 데이터셋
비교 실험이다. 두 실험은 `proposed-monaural.pt`에서 새로 시작하며 Teacher Weight는
`0.1`로 고정한다.

## 실험

| 실험 | 학습 데이터 | 초기화 | Teacher Weight |
|---|---|---|---|
| `20260831_soft_mococo_360plus100_fixed_teacher0p1` | train-clean-360 + train-clean-100 | proposed-monaural.pt | 0.1 fixed |
| `20260831_soft_mococo_360only_fixed_teacher0p1` | train-clean-360 | proposed-monaural.pt | 0.1 fixed |

모델 구조, augmentation, optimizer, scheduler, seed, batch size, validation set은
두 실험에서 동일하다. 기존 `soft_gt_252ep`는 resume된 checkpoint이므로 직접 재현값이
아닌 TensorBoard 비교 기준으로만 사용한다.

## 실행

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum
sbatch Running_Lab/data_robust_debug/script/slurm_stage0_360plus100.sh
sbatch Running_Lab/data_robust_debug/script/slurm_stage0_360only.sh
```

두 job은 표준 `Soft_MOCOCO/train_soft_moco_encoder.py`를 사용한다. 따라서 최근
teacher-decay 실험의 수동 epoch aggregation은 사용하지 않고 Lightning의 표준
sample-weighted aggregation을 사용한다.

## TensorBoard

46009 포트에는 다음 세 run을 표시한다.

- `data_robust_360plus100`
- `data_robust_360only`
- `soft_gt_252ep`

로그는 각 실험의 `stage0_moco/lightning_logs/0`에 저장된다.

## 결과

실험 완료 후 `exp/RESULTS.md`에 best epoch, best validation loss, train/validation
곡선의 변동성 및 기존 `soft_gt_252ep`와의 차이를 기록한다.
