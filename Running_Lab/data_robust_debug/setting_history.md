# Setting History

## 2026-08-31: 데이터셋 재현성 점검

### Objective

현재 `soft_mococo` 계열에서 train loss는 감소하지만 validation loss가 크게 변동하는
현상을 데이터셋 차이와 로깅 집계 차이로 분리한다.

### Applied

- 초기 weight는 `/gpfs/home1/wjs6800/SIALab/PNFlowTSE/checkpoints/proposed-monaural.pt`로 고정한다.
- `improved-monaural.pt`는 사용하지 않는다.
- Teacher Weight는 `0.1`로 고정한다.
- 두 실험 모두 `num_blocks=3`, 동일 projection head, 동일 augmentation을 사용한다.
- 두 실험 모두 `speaker_aware_queue=true`를 사용한다.
- `360+100`과 `360-only` 사이에서 `dataset.train_roots`만 변경한다.
- `resume: null`로 설정하여 기존 `soft_gt_252ep`의 optimizer, EMA, queue 상태를 상속하지 않는다.
- 최근 teacher-decay trainer의 수동 aggregation 대신 표준 Soft trainer를 사용한다.

### Interpretation

기존 `soft_gt_252ep`는 이전 checkpoint에서 resume한 실험이므로 새 실험과 곡선이
완전히 일치할 것으로 기대하지 않는다. 기존 run은 TensorBoard의 historical reference로
함께 표시한다. 새 두 실험의 차이를 우선 데이터셋 효과로 해석한다.
