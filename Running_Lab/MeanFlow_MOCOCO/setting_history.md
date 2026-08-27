# Setting History

## 2026-08-26 Soft Best MR-jitter 360-only Stage1

남는 GPU를 이용해 Soft_MOCOCO Best Stage0 encoder를 고정 조건으로 사용하는
별도 Flow Stage1을 추가한다. 원 논문 baseline과의 형평성을 위해 Stage1의
학습 음성도 `LibriSpeech/train-clean-360`만 사용한다.

- Stage0 checkpoint: `Soft_MOCOCO/exp/20260801_soft_mococo/stage0_moco/checkpoints/pn_encoder_best.pt`
- Stage1 loss: `mr_jitter.enabled=true`, `sigma=0.25`, `loss.gamma=1.0`
- Decoder: 기존 `sia_fm_tse` Flow Generator, depth/hidden size 유지
- Validation: NFE=1
- t-predictor: 이 Job에는 사용하지 않으며 Oracle mixing ratio 진단용 Stage1로 분리
- 출력: `exp/20260826_soft_mococo_best_mrjitter_360only/`
