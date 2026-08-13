# Setting History

## Q001

- Date: 2026-08-01
- Status: Resolved
- Question:
  MoCo objective에 Teacher embedding 보존 loss를 추가할 것인가?

- Why this matters:
  Contrastive update가 Teacher가 이미 제공하던 decoder-compatible representation을
  빠르게 변경하면 Stage1 separator가 embedding을 해석하지 못할 수 있다.

- Codex recommendation:
  frozen Teacher의 dense PN condition embedding과 student embedding 사이 cosine loss를
  추가하고, 기본 `teacher_loss_weight=0.1`로 시작한다. MoCo loss가 주 objective이므로
  Teacher loss는 regularizer로 사용한다.

- Experimenter answer:
  MoCo와 Baseline/Teacher loss를 함께 사용해 representation을 서서히 개선한다.

- Applied decision:
  `L_total = L_MoCo + teacher_loss_weight * L_teacher`를 적용했다.
  `teacher_loss_weight`는 YAML hyperparameter로 노출했다.

- Related files:
  - `pn_soft_mococo/soft_moco.py`
  - `configs/config_soft_moco.yaml`
  - `train_moco_encoder.py`

## Q002

- Date: 2026-08-03
- Status: Resolved
- Question:
  Stage0 학습 길이에 따른 encoder 효과를 확인하기 위해 50 epoch checkpoint를 고정하고,
  Soft-leakage Stage1 학습과 평가를 독립 실행할 것인가?

- Why this matters:
  Stage0의 contrastive validation loss 최저 시점과 Stage1 분리 성능 최적 시점은 다를 수 있다.
  Stage1을 이전 실험에서 이어 학습하면 encoder checkpoint 효과와 Stage1 학습량이 섞인다.

- Codex recommendation:
  `pn_encoder_50ep.pt`를 입력으로 고정하고, Stage1은 baseline 초기 가중치에서 매번 새로
  학습한 뒤 동일한 fixed-count evaluation으로 비교한다.

- Experimenter answer:
  Stage1 실험을 Soft-leakage 방식으로 진행하고, Stage0 50 epoch checkpoint를 복사한 뒤
  바로 학습할 수 있는 실험 베이스를 구성한다.

- Applied decision:
  `exp/20260803_stage0_50ep_softleakage_test/`를 생성했다. Stage0 입력 checkpoint는
  `stage0_moco/checkpoints/pn_encoder_50ep.pt`로 고정했고, Soft-leakage Stage1 및 평가
  전용 실행/Slurm 스크립트를 추가했다.

- Related files:
  - `configs/config_tfgridnet_soft_leakage_20260803_stage0_50ep_test.yaml`
  - `script/train_stage1_softleakage_20260803.sh`
  - `script/slurm_stage1_softleakage_20260803.sh`

## Q003

- Date: 2026-08-03
- Status: Resolved
- Question:
  Stage0 50 epoch encoder의 Stage1 성능을 확인할 때 leakage loss를 제거한 대조 실험을
  별도로 실행할 것인가?

- Why this matters:
  `target_orthogonal_leakage_loss`가 SI-SDR보다 큰 scale로 학습을 지배해 NaN을 유발할 수
  있으므로, encoder checkpoint 효과와 leakage regularizer 효과를 분리해야 한다.

- Applied decision:
  동일한 `pn_encoder_50ep.pt`를 사용하고, leakage loss를 호출하지 않는 표준 TFGridNet
  trainer로 Stage1과 fixed-count evaluation을 실행하는
  `20260803_stage0_50ep_soft_wo_leakage_test`를 추가했다.

- Related files:
  - `configs/config_tfgridnet_soft_wo_leakage_20260803_stage0_50ep_test.yaml`
  - `script/run_tfgridnet_no_leakage.py`
  - `script/train_stage1_wo_leakage_20260803.sh`
  - `script/slurm_stage1_wo_leakage_20260803.sh`

## Q004

- Date: 2026-08-03
- Status: Resolved
- Question:
  Soft-leakage Stage1에서 `target_orthogonal_leakage_loss`의 interferer 입력으로 무엇을
  사용해야 하는가?

- Why this matters:
  해당 loss는 합산된 `background` 또는 negative enrollment가 아닌, 실제 mixture를 구성한
  개별 nuisance waveform `[B, J, T]`를 전제로 projection과 source별 평균을 계산한다.
  aggregate background를 전달하면 loss의 의미와 scale이 달라져 학습 불안정 원인이 된다.

- Applied decision:
  Soft_MOCOCO 전용 Stage1 data adapter가 online mixer의 동일한 `sample[1:]`을
  `nuisance_sources`로 보존하도록 했다. trainer는 aggregate fallback을 제거하고
  `sum(nuisance_sources) == mixture - source`를 첫 batch에서 검증한다. non-finite loss는
  optimizer step 전에 즉시 예외로 중단한다.

- Related files:
  - `pn_soft_mococo/stage1_data.py`
  - `train_tfgridnet_soft.py`
  - `configs/config_tfgridnet_soft_leakage_20260803_stage0_50ep_test.yaml`

## Q005

- Date: 2026-08-05
- Status: Resolved
- Question:
  Soft_MOCOCO의 원래 epoch-100 Stage0 export가 보존되지 않은 상태에서, 현재 남아 있는
  checkpoint 중 100 epoch와 가장 가까운 encoder를 Stage1 Soft-leakage sweep에 사용할 것인가?

- Why this matters:
  Stage0의 `pn_encoder_last.pt`는 고정 파일명으로 이후 epoch에서 덮어써진다. Stage1 sweep은
  입력 encoder를 고정해야 Stage0 epoch 차이에 따른 분리 성능을 비교할 수 있다.

- Codex recommendation:
  metadata 기준으로 epoch 100과 가장 가까운 완료 export인 validation-best epoch 134
  `pn_encoder_best.pt`를 별도 실험 디렉터리에 즉시 복사해 고정한다.

- Experimenter answer:
  현재 상황에서 100 epoch와 가장 유사한 checkpoint를 가져와 Stage1 sweep 작업을 시작한다.

- Applied decision:
  epoch 23의 기존 `last.ckpt`보다 epoch 134가 epoch 100에 더 가깝다. 원본
  `pn_encoder_best.pt`를 `20260805_stage0_134ep_softleakage_test`의
  `pn_encoder_134ep.pt`로 고정 복사하고, 동일한 individual-source Soft-leakage Stage1 및
  fixed-count evaluation을 Slurm 제출 대상으로 구성했다.

- Related files:
  - `configs/config_tfgridnet_soft_leakage_20260805_stage0_134ep_test.yaml`
  - `script/train_stage1_softleakage_20260805_stage0_134ep.sh`
  - `script/slurm_stage1_softleakage_20260805_stage0_134ep.sh`
- `exp/20260805_stage0_134ep_softleakage_test/stage0_moco/checkpoints/pn_encoder_134ep.pt`

## Q006

- Date: 2026-08-06
- Status: Resolved
- Question:
  Stage0가 epoch 159까지 완료된 시점의 encoder를 보존하고, 해당 checkpoint를 사용하는
  별도 Stage1 Soft-leakage sweep을 pending할 것인가?

- Why this matters:
  계속 학습 중인 Stage0의 `pn_encoder_last.pt`는 다음 epoch에서 덮어써지므로, Stage1
  비교군은 encoder 파일과 full resume state를 독립 실험 디렉터리에 고정해야 한다.

- Applied decision:
  full Lightning state의 `epoch=159`, `global_step=50080`을 확인하고 encoder export를
  `pn_encoder_159ep.pt`로, full state를 `stage0_last_159ep.ckpt`로 복사했다. 해당 encoder를
  사용하는 corrected individual-nuisance Soft-leakage Stage1 및 fixed-count evaluation을
  별도 Slurm job으로 제출한다.

- Related files:
  - `configs/config_tfgridnet_soft_leakage_20260806_stage0_159ep_test.yaml`
  - `script/train_stage1_softleakage_20260806_stage0_159ep.sh`
  - `script/slurm_stage1_softleakage_20260806_stage0_159ep.sh`
  - `exp/20260806_stage0_159ep_softleakage_test/stage0_moco/checkpoints/pn_encoder_159ep.pt`
  - `exp/20260806_stage0_159ep_softleakage_test/stage0_moco/checkpoints/stage0_last_159ep.ckpt`

## Q007

- Date: 2026-08-07
- Status: Resolved
- Question:
  Soft_MOCOCO Stage1 Leakage Loss에 residual energy 정규화와 저에너지 window gate를
  적용할 것인가?

- Experimenter answer:
  이번 건에 한해 `Base/Code_Snippet/loss_code.py` 수정을 허용하고, 변경 내용을 Code_Snippet
  Markdown에 상세히 기록한다.

- Applied decision:
  공통 Loss의 revised 기본값을 적용하고 Soft_MOCOCO Stage1 trainer가 target/nuisance
  activity ratio를 TensorBoard에 기록하도록 변경했다. Pending 159 Epoch runtime/source
  config에도 새 설정을 명시했다.

- Related files:
  - `Base/Code_Snippet/loss_code.py`
  - `Base/Code_Snippet/leakage_loss_revision.md`
  - `train_tfgridnet_soft.py`
  - `exp/20260806_stage0_159ep_softleakage_test/config_tfgridnet_soft_runtime.yaml`
