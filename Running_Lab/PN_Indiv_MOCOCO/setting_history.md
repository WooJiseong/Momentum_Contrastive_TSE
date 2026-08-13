# Setting History

## Q001

- Date: 2026-08-01
- Status: Resolved
- Question:
  REAL 합성 데이터의 여러 negative speaker와 WHAM noise를 하나의 negative waveform으로
  합산하지 않고 개별 negative enrollment로 contrastive loss에 넣을 것인가?

- Why this matters:
  기존 PN_MOCOCO는 `neg_cond`를 합산하므로 여러 화자와 noise의 embedding이 하나의
  negative vector로 섞인다. 개별 negative를 사용하면 각 negative와의 거리를 직접
  제어할 수 있다.

- Codex recommendation:
  개별 negative key를 만들고 각 negative logit에 `1/n`을 적용한다. negative speaker
  수에 따른 총 repulsion 증가를 막기 위해 weighted log-mean-exp를 사용한다.

- Experimenter answer:
  Baseline에 대해 개별 negative 화자 및 noise와 멀어지고 positive enrollment와 가까워지는
  Contrastive Loss를 구성한다. negative 화자가 여러 명이면 각 negative는 1/n만큼 반영한다.

- Applied decision:
  `PN_Indiv_MOCOCO` Stage0에 반영했다. positive/negative PN query 경로는 유지하고,
  negative item만 개별 encode한다. padding row는 mask로 loss에서 제외한다.

- Related files:
  - `pn_indiv_mococo/moco_encoder.py`
  - `pn_indiv_mococo/data.py`
  - `train_moco_encoder.py`

## Q002

- Date: 2026-08-02
- Status: Resolved
- Question:
  20260801 PN_Indiv_MOCOCO의 NaN Stage0 로그를 폐기하고, Stage1 TFGridNet loss에
  `Base/Code_Snippet/loss_code.py`의 `target_orthogonal_leakage_loss`를 사용할 것인가?

- Why this matters:
  기존 20260801 PN_Indiv_MOCOCO Stage0 run은 `train_loss`, `val_loss`,
  `pos_sim`, `neg_sim`이 처음부터 NaN으로 기록되어 비교 가능한 결과가 아니다.
  또한 leakage loss를 Stage1에 적용하면 분리 loss 정의가 기존 PN_MOCOCO와 달라진다.

- Codex recommendation:
  잘못된 Stage0 로그와 체크포인트를 정리한 뒤 새로 시작한다. Stage1에서는 분리 성능을 유지하기 위해
  기존 SI-SDR loss에 target-orthogonal leakage loss를 가중합으로 추가한다.

- Experimenter answer:
  PN_INDIV_MOCOCO를 디버깅하고 `Base/Code_Snippet/loss_code.py`의 코드를 loss로 사용한다.
  현재 잘못된 로그 및 파일들을 정리하고, 한 번에 돌리는 shell script와 Slurm 명령을 준비한다.

- Applied decision:
  Stage0는 invalid/padded negative row를 encode하지 않고, loss 내부에서 비유한 negative logit이
  TensorBoard로 전파되지 않도록 수정했다. Stage1은
  `loss.type=si_sdr_plus_target_orthogonal_leakage`,
  `orthogonal_leakage_weight=0.1`로 실행한다. 20260801의 NaN 로그와 체크포인트는 삭제한다.

- Related files:
  - `pn_indiv_mococo/moco_encoder.py`
  - `pn_indiv_mococo/data.py`
  - `pn_indiv_mococo/losses.py`
  - `train_moco_encoder.py`
  - `train_tfgridnet_indiv.py`
  - `configs/config_tfgridnet_indiv.yaml`
  - `script/train_eval_20260801.sh`
  - `script/slurm_train_eval_20260801.sh`

## Q003

- Date: 2026-08-04
- Status: Resolved
- Question:
  PN_Indiv_MOCOCO Stage1의 `target_orthogonal_leakage_loss`에 aggregate background가 아니라
  actual mixture의 개별 nuisance source를 전달하고, 완료된 20260802 Stage0 best encoder로
  별도 Stage1 실험을 시작할 것인가?

- Why this matters:
  `target_orthogonal_leakage_loss`는 `[B, J, T]` 개별 방해 음원을 전제로 한다. aggregate
  background `[B, T]`나 negative enrollment를 전달하면 source별 leakage 정의가 달라진다.
  또한 기존 Stage1 산출물과 올바른 데이터 계약으로 재학습한 결과를 분리해야 한다.

- Codex recommendation:
  online mixer가 생성한 동일 `sample[1:]`을 `nuisance_sources`로 보존하고,
  `sum(nuisance_sources) == mixture - source`를 첫 batch에서 검증한다. Stage0는 검증
  성능으로 선택된 epoch 88 `pn_encoder_best.pt`를 초기 encoder로 고정한다.

- Experimenter answer:
  PN_Indiv_MOCOCO의 TFGridNet Stage1 학습 Script를 작성 후 Slurm으로 제출한다.

- Applied decision:
  `20260804_indiv_mococo_stage1_individual_nuisance`를 별도 실험으로 구성했다.
  Stage1 objective는 `-SI-SDR + 0.1 * target_orthogonal_leakage_loss`이며, leakage term은
  negative enrollment가 아닌 actual `sample[1:]`을 사용한다. full Lightning `last.ckpt`는
  optimizer와 scheduler 상태를 포함하며 `RESUME_STAGE1`으로 재개할 수 있다.

- Related files:
  - `pn_indiv_mococo/stage1_data.py`
  - `train_tfgridnet_indiv.py`
  - `configs/config_tfgridnet_indiv_stage1_20260804.yaml`
  - `script/train_tfgridnet_stage1_20260804.sh`
  - `script/slurm_tfgridnet_stage1_20260804.sh`

## Q004

- Date: 2026-08-07
- Status: Resolved
- Question:
  Stage1 Leakage Loss의 residual energy 정규화와 target/nuisance activity gate를 공통
  구현에 적용할 것인가?

- Experimenter answer:
  이번 건에 한해 `Base/Code_Snippet/loss_code.py` 수정을 허용하고, 변경 내용을 Code_Snippet
  Markdown에 상세히 기록한다.

- Applied decision:
  `normalize_residual_energy: true`, `activity_gate: true`, 상대 activity threshold `0.01`을
  기본값으로 적용했다. 실행 중인 프로세스는 기존 import 코드를 유지하고, 이후 시작되는
  Stage1 작업부터 revised Loss를 사용한다.

- Related files:
  - `Base/Code_Snippet/loss_code.py`
  - `Base/Code_Snippet/leakage_loss_revision.md`
  - `train_tfgridnet_indiv.py`
