# Setting History

## Q001

- Date: 2026-08-04
- Status: Resolved
- Question:
  PN_Indiv_MOCOCO의 individual-negative Stage0 objective에 Soft_MOCOCO의 frozen
  Teacher embedding cosine term을 결합할 것인가?

- Why this matters:
  individual-negative MoCo는 각 nuisance enrollment를 직접 밀어낼 수 있지만, contrastive
  update가 original PN separator가 학습한 condition embedding 공간에서 과도하게 벗어날 수 있다.

- Codex recommendation:
  PN_Indiv_MOCOCO의 CE와 `1/n` negative log-mean-exp 구조를 유지하고, Soft_MOCOCO와
  같은 frozen initial-PN Teacher cosine term을 weight `0.1`로 추가한다.

- Experimenter answer:
  Soft_MOCOCO 방식의 Stage0 loss와 PN_Indiv_MOCOCO의 개별 negative MoCo를 결합한
  Soft_Indiv_MOCOCO Lab을 구성한다.

- Applied decision:
  `L_stage0 = L_individual_moco + 0.1 * L_fixed_teacher_cosine`으로 구현했다. Teacher는
  initial PN encoder에서 복제한 뒤 gradient와 EMA 갱신 없이 고정한다. EMA momentum encoder는
  MoCo key 생성용으로 별도로 유지한다.

- Related files:
  - `pn_soft_indiv_mococo/soft_indiv_moco.py`
  - `train_moco_encoder.py`
  - `configs/config_soft_indiv_moco.yaml`

## Q002

- Date: 2026-08-04
- Status: Resolved
- Question:
  Stage1 leakage loss에 aggregate background 대신 무엇을 전달하고, Stage0 checkpoint
  sweep과 재개 상태를 어떻게 보존할 것인가?

- Why this matters:
  `target_orthogonal_leakage_loss`는 actual mixture의 개별 nuisance `[B, J, T]`를 전제로
  계산한다. 또한 Stage0 epoch에 따른 Stage1 성능 비교에는 encoder export와 full resume
  checkpoint가 모두 필요하다.

- Codex recommendation:
  online mixer의 동일 `sample[1:]`을 `nuisance_sources`로 보존하고,
  `sum(nuisance_sources) == mixture - source`를 검증한다. 매 50 completed epochs마다
  full Lightning checkpoint와 PN encoder export를 함께 저장하고, seed/RNG/optimizer snapshot을
  exp 내부 `resume_state/`에 별도로 기록한다.

- Experimenter answer:
  기존 Stage1 loss와 Leakage Stage1 loss를 사용하고, 50 epoch 배수 checkpoint sweep 및
  optimizer 상태와 seed를 포함한 언제든지 가능한 재개 환경을 구성한다.

- Applied decision:
  `L_stage1 = -SI-SDR + 0.1 * target_orthogonal_leakage_loss`로 구성했다. Stage1은
  `sample[1:]` source를 사용하며 non-finite loss와 nuisance reconstruction contract를
  fail-fast로 검사한다. Stage0 `stage0_epoch_XXX.ckpt`에는 optimizer/RNG가 저장되고,
  `resume_state/`에는 별도 optimizer/RNG snapshot을 저장한다.

- Related files:
  - `pn_soft_indiv_mococo/data.py`
  - `pn_soft_indiv_mococo/resume.py`
  - `train_tfgridnet_soft_indiv.py`
  - `configs/config_tfgridnet_soft_indiv.yaml`
  - `script/train_eval_20260804.sh`

## Q003

- Date: 2026-08-06
- Status: Resolved
- Question:
  Soft_Indiv_MOCOCO의 individual negative contrastive loss에서 negative 수 `n`에 대한
  `1/n` 평균화를 제거한 비교군을 추가할 것인가?

- Why this matters:
  현재 loss는 normalized embedding similarity를 `logsumexp`로 집계한 뒤 `-log(n)`을
  적용한다. 벡터 정규화는 각 similarity의 범위를 제한하지만, `1/n`을 제거하면 negative
  개수가 늘어날 때 aggregate logit과 loss scale이 함께 커지는 효과가 남는다.

- Codex recommendation:
  다른 조건은 유지하고 `negative_aggregation: sum` 비교군을 별도 experiment로 실행한다.
  이 조건은 `logsumexp(negative_logits)`를 사용하며, 기존 `mean` 조건의 기본값은 유지한다.

- Experimenter answer:
  기존 Loss에 `1/n` term이 없는 비교군을 만들고, Best와 별개로 50 Epoch 배수 checkpoint를
  저장하여 비교한다.

- Applied decision:
  `20260806_soft_indiv_mococo_no_1n`을 생성한다. Soft Teacher term, EMA, normalized
  embedding, 데이터셋, optimizer, 300 Epoch 설정은 기존 Soft_Indiv와 동일하게 유지하고
  `contrastive.negative_aggregation: sum`만 적용한다. Best checkpoint와 50 Epoch 배수의
  full checkpoint/export/resume state를 모두 실험 폴더 안에 저장한다.

- Related files:
  - `PN_Indiv_MOCOCO/pn_indiv_mococo/moco_encoder.py`
  - `configs/config_soft_indiv_moco_no_1n.yaml`
  - `script/slurm_train_no_1n_20260806.sh`

## Q004

- Date: 2026-08-07
- Status: Resolved
- Question:
  Soft_Indiv_MOCOCO Stage1의 Leakage Loss에 residual energy 정규화와 저에너지
  target/nuisance gate를 적용할 것인가?

- Experimenter answer:
  이번 건에 한해 Base Loss 수정 권한을 부여하고 변경 내용을 Code_Snippet Markdown에
  기록한다.

- Applied decision:
  공통 `target_orthogonal_leakage_loss`에 revised 기본값을 적용하고 Soft_Indiv Stage1
  trainer가 새 gate 설정과 activity 진단 지표를 기록하도록 변경했다.

- Related files:
  - `Base/Code_Snippet/loss_code.py`
  - `Base/Code_Snippet/leakage_loss_revision.md`
  - `train_tfgridnet_soft_indiv.py`
