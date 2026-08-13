# Setting History

## Q001

- Date: 2026-07-29
- Status: Resolved
- Question:
  PN_MOCOCO에서 sampling evaluation 대신 실제 전체 test-set evaluation을 사용할 수 있는가?
- Why this matters:
  PN_MOCOCO를 baseline과 비교하려면 안정적이고 충분히 큰 평가 protocol이 필요하다.
- Codex recommendation:
  Running_Lab의 online mixer 기반 방법론은 유지한다. `data.datasets.build_test_dataset`이 유한한 test manifest가 아니라 `eval.test_n`개 pseudo-mixture를 생성하므로, 진짜 full eval 대신 더 큰 deterministic fixed-count evaluation을 사용한다.
- Experimenter answer:
  사용자는 full eval을 요청했으며, 불가능하면 eval 수를 늘리라고 지시했다.
- Applied decision:
  supervised TFGridNet evaluation에 `eval.mode: fixed_count`, `eval.test_n: 5000`, `eval.batch_size: 4`, `eval.num_workers: 2`를 적용했다.
- Related files:
  - `configs/config_tfgridnet_supervised.yaml`
  - `eval_tfgridnet.py`

## Q002

- Date: 2026-07-29
- Status: Resolved
- Question:
  두 단계로 구성된 PN_MOCOCO 산출물을 기존 결과를 덮어쓰지 않도록 어떻게 저장해야 하는가?
- Why this matters:
  MoCo encoder stage와 TFGridNet stage는 서로 의존하는 artifact를 생성하므로, 하나의 추적 가능한 output directory 안에 stage별 결과가 남아야 한다.
- Codex recommendation:
  pipeline 실행마다 `exp/YYYYMMDD_pn_mococo[_runNN]/` 폴더를 생성하고, 그 안에 `stage0_moco/`, `stage1_tfgridnet/`, `evaluation/`, source config, runtime config, command, environment metadata를 저장한다.
- Experimenter answer:
  사용자는 기존 Running_Lab 실험을 `exp_setting.md`에 맞게 정리하고 개선하라고 요청했다.
- Applied decision:
  `script/prepare_runtime_config.py`를 추가하고 train/eval script가 기본적으로 하나의 run별 output folder를 사용하도록 연결했다.
- Related files:
  - `script/prepare_runtime_config.py`
  - `script/train_eval.sh`
  - `script/run_train.sh`
  - `script/run_eval.sh`

## Q003

- Date: 2026-07-29
- Status: Resolved
- Question:
  PN_MOCOCO 정리 과정에서 queue size, optimizer, batch size, data split, model structure를 바꿔도 되는가?
- Why this matters:
  이런 값을 바꾸면 기존 MoCo 비교 실험 정리가 아니라 새로운 가설 검증 실험이 된다.
- Codex recommendation:
  Running_Lab의 기존 방법론을 유지한다. 즉 momentum encoder, queue size 4096, InfoNCE-style objective, 동일한 LibriSpeech/WHAM data contract, 동일한 batch size, 동일한 4-GPU DDP 설정을 보존한다.
- Experimenter answer:
  사용자는 Running_Lab에서 사용되는 방법론을 그대로 쓰라고 지시했다.
- Applied decision:
  이 정리 작업 중 MoCo method hyperparameter는 변경하지 않았다.
- Related files:
  - `configs/config_moco_encoder.yaml`
  - `configs/config_tfgridnet_supervised.yaml`

## Q004

- Date: 2026-07-29
- Status: Resolved
- Question:
  shell script를 기존 `scripts/`에 둘 것인가, 아니면 `exp_setting.md`의 `script/` 구조로 이동할 것인가?
- Why this matters:
  `exp_setting.md`는 각 Lab의 실행 script를 `Running_Lab/<Lab_Name>/script/` 아래에서 관리하도록 요구한다.
- Codex recommendation:
  실행 위치가 하나만 남도록 PN_MOCOCO active execution script를 `script/`로 이동한다.
- Experimenter answer:
  사용자는 각 repo가 `exp_setting.md`에 부합할 때까지 정리하라고 지시했다.
- Applied decision:
  PN_MOCOCO 실행 script를 `script/`로 이동했다.
- Related files:
  - `script/train_eval.sh`
  - `script/run_train.sh`
  - `script/run_eval.sh`

## Q005

- Date: 2026-07-29
- Status: Resolved
- Question:
  PN_MOCOCO에 복사되어 있던 legacy PN-Enroll script를 이 Lab에 계속 둘 것인가?
- Why this matters:
  legacy script는 `output/`에 결과를 쓰고, 이전 sampled evaluation entrypoint를 사용하며, `Base/TSE-through-Positive-Negative-Enroll`에 보존된 코드와 중복된다.
- Codex recommendation:
  active PN_MOCOCO pipeline에서 참조하지 않는 legacy 파일은 제거하고 원본 구현은 `Base/`에만 보존한다.
- Experimenter answer:
  사용자는 Running_Lab repo가 `exp_setting.md`에 부합하도록 정리하라고 요청했다.
- Applied decision:
  이 Lab에서 old `train.py`, `train_improved-model.py`, `eval_monaural.py`, `eval_binaural.py`, old Hyperparameter config, old dataset wrapper, old improved_model 파일, `utils.py`를 제거했다.
- Related files:
  - `train_moco_encoder.py`
  - `train_tfgridnet.py`
  - `eval_tfgridnet.py`
  - `script/train_eval.sh`
  - `README.md`

## Q006

- Date: 2026-07-29
- Status: Resolved
- Question:
  현재 shell에서 CUDA가 보이지 않을 때 5000-item PN_MOCOCO evaluation은 어떻게 실행해야 하는가?
- Why this matters:
  fixed-count evaluation을 CPU에서 실행하면 너무 느리지만, `exp_setting.md`에 맞게 재현 가능한 command와 output은 남겨야 한다.
- Codex recommendation:
  직접 실행 evaluation에는 GPU guard를 유지한다. 호스트가 `gate*`가 아니고 CUDA가 보이면 현재 연산노드를 사용하며, CUDA가 보이지 않는 경우에는 run folder와 runtime config를 만든 뒤 Slurm 단일 GPU job으로 evaluation을 제출한다.
- Experimenter answer:
  사용자는 full eval이 불가능하면 eval 수를 늘리라고 했고, `gate*`가 아닌 연산노드에서는 해당 노드를 활용하라고 지시했다.
- Applied decision:
  `script/submit_eval_slurm.sh`를 추가하고, 현재 host가 `gate*`가 아니며 CUDA가 보이면 직접 실행하고 그렇지 않으면 Slurm에 제출하도록 했다.
- Related files:
  - `script/submit_eval_slurm.sh`
  - `script/check_eval_result.py`
  - `script/train_eval.sh`
  - `script/run_eval.sh`

## Q007

- Date: 2026-07-29
- Status: Resolved
- Question:
  Slurm에서 미리 생성한 runtime config로 실행할 때도 `command.txt`와 `environment.txt`를 남겨야 하는가?
- Why this matters:
  `PREPARED_RUN=1`로 실행하면 이미 생성된 config를 재사용하지만, 이 경우에도 실제 실행 command, host, Python/PyTorch/CUDA 정보, Slurm job id가 보존되어야 재현성이 유지된다.
- Codex recommendation:
  `PREPARED_RUN=1`인 경우에도 metadata 파일이 아직 없으면 실행 직전에 `command.txt`와 `environment.txt`를 기록한다. 기존 metadata가 있으면 덮어쓰지 않는다.
- Experimenter answer:
  사용자는 각 repo가 `exp_setting.md`에 부합할 때까지 수행하라고 지시했다.
- Applied decision:
  `script/train_eval.sh`에 prepared run metadata 기록 로직과 `RECORD_RUNTIME_ONLY=1` 검증 모드를 추가했다.
- Related files:
  - `script/train_eval.sh`
  - `exp/20260729_pn_mococo_eval/`
