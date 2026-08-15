# contrastive_momentum 파일 정리표

이 문서는 레포지토리 최상단에 있는 실행 파일과 보조 도구를 정리하기 위한
목록이다. 실험 Lab 내부의 학습 코드, 모델 코드, 설정 파일은 이 문서의 정리
대상이 아니며, 각 Lab의 실행 구조를 그대로 유지한다.

## 상태 표기

| 상태 | 의미 |
|---|---|
| `유지` | 현재 실험 또는 반복적인 분석에 사용하므로 유지한다. |
| `조건부 유지` | 재사용 가능성이 있으나 사용 전에 실험 경로와 설정을 확인한다. |
| `보관` | 과거 실험 기록 또는 결과 산출물이다. 코드 실행용으로는 필요하지 않다. |
| `삭제 후보` | 일회성 실행 또는 중복 파일이다. 관련 Job과 결과를 확인한 뒤 정리한다. |
| `Lab 관리` | `Running_Lab/<Lab>` 내부 소유 코드이다. 최상단 정리에서 건드리지 않는다. |

## 우선 결론

현재 반복적으로 사용할 최상단 도구는 다음 세 묶음이다.

1. [`exp_setting.md`](exp_setting.md): 실험 작성 규칙과 기록 기준
2. [`debug/`](debug/): 모든 Lab의 1 GPU, 1 epoch, line-by-line 검증 설정
3. [`tools/`](tools/): TensorBoard 지표 감시, baseline 결과 연결, 음성 샘플 생성

최상단의 `depth_sweep_*`, `prepare_depth_sweep.py`, `slurm_*depth_sweep*` 파일은
2026-07-30 depth sweep을 위한 고정 경로 일괄 실행 묶음이다. 새로운 실험의
일반 진입점으로 사용하지 않고, 필요하면 먼저 경로와 Job dependency를 수정한다.

## 유지할 공통 파일

| 파일/디렉터리 | 목적 | 추후 사용 여부 |
|---|---|---|
| [`exp_setting.md`](exp_setting.md) | 실험 디렉터리, checkpoint, 로그, Slurm 기록을 만드는 공통 규칙 | `유지` |
| [`debug/README.md`](debug/README.md) | Stage0/Stage1 debug 실행 절차와 PDB 명령 | `유지` |
| [`debug/*.yaml`](debug/) | PN, PN-Indiv, Soft, Soft-Indiv 및 baseline의 1 GPU debug 설정 | `유지` |
| [`tools/README.md`](tools/README.md) | 공통 보조 도구 사용법 | `유지` |
| [`tools/watch_experiment_metrics.py`](tools/watch_experiment_metrics.py) | TensorBoard event를 읽어 validation graph PNG를 생성하고 watch | `유지` |
| [`tools/fill_eval_baseline.py`](tools/fill_eval_baseline.py) | Eval 결과 JSON에 baseline 비교와 `result_summary.md`를 채움 | `유지` |
| [`tools/sample_audio_mkr/`](tools/sample_audio_mkr/) | 완료된 Stage1 checkpoint에서 deterministic sample 음성과 metrics YAML 생성 | `유지` |
| [`exp/`](exp/) | 최상단 공통 도구의 결과 및 분석 산출물 | `생성 결과만 보관` |

### 공통 도구의 사용 기준

- 학습과 Eval의 실제 진입점은 `Running_Lab/<Lab>/script/` 아래 파일이다.
- `tools/`는 학습을 대신하지 않는다. checkpoint 또는 TensorBoard event가
  이미 있는 실험에 대해 분석과 데모를 수행한다.
- `debug/` 설정은 smoke test와 PDB 분석 전용이다. 논문용 학습이나 5000-sample
  Eval 설정으로 사용하지 않는다.

## 최상단 실행 파일

### `one_time_script.sh`

| 항목 | 내용 |
|---|---|
| 목적 | 기존 PN_MOCOCO Stage1 checkpoint를 model-only checkpoint로 변환하고 low-LR Stage2 fine-tuning을 시작 |
| 호출 대상 | `Running_Lab/PN_MOCOCO/script/prepare_runtime_config.py`, `script/train_eval.sh tfgridnet` |
| 특징 | GPU 4장, low learning rate, early stopping, Stage2 성격의 일회성 설정 |
| 주의 | 지정된 runtime config와 checkpoint를 생성하며, 환경변수와 기존 checkpoint 경로에 의존 |
| 판정 | `조건부 유지` 또는 실험 종료 후 `삭제 후보` |

현재 일반적인 Stage0/Stage1/Stage1 Eval의 진입점은 이 파일이 아니다.
다시 사용할 때는 `BEST_CKPT`, `MODEL_ONLY_CKPT`, `NUM_GPUS`, `LOW_LR`,
`NUM_EPOCHS`, `PATIENCE`를 먼저 확인한다.

### `check_png_renderer.sh`

| 항목 | 내용 |
|---|---|
| 목적 | compute node에서 PIL, cairosvg, ImageMagick renderer 가용 여부 확인 |
| 결과 | `exp/20260801_tensorboard_depth_sweep/png-check-<job>.out` |
| 판정 | 환경을 이미 확인했다면 `삭제 후보`; 재현성 기록이 필요하면 `보관` |

이 파일은 모델 학습이나 일반적인 metric graph 생성에 필수적이지 않다.

## 과거 Depth Sweep 실행 묶음

다음 파일들은 Baseline과 PN_MOCOCO에 대해 depth `3, 4, 6`을 비교한
2026-07-30 실험을 위해 작성되었다. 대부분 절대 경로와 특정 Job 번호를
하드코딩하고 있으므로, 새 sweep에 그대로 재사용하지 않는다.

### 실행 생성기

| 파일 | 목적 | 판정 |
|---|---|---|
| [`prepare_depth_sweep.py`](prepare_depth_sweep.py) | depth별 YAML과 Lab별 `slurm_train_eval.sh`를 자동 생성 | `조건부 유지` |

`DEPTHS = (3, 4, 6)`, `NUM_EPOCHS = 50`, `DATE = 20260730`이 고정되어
있다. 재사용하려면 날짜, output directory, checkpoint 경로, Slurm 자원을
먼저 수정해야 한다.

### 결과 수집

| 파일 | 목적 | 판정 |
|---|---|---|
| [`collect_depth_sweep_results.py`](collect_depth_sweep_results.py) | manifest의 6개 run에서 TensorBoard, train log, Eval JSON을 모아 summary 생성 | `조건부 유지` |
| [`slurm_collect_depth_sweep_20260730.sh`](slurm_collect_depth_sweep_20260730.sh) | depth sweep 종료 후 위 수집기를 Slurm으로 실행 | `삭제 후보` 또는 `보관` |

수집기는 다음 고정 파일을 읽고 쓴다.

```text
depth_sweep_manifest_20260730.yaml
depth_sweep_summary_20260730.json
depth_sweep_summary_20260730.md
depth_sweep_final_sacct_20260730.txt
```

새 실험에서는 실험별 manifest와 output path를 인자로 받는 일반화가 필요하다.

### TensorBoard scalar 추출과 그래프 생성

| 파일 | 목적 | 판정 |
|---|---|---|
| [`extract_depth_sweep_tensorboard.py`](extract_depth_sweep_tensorboard.py) | 고정된 6개 run의 scalar를 Markdown으로 추출 | `보관` 또는 `조건부 유지` |
| [`slurm_extract_depth_sweep_tensorboard.sh`](slurm_extract_depth_sweep_tensorboard.sh) | scalar 추출 Slurm wrapper | `삭제 후보` |
| [`plot_depth_sweep_tensorboard.py`](plot_depth_sweep_tensorboard.py) | Matplotlib으로 depth별 validation/train 그래프 생성 | `조건부 유지` |
| [`slurm_plot_depth_sweep_tensorboard.sh`](slurm_plot_depth_sweep_tensorboard.sh) | Matplotlib plotting Slurm wrapper | `삭제 후보` |
| [`render_depth_sweep_png.py`](render_depth_sweep_png.py) | Pillow로 PNG 그래프 생성; Matplotlib 대체 경로 | `조건부 유지` |
| [`slurm_render_depth_sweep_png.sh`](slurm_render_depth_sweep_png.sh) | Pillow PNG 생성 Slurm wrapper | `삭제 후보` |

`plot_depth_sweep_tensorboard.py`는 분석용 원본 그래프 경로이고,
`render_depth_sweep_png.py`는 Matplotlib 설치가 어려울 때의 대체 경로이다.
둘 다 일반적인 새 실험 분석에는 고정된 `RUNS`를 수정해야 한다.

## 중복 후보: v2 파일

| 파일 | 실제 차이 | 판정 |
|---|---|---|
| [`render_depth_sweep_png_v2.py`](render_depth_sweep_png_v2.py) | 기존 renderer와 대부분 동일하며 series 이름 parsing과 변수명만 정리 | `중복 후보` |
| [`slurm_plot_depth_sweep_tensorboard_v2.sh`](slurm_plot_depth_sweep_tensorboard_v2.sh) | 동일한 Python plotting을 실행하며 working directory와 partition 지정 방식이 다름 | `중복 후보` |
| [`slurm_render_depth_sweep_png_v2.sh`](slurm_render_depth_sweep_png_v2.sh) | `render_depth_sweep_png_v2.py`와 별도 output log 이름 사용 | `중복 후보` |

v1/v2 중 하나를 즉시 지우기보다는 결과가 동일한지 확인한 뒤 하나의
일반화된 plotting tool로 통합하는 것이 적절하다. 현재는 두 버전을 모두
일반 실험 진입점으로 취급하지 않는다.

## 과거 Depth Sweep 산출물

다음 파일은 실행 코드가 아니라 2026-07-30 결과 기록이다.

| 파일 | 내용 | 판정 |
|---|---|---|
| `depth_sweep_manifest_20260730.yaml` | 6개 sweep run의 project, depth, Slurm script 정보 | `보관` |
| `depth_sweep_jobs_20260730.md` | 제출된 depth sweep Job 기록 | `보관` |
| `depth_sweep_final_sacct_20260730.txt` | 최종 Slurm 상태/exit code 기록 | `보관` |
| `depth_sweep_summary_20260730.json` | 수집된 구조화 결과 | `보관` |
| `depth_sweep_summary_20260730.md` | 사람이 읽는 depth sweep 요약 | `보관` |
| `depth_sweep_collect-850456.out` | 결과 수집 Job stdout | `보관` 또는 `삭제 후보` |

삭제할 경우 요약 Markdown과 manifest는 남기고, 대용량/중복 stdout만 먼저
정리하는 것이 안전하다.

## Lab 내부 파일과의 경계

다음 디렉터리는 실험 프로세스의 실제 소유 영역이다. 최상단 중복 파일을
정리할 때 아래 파일을 함께 삭제하거나 이동하지 않는다.

| 경로 | 역할 |
|---|---|
| `Running_Lab/PN_MOCOCO/` | 기본 MoCo Stage0와 TFGridNet Stage1 |
| `Running_Lab/PN_Indiv_MOCOCO/` | 개별 Negative Stage0/Stage1 |
| `Running_Lab/Soft_MOCOCO/` | Soft Teacher Stage0/Stage1 |
| `Running_Lab/Soft_Indiv_MOCOCO/` | Soft Teacher + individual Negative |
| `Running_Lab/Attn_MOCOCO/` | lightweight attention pooling 실험 |
| `Running_Lab/TSE-through-Positive-Negative-Enroll/` | 원본 PN-Enroll baseline |
| `Base/` | 원본/reference 코드와 공통 Leakage Loss snippet |

각 Lab의 `README.md`, `configs/`, `script/`, `train_*.py`, `eval_*.py`는
해당 Lab의 실행 계약이다. 공통화할 수 있어 보이더라도 현재 실험 재현성을
확인하기 전에는 최상단 정리 대상으로 보지 않는다.

## 권장 정리 순서

1. 이 문서를 기준으로 현재 실행 중인 Slurm Job과 연결된 파일을 먼저 제외한다.
2. `one_time_script.sh`가 필요한 진행 중 Stage2가 없는지 확인한다.
3. `depth_sweep_manifest_20260730.yaml`과 summary Markdown/JSON을 보관한다.
4. depth sweep의 v1/v2 wrapper 중 재사용할 한 버전을 정한다.
5. 사용하지 않는 depth sweep Slurm wrapper와 `check_png_renderer.sh`를
   `archive/`로 이동하거나 삭제한다.
6. 이후에만 `__pycache__`, 대용량 stdout, 중복 PNG 산출물을 정리한다.

이 문서는 정리 기준만 제공하며, 파일 이동과 삭제는 수행하지 않았다.
