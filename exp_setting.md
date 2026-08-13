# Experiment Workspace Policy

이 문서는 실험 코드의 구성, 수정, 실행 및 결과 관리에 대한 기본 정책을 정의한다.

Codex는 이 문서의 정책을 최우선으로 준수해야 한다.

---

## 1. 기본 원칙

### 1.1 최소 코드 원칙

실험에 직접적으로 필요하지 않은 코드는 삭제한다.

* 사용되지 않는 함수
* 사용되지 않는 클래스
* 사용되지 않는 import
* 이전 실험에서만 사용된 조건문
* 중복된 데이터 처리 코드
* 중복된 모델 구현
* 디버깅용 임시 코드
* 주석 처리된 구형 코드

단, 코드 삭제 전에 현재 실험에서 참조되지 않는 코드인지 반드시 확인한다.

판단이 불확실한 경우 임의로 삭제하지 않고 해당 Lab의 `setting_history.md`에 질문을 작성한다.

가능하면 하나의 실험 폴더에는 해당 실험을 실행하는 데 필요한 최소한의 코드만 유지한다.

코드를 복잡하게 일반화하기보다 현재 실험의 목적이 명확하게 드러나는 단순한 구현을 우선한다.

---

### 1.2 실험 산출물 관리

다음 산출물은 반드시 해당 실험의 `exp` 폴더 내부에 저장한다.

* 로그 파일
* 모델 가중치
* 체크포인트
* 실험 설정 파일
* 평가 결과
* 추론 결과
* 시각화 결과
* TensorBoard 및 기타 모니터링 데이터

실험 폴더 이름에는 다음 정보가 포함되어야 한다.

```text
YYYYMMDD_실험제목
```

예시:

```text
exp/
└── 20260729_moco_queue_size_ablation/
    ├── config.yaml
    ├── train.log
    ├── checkpoints/
    ├── evaluation/
    └── outputs/
```

같은 날짜에 동일한 주제의 실험을 여러 번 실행할 경우 순번 또는 핵심 설정을 추가한다.

```text
20260729_moco_queue4096_run01
20260729_moco_queue8192_run01
```

실험 결과를 프로젝트 루트, 소스 코드 폴더 또는 `script` 폴더에 저장하지 않는다.

---

## 2. 디렉터리 구성

프로젝트는 다음 구조를 기본으로 사용한다.

```text
project_root/
├── Base/
│
├── Running_Lab/
│   ├── Baseline/
│   ├── MoCo/
│   ├── Loss_Improvement/
│   └── Other_Experiment/
│
└── exp_setting.md
```

---

### 2.1 `Base` 폴더

`Base` 폴더에는 원본 Baseline 코드를 보관한다.

```text
Base/
└── Baseline_name/
```

`Base/`은 다음 목적으로만 사용한다.

* 원본 구현 확인
* 기존 동작 확인
* 비교 기준 확인
* 필요한 코드 참조
* Baseline 재현 조건 확인

`Base/` 내부 베이스라인 폴더 및 파일은 읽기 전용으로 취급한다.

Codex는 사용자의 명시적인 요청 없이 다음 작업을 수행해서는 안 된다.

* 파일 수정
* 파일 삭제
* 파일명 변경
* 디렉터리 이동
* 자동 포맷팅
* import 경로 변경
* 설정값 변경

Baseline을 수정해야 하는 상황이 발생하면 직접 수정하지 않고, 먼저 해당 사유를 `setting_history.md`에 질문으로 작성한다.

---

### 2.2 `Running_Lab` 폴더

실제로 수정하고 실행하는 코드는 모두 `Running_Lab` 아래에 둔다.

각 비교 실험은 독립된 Lab 폴더로 분리한다.

예시:

```text
Running_Lab/
├── Baseline/
├── PN_MOCOCO/
├── Loss_Improvement/
├── Encoder_Improvement/
└── Data_Augmentation/
```

하나의 Lab 폴더에서는 하나의 핵심 가설만 검증하는 것을 원칙으로 한다.

예를 들어 MoCo와 Loss 개선을 동시에 적용하는 실험은 기존 폴더에 섞지 않고 별도의 Lab으로 생성한다.

```text
Running_Lab/
└── MoCo_Loss_Improvement/
```

각 Lab 폴더의 권장 구조는 다음과 같다.

```text
Running_Lab/<Lab_Name>/
├── README.md
├── setting_history.md
├── config/
├── script/
├── src/
├── tests/
└── exp/
```

프로젝트 구조상 `src`가 불필요한 경우 생략할 수 있다. 단, 코드가 여러 위치에 무분별하게 분산되지 않도록 한다.

---

### 2.3 Lab 이름 규칙

Lab 이름은 실험의 핵심 변경 사항이 드러나도록 작성한다.

좋은 예시:

```text
MoCo
MoCo_Queue_Ablation
Loss_Triplet
Loss_InfoNCE
Encoder_TFGridNet
EMA_Momentum_Ablation
```

피해야 할 예시:

```text
test
test2
new
final
final2
experiment
temp
```

---

## 3. 의사결정 및 질문 관리

### 3.1 임의 판단 금지

실험 결과 또는 구현에 영향을 줄 수 있는 중요한 사항은 Codex가 자의적으로 결정하지 않는다.

다음과 같은 항목은 중요 의사결정에 해당한다.

* 학습률 변경
* 배치 크기 변경
* Loss 가중치 변경
* 모델 구조 변경
* 데이터 전처리 변경
* 데이터셋 제외 또는 추가
* 평가 지표 변경
* 체크포인트 선택 기준 변경
* 학습 중단 기준 변경
* 기존 기능 삭제
* Baseline과 다른 기본값 적용
* 재현성에 영향을 주는 설정 변경
* 다중 GPU 실행 방식 변경
* 학습 데이터 또는 검증 데이터 분할 변경

판단이 필요한 경우 해당 Lab의 `setting_history.md`에 Q&A 형식으로 기록한다.

---

### 3.2 `setting_history.md` 형식

`setting_history.md`의 모든 질문, 답변, 판단 근거 및 적용 결과는 반드시 한국어로 작성한다.
코드명, 파일명, 라이브러리명 등 고유 기술 용어는 원문 표기를 허용한다.

다음 형식을 사용한다.

```markdown
# Setting History

## Q001

- Date: 2026-07-29
- Status: Pending
- Question:
  MoCo queue size를 Baseline의 4096으로 유지할까요,
  아니면 GPU 메모리를 고려하여 8192로 확장할까요?

- Why this matters:
  Queue size는 negative sample 수와 GPU 메모리 사용량에 영향을 줄 수 있습니다.

- Codex recommendation:
  최초 비교에서는 다른 조건을 유지하기 위해 4096을 권장합니다.

- Experimenter answer:
  <!-- 실험자가 작성 -->

- Applied decision:
  <!-- 답변을 코드에 반영한 뒤 Codex가 작성 -->

- Related files:
  - config/train.yaml
  - src/model/moco.py
```

실험자의 답변이 없고 상태가 `Pending`이면 중요한 설정을 임의로 적용하지 않는다.

답변을 받은 뒤 다음과 같이 상태를 변경한다.

```markdown
- Status: Resolved
```

단순한 문법 수정, 경로 오타 수정, 명백한 런타임 오류 수정 등 실험 조건에 영향을 주지 않는 수정은 별도 질문 없이 수행할 수 있다.

---

### 3.3 질문 번호 관리

질문은 Lab별로 순차 번호를 사용한다.

```text
Q001
Q002
Q003
```

기존 질문 번호를 재사용하거나 삭제하지 않는다.

결정이 번복되면 기존 기록을 수정해서 덮어쓰지 않고 새로운 질문을 추가한다.

---

## 4. 실행 스크립트 관리

### 4.1 스크립트 위치

각 실험 실행에 필요한 셸 스크립트는 해당 Lab의 `script` 폴더에서 관리한다.

```text
Running_Lab/<Lab_Name>/script/
```

예시:

```text
script/
├── train_single_gpu.sh
├── train_multi_gpu.sh
├── evaluate.sh
└── inference.sh
```

프로젝트 루트에 실험별 셸 스크립트를 무분별하게 생성하지 않는다.

---

### 4.2 스크립트 상단 주석

모든 `.sh` 파일의 최상단에는 다음 내용을 주석으로 작성한다.

* 스크립트 목적
* 실행 위치
* 단일 GPU 실행 명령어
* 다중 GPU 실행 명령어
* 필요한 인자
* 출력 경로
* 필요한 환경 또는 가상환경
* Slurm에서 사용하는 경우 실행 예시

예시:

```bash
#!/usr/bin/env bash

# Purpose:
#   Train the MoCo experiment.
#
# Run from:
#   project_root/Running_Lab/MoCo
#
# Single GPU:
#   bash script/train.sh --gpus 1
#
# Multi GPU:
#   torchrun --standalone --nproc_per_node=4 \
#       src/train.py \
#       --config config/train.yaml
#
# Slurm interactive example:
#   srun -p gpu6 \
#       --gres=gpu:4 \
#       --cpus-per-task=16 \
#       --mem=48G \
#       --time=48:00:00 \
#       --pty bash
#
# Output:
#   exp/YYYYMMDD_experiment_name/
#
# Environment:
#   conda activate pnflowtse
```

다중 GPU를 지원하는 코드라면 반드시 다중 GPU 실행 예시를 작성한다.

다중 GPU를 지원하지 않는다면 상단 주석에 명확하게 작성한다.

```bash
# Multi GPU:
#   Not supported.
```

---

### 4.3 안전한 셸 스크립트

셸 스크립트는 가능한 경우 다음 설정을 사용한다.

```bash
set -euo pipefail
```

스크립트 내부에서 절대 경로를 과도하게 하드코딩하지 않는다.

가능하면 스크립트 자신의 위치를 기준으로 경로를 계산한다.

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
```

실험 출력 폴더가 이미 존재할 경우 기존 결과를 자동으로 덮어쓰지 않는다.

덮어쓰기가 필요한 경우 실험자의 명시적인 승인을 받는다.

---

## 5. Baseline 비교 정책

모든 개선 실험은 대응되는 Baseline과 비교 가능해야 한다.

최소한 다음 조건을 가능한 한 동일하게 유지한다.

* 데이터셋
* train/validation/test split
* random seed
* batch size
* epoch 수
* optimizer
* learning rate schedule
* 평가 지표
* 체크포인트 선택 기준
* 데이터 전처리
* GPU 수 또는 effective batch size

개선점과 직접 관계없는 조건은 변경하지 않는다.

Baseline과 다른 설정이 필요한 경우 `setting_history.md`에 변경 이유를 기록한다.

각 Lab의 `README.md`에는 다음 내용을 간단히 작성한다.

```markdown
# MoCo Experiment

## Hypothesis

MoCo queue를 도입하면 더 다양한 negative sample을 사용하여
representation quality가 향상될 것이다.

## Baseline

`Base/Baseline`

## Main Difference

- Momentum encoder 추가
- Negative queue 추가
- InfoNCE loss 적용

## Controlled Conditions

- Dataset
- Batch size
- Optimizer
- Learning rate
- Training epochs

## Evaluation Metrics

- Validation loss
- Validation accuracy
- Positive similarity
- Maximum negative similarity
```

---

## 6. 실험 재현성 정책

각 실험은 동일한 설정으로 다시 실행할 수 있어야 한다.

실험 실행 시 다음 정보를 보존한다.

* 전체 config
* 실행 명령어
* Git commit hash
* random seed
* Python 버전
* PyTorch 버전
* CUDA 버전
* GPU 정보
* 사용한 GPU 수
* 시작 시간
* 종료 시간
* 데이터 버전 또는 데이터 경로
* 체크포인트 경로

가능하면 실행 시 자동으로 환경 정보를 기록한다.

예시:

```bash
git rev-parse HEAD > "${EXP_DIR}/git_commit.txt"
python --version > "${EXP_DIR}/environment.txt"
python -c "import torch; print(torch.__version__)" >> "${EXP_DIR}/environment.txt"
nvidia-smi >> "${EXP_DIR}/environment.txt"
```

재현성 정보는 해당 실험의 `exp/YYYYMMDD_실험제목/` 내부에 저장한다.

---

## 7. 설정 파일 정책

학습 설정은 가능한 한 코드 내부에 하드코딩하지 않고 설정 파일로 분리한다.

권장 위치:

```text
config/
├── train.yaml
├── model.yaml
└── data.yaml
```

실제로 실행한 설정 파일은 원본 설정과 별개로 실험 폴더에 복사한다.

```text
exp/20260729_moco_queue4096/config.yaml
```

실험 실행 후 원본 config가 수정되더라도 기존 실험의 설정이 보존되어야 한다.

명령행 인자로 설정을 덮어썼다면 최종 적용값을 별도로 기록한다.

---

## 8. 변경 범위 제한

Codex는 작업 요청과 직접 관계없는 대규모 리팩터링을 수행하지 않는다.

예를 들어 Loss 함수 수정 요청을 받은 경우 다음 작업을 동시에 수행하지 않는다.

* 전체 데이터로더 재작성
* 모델 파일 구조 전체 변경
* 변수명 전면 변경
* 불필요한 dependency 추가
* 학습 프레임워크 교체

추가 변경이 필요하다고 판단되면 먼저 이유와 영향 범위를 `setting_history.md`에 기록한다.

한 번의 변경에서는 하나의 핵심 목적을 유지한다.

---

## 9. Dependency 관리

새로운 패키지를 추가하기 전에 기존 dependency로 구현할 수 있는지 확인한다.

새 dependency가 필요한 경우 다음 내용을 기록한다.

* 패키지 이름
* 버전
* 도입 이유
* 기존 환경과의 충돌 가능성
* 대체 가능한 방법

dependency 버전은 가능한 한 고정한다.

예시:

```text
torch==2.6.0
torchaudio==2.6.0
numpy==1.23.5
```

실험에 사용하지 않는 dependency는 제거한다. 단, 다른 Lab이나 Baseline의 환경을 직접 수정하지 않는다.

---

## 10. 실험 보호 정책

### 10.1 결과 덮어쓰기 금지

기존 실험 결과를 자동으로 덮어쓰거나 삭제하지 않는다.

다음 파일은 특히 보호한다.

* 체크포인트
* 로그
* 최종 평가 결과
* config
* `setting_history.md`

같은 이름의 실험 폴더가 이미 존재하면 새로운 run 번호를 부여한다.

```text
20260729_moco_run01
20260729_moco_run02
```

---

### 10.2 실행 중 파일 수정 제한

학습이 실행 중인 Lab의 핵심 코드를 수정하지 않는다.

수정이 필요하면 다음 중 하나를 수행한다.

1. 현재 실행을 종료한 뒤 수정한다.
2. 새로운 Lab 또는 새로운 코드 복사본에서 수정한다.
3. 새로운 Git branch 또는 commit에서 수정한다.

실행 중 코드와 기록된 commit이 일치해야 한다.

---

## 11. 검증 정책

코드를 수정한 뒤 전체 학습을 바로 실행하지 않는다.

가능하면 다음 순서로 검증한다.

1. import 및 syntax 검사
2. config 로딩 검사
3. 데이터 1 batch 로딩
4. forward pass 검사
5. loss 계산 검사
6. backward pass 검사
7. 짧은 smoke test
8. 전체 학습 실행

예시:

```text
1 epoch
10 steps
작은 subset
단일 GPU
```

다중 GPU 학습 전에는 단일 GPU 실행이 정상 동작하는지 우선 확인한다.

Loss 또는 모델 구조를 변경한 경우 다음을 확인한다.

* NaN 또는 Inf 발생 여부
* gradient 존재 여부
* gradient norm 이상 여부
* tensor shape
* device mismatch
* 다중 GPU 간 batch 처리
* checkpoint 저장 및 로드

---

## 12. 실험 종료 및 결과 요약

각 실험이 끝나면 실험 폴더에 `result_summary.md`를 작성한다.

예시:

```markdown
# Result Summary

## Experiment

MoCo queue size 4096

## Date

2026-07-29

## Hypothesis

Queue 기반 negative sample이 representation 학습을 개선할 것이다.

## Main Settings

- GPUs: 4 × NVIDIA A10
- Batch size: 32 per GPU
- Effective batch size: 128
- Queue size: 4096
- EMA momentum: 0.9992
- Epochs: 200

## Best Result

- Validation loss:
- Validation accuracy:
- Positive similarity:
- Maximum negative similarity:

## Baseline Comparison

| Metric | Baseline | Experiment | Difference |
|---|---:|---:|---:|
| Validation loss | | | |
| Validation accuracy | | | |
| Positive similarity | | | |
| Maximum negative similarity | | | |

## Conclusion

<!-- 가설이 지지되었는지 작성 -->

## Issues

<!-- 학습 중 발생한 문제 작성 -->

## Next Experiment

<!-- 다음 실험 제안 작성 -->
```

Codex는 결과 수치만 보고 성공 여부를 단정하지 않는다.

평가 기준이 불명확한 경우 `setting_history.md`에 질문을 작성한다.

---

## 13. Git 변경 정책

가능하면 각 실험의 주요 변경은 독립적인 commit으로 관리한다.

권장 commit 예시:

```text
feat(moco): add momentum encoder and queue
exp(moco): add queue size 4096 configuration
fix(moco): correct distributed queue synchronization
docs(moco): record experiment settings and results
```

하나의 commit에 여러 실험의 변경을 섞지 않는다.

실험 실행 전에 현재 코드 상태를 commit하고 해당 commit hash를 실험 폴더에 기록한다.

---

## 14. Codex 작업 보고 형식

코드 작업을 완료한 뒤 Codex는 다음 내용을 보고한다.

```markdown
## 변경 내용

- 수정한 파일
- 추가한 파일
- 삭제한 파일
- 삭제한 코드의 이유

## 실험 조건 영향

- 변경된 실험 조건
- 유지된 Baseline 조건
- 재현성에 영향을 줄 수 있는 사항

## 검증 결과

- 수행한 테스트
- 성공 여부
- 아직 검증하지 못한 항목

## 확인이 필요한 사항

- `setting_history.md`에 추가한 질문
- 실험자의 답변이 필요한 결정
```

단순히 “완료했습니다”라고만 응답하지 않는다.

---

## 15. 금지 사항

Codex는 다음 작업을 수행하지 않는다.

* `Base/Baseline`을 사용자 승인 없이 수정
* 기존 실험 결과 자동 삭제
* 기존 체크포인트 자동 덮어쓰기
* 중요한 실험 설정을 임의로 결정
* 서로 다른 핵심 가설을 하나의 Lab에 무분별하게 혼합
* 실행한 config를 기록하지 않고 학습 수행
* Baseline과 달라진 조건을 기록하지 않고 비교
* 동작 확인 없이 전체 장시간 학습 실행
* 현재 실험과 무관한 대규모 리팩터링
* 결과가 좋다는 이유만으로 실패 run이나 불리한 결과를 삭제
* 코드 안에 서버 사용자명이나 개인 절대 경로를 새로 하드코딩
* 실행 중인 실험의 소스 코드를 추적 없이 수정
* `setting_history.md`를 한국어가 아닌 언어로 작성

---

## 16. 정책 우선순위

정책 간 충돌이 발생할 경우 다음 우선순위를 따른다.

1. 실험자의 명시적인 최신 지시
2. `setting_history.md`에서 해결된 결정
3. 이 `exp_setting.md`
4. 해당 Lab의 `README.md`
5. 기존 코드의 기본 동작
6. Codex의 자체 판단

Codex의 판단보다 실험자가 기록한 결정이 항상 우선한다.
