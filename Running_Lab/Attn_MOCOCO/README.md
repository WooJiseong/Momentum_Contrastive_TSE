# Attn_MOCOCO

`Soft_MOCOCO`를 기반으로 Stage0 Projection Head의 Global Mean을
40-frame Lightweight Attention Pooling으로 교체한 controlled experiment다.

## Hypothesis

모든 시간 프레임을 동일하게 평균내는 대신, 40-frame token별로 학습된 중요도를
계산하면 침묵·저에너지 구간의 영향은 줄이고 화자 정보가 강한 구간을 강조할 수 있다.

## Main Difference

- Positive/Negative Encoder와 Teacher imitation term은 `Soft_MOCOCO`와 동일
- Stage0 Projection Head:
  - 기존: `emb.mean(dim=2)`
  - 변경: `avg_pool1d(kernel_size=40, stride=40)` 후 token별 Lightweight Attention gate
- Full `T x T` Self-Attention은 사용하지 않음
- Stage1은 동일한 Stage0 export에 대해 두 변형을 별도 실행

## Stage1 Variants

1. `stage1_tfgridnet_si_sdr`: SI-SDR loss only
2. `stage1_tfgridnet_si_sdr_leakage`: SI-SDR + revised target-orthogonal Leakage Loss

Leakage 변형은 Negative Enrollment나 합산 background가 아니라, 실제 mixture를 구성한
개별 `sample[1:]` nuisance source를 사용하고 `sum(nuisance_sources) == mixture-source`
계약을 검증한다.

## Run

```bash
cd contrastive_momentum/Running_Lab/Attn_MOCOCO

# Stage0 only
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train.sh moco

# Stage1 only, after pn_encoder_best.pt exists
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train.sh stage1

# Complete workflow
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/run_train.sh all
```

Slurm:

```bash
sbatch script/slurm_train_attn_mococo.sh
```

Stage0 exports are written to:

```text
exp/20260813_attn_mococo_40frame_lightweight_attention/stage0_moco/checkpoints/
  pn_encoder_best.pt
  pn_encoder_last.pt
  pn_encoder_50ep.pt
  pn_encoder_100ep.pt
  ...
```

The two Stage1 variants have independent logs and checkpoints under the same experiment
directory, so their results cannot overwrite each other.

Stage0는 PN_MOCOCO의 speaker-aware queue를 재사용한다. 각 queue embedding에
target speaker ID를 함께 저장하고, 알려진 동일 화자 항목을 false negative로부터
마스킹한다.

`884321`의 이전 제출은 학습 코드에 진입하기 전에 gpu4 QOS의 사용자별 GPU 한도
(`QOSMaxGRESPerUser`)로 거절됐다. 현재 제출본은 batch shell에서 `python` 명령이
없는 환경도 처리하도록 pnflowtse Python 절대 경로를 사용하며, 기존 GPU 작업 종료
dependency를 걸어 QOS 충돌을 피한다.

gpu4 노드는 노드당 A6000 4장이고, 현재 Lightning DDP 설정의 `devices`는 노드별
GPU 수다. 따라서 3+4의 불균일한 2노드 7GPU 실행은 현재 코드와 Slurm GRES로
표현할 수 없다. 이 실험은 `num_nodes: 1`, `num_gpus: 4`로 제출했다. 2노드 DDP를
추가하려면 양쪽 노드에 동일한 GPU 수를 할당하고 world size를 그에 맞춰 별도로
검증해야 한다.
