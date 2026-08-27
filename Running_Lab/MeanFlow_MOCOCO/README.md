# MeanFlow_MOCOCO

`../sia_fm_tse`의 MeanFlow Stage1 Decoder와 학습 루프를 사용하고, 현재 MOCOCO Lab에서 export한
`pn_encoder*.pt`를 frozen enrollment encoder로 주입하는 실험 Lab이다.

기본 설정은 고정-count 5000 Eval에서 가장 좋은 Soft_MOCOCO Stage0 50-epoch checkpoint를 가리킨다.
Stage0 종류를 바꾸고 싶으면 Slurm 제출 시 `STAGE0_CKPT`를 지정한다.

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/MeanFlow_MOCOCO
sbatch script/slurm_train_meanflow_mococo.sh

# PN/Indiv/Attn 등 다른 MOCOCO export checkpoint 사용
sbatch --export=ALL,STAGE0_CKPT=/absolute/path/to/pn_encoder_best.pt \
  script/slurm_train_meanflow_mococo.sh
```

학습 로그와 checkpoint는 `exp/20260816_soft_mococo_meanflow_stage1/` 아래에 저장된다.
50 epoch마다 `epoch_*.ckpt`를 저장하고 `val_loss` 기준 best 및 last checkpoint도 보존한다.

## 현재 MeanFlow 비교 실험의 주의점

기존 `20260818_*default_flow*` 실행은 아래 이유로 기본 Stage1과 공정한 비교가
아니다.

1. `t_predicter_best.ckpt`는 `proposed-monaural.pt`로 학습됐지만 flow 본체는
   Soft_MOCOCO `pn_encoder_50ep.pt`를 사용했다. t-predictor는 PN embedding을 입력으로
   받으므로 두 checkpoint를 섞으면 예측한 시작 시각 `m_hat`의 입력 분포가 달라진다.
2. pure Rectified Flow 학습은 `r=t`만 보는데, 기존 validation 1-step은 `r=1`을
   전달했다. wrapper는 이 모드에서 validation도 `r=t`가 되도록 보정한다.
3. `loss.gamma=0`은 adaptive loss를 `delta/(delta+c)` 형태로 포화시킨다. 실제 로그의
   `train_loss≈0.999`는 이 포화 상태와 일치한다. decoder 자체를 비교할 때는 plain MSE에
   해당하는 `gamma=1`을 별도 실험으로 확인해야 한다.

새로 실행할 때 wrapper는 flow Stage0와 t-predictor의 학습 Stage0가 다르면 즉시 중단한다.
`paths.t_predicter_ckpt: null`은 true `mixing_ratio`를 쓰는 oracle 진단용으로만 사용한다.
실제 predicted 모드의 결과를 얻으려면 선택한 MOCOCO Stage0로 t-predictor를 먼저
재학습해야 한다.
