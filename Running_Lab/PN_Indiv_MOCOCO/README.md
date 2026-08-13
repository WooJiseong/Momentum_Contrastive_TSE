# PN_Indiv_MOCOCO

Teacher embedding imitation checkpoint에 대해 개별 negative enrollment을 유지하는
momentum contrastive pretraining을 수행하는 별도 Lab이다.

이 Lab은 `exp_setting.md`의 REAL 데이터 규칙을 따른다. `neg_cond`에 포함된 각
negative speaker와 WHAM noise를 합산하지 않고 개별 negative key로 encode한다.

## Objective

각 query는 기존 PN encoder 경로로 만든 positive/negative 조건 embedding이다.
positive key와의 similarity는 높이고, 각 negative enrollment key와의 similarity는
낮춘다. negative key들의 전체 contribution은 `1/n`으로 평균화한다.

```text
L = CE([positive_similarity, logmeanexp(negative_similarities)])
```

따라서 negative speaker 수가 늘어도 negative 항의 총 weight가 증가하지 않는다.
WHAM noise row도 별도의 negative item으로 취급한다.

Stage1 TFGridNet은 기존 PN_MOCOCO supervised trainer 조건을 유지하되,
`Base/Code_Snippet/loss_code.py`의 `target_orthogonal_leakage_loss`를
SI-SDR loss에 더해 target-orthogonal leakage를 줄인다.

## Run

```bash
cd contrastive_momentum/Running_Lab/PN_Indiv_MOCOCO
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29820 bash script/train_eval_20260801.sh all
```

Stage1은 기존 PN_MOCOCO의 supervised TFGridNet trainer와 동일한 조건으로 실행하며,
`paths.encoder_override_ckpt`만 이 Lab의 Stage0 export를 가리킨다.

## Slurm

```bash
cd contrastive_momentum/Running_Lab/PN_Indiv_MOCOCO
mkdir -p exp/20260801_indiv_mococo
sbatch script/slurm_train_eval_20260801.sh
```

## Output

모든 산출물은 `exp/YYYYMMDD_*` 아래에 저장한다.
