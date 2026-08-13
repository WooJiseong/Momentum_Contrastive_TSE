# Stage0 50 Epoch Soft Without Leakage

This run uses the same pinned Soft_MOCOCO Stage0 epoch-50 export as
`20260803_stage0_50ep_softleakage_test`, but trains Stage1 with only the
standard SI-SDR objective. The target-orthogonal leakage loss is not called.

Input checkpoint:

```text
stage0_moco/checkpoints/pn_encoder_50ep.pt
SHA256: c8f5d5d3c2c14443b336264c5633fbf6d69470b8663882a191d686a3e745fdeb
```

Run locally:

```bash
cd contrastive_momentum/Running_Lab/Soft_MOCOCO
CUDA_VISIBLE_DEVICES=0,1,2,3 bash script/train_stage1_wo_leakage_20260803.sh all
```

Submit to Slurm:

```bash
cd contrastive_momentum/Running_Lab/Soft_MOCOCO
sbatch script/slurm_stage1_wo_leakage_20260803.sh
```
