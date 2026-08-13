# Soft_MOCOCO Stage0 Epoch-134 Revised Leakage Loss

This is a clean Stage1 experiment initialized from the preserved Stage0
epoch-134 encoder. It does not resume the previous Stage1 checkpoint because
that checkpoint used the earlier Leakage Loss implementation.

Stage1 uses the revised Leakage Loss with residual-energy normalization and
target/nuisance activity gating. The Stage1 checkpoint and optimizer state are
created from scratch with `checkpoint.resume: null`.

The frozen Stage0 encoder is copied to:

```text
stage0_moco/checkpoints/pn_encoder_134ep.pt
```

Submit from `Running_Lab/Soft_MOCOCO`:

```bash
sbatch script/slurm_stage1_softleakage_20260807_stage0_134ep_revised.sh
```
