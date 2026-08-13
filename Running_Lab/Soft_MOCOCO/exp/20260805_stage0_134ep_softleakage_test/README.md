# Soft_MOCOCO Stage0 Epoch-134 Sweep

The original epoch-100 Stage0 encoder export was overwritten by the continuing
Stage0 job. The closest completed encoder still available is the validation-best
epoch-134 export. `pn_encoder_134ep.pt` is a frozen copy made before Stage1
submission.

This directory contains the Stage1 config snapshot, encoder checksum, logs,
TensorBoard events, full Lightning checkpoints, and evaluation output.

Submit from `Running_Lab/Soft_MOCOCO`:

```bash
sbatch script/slurm_stage1_softleakage_20260805_stage0_134ep.sh
```
