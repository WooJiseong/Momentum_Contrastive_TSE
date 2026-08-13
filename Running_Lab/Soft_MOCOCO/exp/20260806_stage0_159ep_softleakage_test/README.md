# Soft_MOCOCO Stage0 Epoch-159 Sweep

This Stage1 sweep uses a frozen copy of the completed Stage0 epoch-159 encoder.
It is independent of the epoch-134 sweep and trains TFGridNet with the corrected
individual-nuisance Soft-leakage loss.

Frozen Stage0 artifacts:

- `stage0_moco/checkpoints/pn_encoder_159ep.pt`: Stage1 encoder input
- `stage0_moco/checkpoints/stage0_last_159ep.ckpt`: full Lightning checkpoint
  (`epoch=159`, `global_step=50080`) for future Stage0 resumption

Submit from `Running_Lab/Soft_MOCOCO`:

```bash
sbatch script/slurm_stage1_softleakage_20260806_stage0_159ep.sh
```
