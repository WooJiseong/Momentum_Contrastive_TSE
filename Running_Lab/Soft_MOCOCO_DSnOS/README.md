# Soft_MOCOCO_DSnOS

This Lab tests dataset-level split and oversampling while keeping the
Soft_MOCOCO model, loss, augmentation, optimizer, scheduler, queue, and
validation protocol unchanged.

The sampler chooses each configured `train_roots` entry with equal
probability, then uses the original online mixer inside the selected root.
For train-clean-360 + train-clean-100 this reproduces a 50:50 root split, so
the smaller train-clean-100 root is oversampled relative to its speaker count.
For one root, the speakers are first split into 78.6:21.4 partitions, matching
the speaker-count ratio of train-clean-360 versus train-clean-100, and the two
partitions are then sampled 50:50.

## Stage0

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO_DSnOS
sbatch script/slurm_stage0.sh
```

The experiment stores its config, TensorBoard events, checkpoints, and Slurm
output under `exp/20260901_soft_moco_dsnos_360only_split50/stage0_moco/`.
