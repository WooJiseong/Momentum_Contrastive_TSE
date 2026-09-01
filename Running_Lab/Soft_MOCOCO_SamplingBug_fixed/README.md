# Soft_MOCOCO_SamplingBug_fixed

This Lab tests the effect of dataset construction in Soft_MOCOCO Stage0.

The original adapter creates one `NoisyFlowTSEDataset` per configured root and
selects one root with equal probability. This Lab keeps the original mixer
operations, but builds one global speaker/file pool across all train roots.
Target, mixture interferers, and enrollment speakers are sampled from that
global pool. With one root, the same implementation reduces to a single
360-only pool.

The Soft_MOCOCO loss, frozen Teacher, augmentation, optimizer, scheduler,
queue, batch size, DDP setup, and validation split are unchanged.

## Stage0

```bash
cd /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO_SamplingBug_fixed
sbatch script/slurm_stage0_360.sh
sbatch script/slurm_stage0_360plus100.sh
```

Each experiment stores its config, TensorBoard events, checkpoints, and Slurm
output under its own `exp/YYYYMMDD_experiment_name/` directory.

The validation set remains `LibriSpeech/dev-clean` with `wham_noise/cv`; only
the train speaker-pool construction changes between the old and new adapters.
