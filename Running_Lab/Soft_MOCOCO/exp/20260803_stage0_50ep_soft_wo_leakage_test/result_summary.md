# Result Summary

## Experiment

stage0_50ep_soft_wo_leakage_test

## Main Settings

- Checkpoint: /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260803_stage0_50ep_soft_wo_leakage_test/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt
- Evaluation mode: fixed_count
- Evaluation items: 5000
- Configured test_n: 5000
- Full eval status: unavailable for the top-level online mixer; fixed-count deterministic evaluation is used.

## Best Result

- SI-SDR: 2.851979
- SI-SDRi: 8.264222
- SNR: 4.786998
- SNRi: 10.195490

## Baseline Comparison

| Metric | Baseline | Soft Stage0-50, no leakage |
|---|---:|---:|
| SI-SDR | 2.808985 | 2.851979 |
| SI-SDRi | 8.221228 | 8.264222 |
| SNR | 4.815443 | 4.786998 |
| SNRi | 10.223935 | 10.195490 |

## Issues

- Full file-manifest evaluation is not available through data.datasets.build_test_dataset.

## Next Experiment

- Fill the baseline comparison after running the baseline with the same eval.test_n and dataset config.
