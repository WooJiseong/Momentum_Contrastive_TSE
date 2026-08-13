# Result Summary

## Experiment

soft_mococo_stage0_159ep_softleakage

## Main Settings

- Checkpoint: /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260806_stage0_159ep_softleakage_test/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt
- Evaluation mode: fixed_count
- Evaluation items: 5000
- Configured test_n: 5000
- Full eval status: unavailable for the top-level online mixer; fixed-count deterministic evaluation is used.

## Best Result

- SI-SDR: 2.841531
- SI-SDRi: 8.253774
- SNR: 4.776159
- SNRi: 10.184651

## Baseline Comparison

| Metric | Baseline | PN_MOCOCO | Difference |
|---|---:|---:|---:|
| SI-SDR | | | |
| SI-SDRi | | | |
| SNR | | | |
| SNRi | | | |

## Issues

- Full file-manifest evaluation is not available through data.datasets.build_test_dataset.

## Next Experiment

- Fill the baseline comparison after running the baseline with the same eval.test_n and dataset config.
