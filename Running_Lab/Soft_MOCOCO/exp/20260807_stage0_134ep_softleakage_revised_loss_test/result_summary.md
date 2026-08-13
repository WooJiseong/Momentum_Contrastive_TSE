# Result Summary

## Experiment

soft_mococo_stage0_134ep_softleakage_revised_loss

## Main Settings

- Checkpoint: /gpfs/home1/wjs6800/SIALab/PNFlowTSE/contrastive_momentum/Running_Lab/Soft_MOCOCO/exp/20260807_stage0_134ep_softleakage_revised_loss_test/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt
- Evaluation mode: fixed_count
- Evaluation items: 5000
- Configured test_n: 5000
- Full eval status: unavailable for the top-level online mixer; fixed-count deterministic evaluation is used.

## Best Result

- SI-SDR: 2.848954
- SI-SDRi: 8.261197
- SNR: 4.781159
- SNRi: 10.189650

## Baseline Comparison

Reference baseline: `Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260729_tfgridnet_baseline_resume200_best_eval` (`tfgridnet_baseline_resume200_best`).

The reference uses the same causal TFGridNet architecture, dataset/noise configuration, `fixed_count` evaluation, and 5000 evaluation items. It resumes the original Baseline from its Early-Stopping checkpoint and continues to the 200-Epoch budget with Early Stopping disabled, making it the appropriate equal-budget Baseline for this Stage1 comparison. Difference is calculated as `Soft_MOCOCO - Baseline`.

| Metric | Baseline | Soft_MOCOCO | Difference |
|---|---:|---:|---:|
| SI-SDR | 2.808985 | 2.848954 | +0.039969 |
| SI-SDRi | 8.221228 | 8.261197 | +0.039969 |
| SNR | 4.815443 | 4.781159 | -0.034285 |
| SNRi | 10.223935 | 10.189650 | -0.034285 |

For reference, the original Early-Stopping-only Baseline (`20260729_tfgridnet_baseline_eval`) scored SI-SDR 2.770537 and SI-SDRi 8.182780. It is retained as a historical result, but is not the primary equal-budget comparison here.

## Issues

- Full file-manifest evaluation is not available through data.datasets.build_test_dataset.

## Next Experiment

- Evaluate the 159-Epoch Leakage Loss checkpoint with the same fixed-count protocol and compare it against this baseline.
