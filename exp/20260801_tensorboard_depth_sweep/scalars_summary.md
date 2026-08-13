# 50-epoch TensorBoard Scalar Summary

Source: 20260730 TFGridNet depth sweep event files.

| Run | Tag | Last step | Last value |
|---|---|---:|---:|
| baseline_n3 | epoch | 15649 | 49.000000 |
| baseline_n3 | hp_metric | 0 | -1.000000 |
| baseline_n3 | lr-AdamW | 15337 | 0.000004 |
| baseline_n3 | train_loss | 15649 | -2.754048 |
| baseline_n3 | train_si_sdr | 15649 | 2.754048 |
| baseline_n3 | train_si_sdri | 15649 | 8.320790 |
| baseline_n3 | val_loss | 15649 | -1.856118 |
| baseline_n3 | val_si_sdr | 15649 | 1.856118 |
| baseline_n3 | val_si_sdri | 15649 | 8.268104 |
| baseline_n3 | val_snr | 15649 | 4.077864 |
| baseline_n3 | val_snri | 15649 | 10.442901 |
| baseline_n4 | epoch | 15649 | 49.000000 |
| baseline_n4 | hp_metric | 0 | -1.000000 |
| baseline_n4 | lr-AdamW | 15337 | 0.000004 |
| baseline_n4 | train_loss | 15649 | -2.784317 |
| baseline_n4 | train_si_sdr | 15649 | 2.784317 |
| baseline_n4 | train_si_sdri | 15649 | 8.351057 |
| baseline_n4 | val_loss | 15649 | -1.888529 |
| baseline_n4 | val_si_sdr | 15649 | 1.888529 |
| baseline_n4 | val_si_sdri | 15649 | 8.300514 |
| baseline_n4 | val_snr | 15649 | 3.091010 |
| baseline_n4 | val_snri | 15649 | 9.456045 |
| baseline_n6 | epoch | 15649 | 49.000000 |
| baseline_n6 | hp_metric | 0 | -1.000000 |
| baseline_n6 | lr-AdamW | 15337 | 0.000004 |
| baseline_n6 | train_loss | 15649 | -2.793966 |
| baseline_n6 | train_si_sdr | 15649 | 2.793966 |
| baseline_n6 | train_si_sdri | 15649 | 8.360709 |
| baseline_n6 | val_loss | 15649 | -1.899788 |
| baseline_n6 | val_si_sdr | 15649 | 1.899788 |
| baseline_n6 | val_si_sdri | 15649 | 8.311773 |
| baseline_n6 | val_snr | 15649 | 0.743760 |
| baseline_n6 | val_snri | 15649 | 7.108797 |
| mococo_n3 | epoch | 15649 | 49.000000 |
| mococo_n3 | hp_metric | 0 | -1.000000 |
| mococo_n3 | lr-AdamW | 15337 | 0.000004 |
| mococo_n3 | train_loss | 15649 | -2.717218 |
| mococo_n3 | train_si_sdr | 15649 | 2.717218 |
| mococo_n3 | train_si_sdri | 15649 | 8.283962 |
| mococo_n3 | val_loss | 15649 | -1.870499 |
| mococo_n3 | val_si_sdr | 15649 | 1.870499 |
| mococo_n3 | val_si_sdri | 15649 | 8.282485 |
| mococo_n3 | val_snr | 15649 | 3.771448 |
| mococo_n3 | val_snri | 15649 | 10.136484 |
| mococo_n4 | epoch | 15649 | 49.000000 |
| mococo_n4 | hp_metric | 0 | -1.000000 |
| mococo_n4 | lr-AdamW | 15337 | 0.000004 |
| mococo_n4 | train_loss | 15649 | -2.745473 |
| mococo_n4 | train_si_sdr | 15649 | 2.745473 |
| mococo_n4 | train_si_sdri | 15649 | 8.312217 |
| mococo_n4 | val_loss | 15649 | -1.894936 |
| mococo_n4 | val_si_sdr | 15649 | 1.894936 |
| mococo_n4 | val_si_sdri | 15649 | 8.306921 |
| mococo_n4 | val_snr | 15649 | 2.425245 |
| mococo_n4 | val_snri | 15649 | 8.790280 |
| mococo_n6 | epoch | 15649 | 49.000000 |
| mococo_n6 | hp_metric | 0 | -1.000000 |
| mococo_n6 | lr-AdamW | 15337 | 0.000004 |
| mococo_n6 | train_loss | 15649 | -2.750139 |
| mococo_n6 | train_si_sdr | 15649 | 2.750139 |
| mococo_n6 | train_si_sdri | 15649 | 8.316879 |
| mococo_n6 | val_loss | 15649 | -1.900875 |
| mococo_n6 | val_si_sdr | 15649 | 1.900875 |
| mococo_n6 | val_si_sdri | 15649 | 8.312860 |
| mococo_n6 | val_snr | 15649 | -0.557605 |
| mococo_n6 | val_snri | 15649 | 5.807431 |

## TensorBoard

Run from the repository root:

```bash
tensorboard --logdir=Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n3_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n4_fullfusion_50ep/stage1_tfgrid_sweep_mococo_mocobest_n4_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0,Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n6_fullfusion_50ep/stage1_tfgridnet/lightning_logs/0
```