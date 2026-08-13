# 20260730 TFGridNet Depth Sweep Jobs

Fixed sweep settings:
- `num_epochs=50`
- `fusion_layer=list(range(n_layers))`
- `n_layers in {3, 4, 6}`
- 4-GPU DDP via `torchrun` with `SLURM_*` unset
- 5000-item eval runs automatically after training

| Condition | n_layers | Run dir | Slurm job |
|---|---:|---|---:|
| Baseline | 3 | `Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n3_fullfusion_50ep` | 849662 |
| MoCo | 3 | `Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n3_fullfusion_50ep` | 850424 |
| Baseline | 4 | `Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n4_fullfusion_50ep` | 849663 |
| MoCo | 4 | `Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n4_fullfusion_50ep` | 850425 |
| Baseline | 6 | `Running_Lab/TSE-through-Positive-Negative-Enroll/exp/20260730_depth_sweep_baseline_n6_fullfusion_50ep` | 849666 |
| MoCo | 6 | `Running_Lab/PN_MOCOCO/exp/20260730_depth_sweep_mococo_mocobest_n6_fullfusion_50ep` | 850426 |

Invalid/canceled MoCo submissions:
- `849664`, `849665`, `849667` used `pn_encoder_best.pt`, which was an epoch-0 export identical to `proposed-monaural.pt`; these were canceled and replaced with the `moco_best.ckpt` runs above.
