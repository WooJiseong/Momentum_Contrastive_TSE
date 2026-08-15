# Setting History

## 20260813

- Created `Attn_MOCOCO` as a new Lab based on `Soft_MOCOCO`.
- Replaced only the Stage0 Projection Head's temporal Global Mean with 40-frame,
  stride-40 Lightweight Attention Pooling.
- Kept the Soft_MOCOCO Teacher imitation term, EMA Momentum encoder, queue, dataset,
  optimizer, scheduler, and output embedding dimension unchanged.
- Added two independent Stage1 variants: SI-SDR only and SI-SDR plus revised
  target-orthogonal Leakage Loss.
- Added 50-epoch Stage0 encoder exports for Stage1 sweep experiments.

- 20260815: Fixed the Slurm wrapper to use the pnflowtse Python interpreter when
  `python` is absent. One-GPU Stage0, Stage1 SI-SDR, and Stage1 Leakage smoke tests
  completed successfully, including checkpoint export and nuisance reconstruction
  validation.
