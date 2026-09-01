# Setting History

## 20260901

- Created `Soft_MOCOCO_SamplingBug_fixed` to isolate the multi-root sampling
  behavior in Stage0.
- Reused the existing Soft_MOCOCO learner and PN_MOCOCO queue implementation.
- Replaced per-root uniform selection with a single namespaced global speaker
  pool. Mixture, enrollment, partial masking, WHAM noise, and padding retain
  the existing online mixer rules.
- The two runs use the same configuration except for `dataset.train_roots`.
