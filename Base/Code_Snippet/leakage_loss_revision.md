# Target-Orthogonal Leakage Loss Revision

- Date: 2026-08-07
- Scope: `target_orthogonal_leakage_loss`
- Affected Labs: `PN_Indiv_MOCOCO`, `Soft_MOCOCO`, `Soft_Indiv_MOCOCO`

## Why This Revision Was Needed

Stage1 used the following objective:

```text
L = -SI-SDR + 0.1 * L_leakage
```

The original leakage term was:

```text
r = est - proj_target(est)
b_perp = nuisance - proj_target(nuisance)
```
$$L_{\text{leakage}} = \frac{\langle r, b_{\perp} \rangle^2}{\Vert{}b_{\perp}\Vert{}^2 \Vert{} \text{target} \Vert{}^2}$$

The residual energy `||r||^2` was missing from the denominator. For an
estimate scaled by `c`, `est' = c * est`, SI-SDR is approximately unchanged,
while the leakage term scales as `c^2`. This created an optimization path that
could reduce output amplitude and still decrease the total loss. A CPU demo
from `PN_Indiv_MOCOCO` confirmed that the estimate RMS was only about 2.2% of
the target RMS.

The original gate only checked whether a nuisance was sufficiently orthogonal
to the target. It did not remove windows where the target or nuisance energy
was near zero. Those windows can make projection and ratio calculations poorly
conditioned. In the affected validation log, the orthogonality gate ratio was
1.0 for the entire run.

## Changes

### 1. Residual Energy Normalization

The default loss now uses:

```text
L_leakage = <r, b_perp>^2
             -----------------------------------------
             ||b_perp||^2 * ||r||^2
```
$$L_{\text{leakage}} = \frac{\langle r, b_{\perp} \rangle^2}{\Vert{}b_{\perp}\Vert{}^2 \Vert{} \text{r} \Vert{}^2}$$
This is the squared cosine similarity between the target-orthogonal residual
and target-orthogonal nuisance. Scaling `est` no longer reduces this term by
itself. The value is bounded by approximately `[0, 1]` when the inputs are
finite and nonzero.

The implementation keeps the old target-energy denominator available through:

```yaml
loss:
  normalize_residual_energy: false
```

That option is intended only for an explicit legacy comparison.

### 2. Target/Nuisance Activity Gate

When enabled, the final gate is:

```text
orthogonality_gate
AND target_activity_gate
AND nuisance_activity_gate
```

The activity thresholds are relative to the maximum window energy within each
sample or nuisance source, so they do not depend on the absolute waveform
normalization:

```text
target_active[w] = target_energy[w]
                   >= target_threshold * max_w(target_energy[w])

nuisance_active[j,w] = nuisance_energy[j,w]
                       >= nuisance_threshold * max_w(nuisance_energy[j,w])
```

The new defaults are:

```yaml
loss:
  activity_gate: true
  target_activity_threshold: 0.01
  nuisance_activity_threshold: 0.01
```

The all-zero reference case is rejected by the gate even when the relative
threshold is zero. `eps` remains a numerical stabilizer; it is not used as a
substitute for the activity gate.

## Logged Diagnostics

The three Stage1 trainers now expose the following additional TensorBoard
scalars:

```text
train_orthogonal_target_activity_ratio
val_orthogonal_target_activity_ratio
train_orthogonal_nuisance_activity_ratio
val_orthogonal_nuisance_activity_ratio
```

Existing `orthogonal_gate_ratio` now describes the combined gate. The
`nuisance_reconstruction_mae` contract remains unchanged and must still be
near zero.

## Interpretation and Limitation

Residual normalization removes the direct incentive to shrink the estimate,
but it does not itself provide a positive amplitude target. The current Stage1
objective still contains `-SI-SDR`, which is scale invariant. Therefore every
new run must also inspect `est_rms / target_rms`, `val_snr`, and absolute
`val_si_sdr`. If amplitude remains suppressed, add a separately controlled
scale-sensitive term such as `-SNR`, log-RMS loss, or waveform reconstruction
loss rather than silently changing this leakage term again.

## Reproducing the Legacy Formula

For an explicit legacy ablation, use:

```yaml
loss:
  normalize_residual_energy: false
  activity_gate: false
```

The default behavior is intentionally the revised behavior described above.
