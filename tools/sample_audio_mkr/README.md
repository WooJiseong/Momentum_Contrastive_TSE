# Sample Audio Maker

`make_samples.py` creates a small deterministic audio demo set for a completed
TFGridNet experiment. It is separate from the 5,000-item fixed-count evaluation:
the default exports only 10 items from the same deterministic `test-clean` online
mixer indices.

The default output is `<experiment>/example/` and contains:

- `sample_XXXX/mixture.wav`: TSE input mixture
- `sample_XXXX/target.wav`: reference target speech
- `sample_XXXX/positive_enrollment.wav` and `negative_enrollment.wav`: model conditions
- `sample_XXXX/estimate.wav`: TSE output
- `sample_XXXX/metrics.yaml`: SI-SDR, SI-SDRi, SNR, SNRi, input metrics, and audio metadata
- `summary_metrics.yaml`: averages over the demo items only
- `manifest.yaml` and `config_resolved.yaml`: reproducibility metadata

WAV files use IEEE float samples and no amplitude normalization, so relative
amplitudes are preserved. Existing `example/` artifacts are protected; use
`--overwrite` only when replacing them intentionally.

## Usage

Run from `contrastive_momentum` after the target Stage1 checkpoint is available.
Run it on a GPU compute node or inside a Slurm GPU allocation; model construction
and causal inference are not suitable for a login node's CPU memory limit. The
default `--device auto` selects `cuda:0` when available.

```bash
/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  tools/sample_audio_mkr/make_samples.py \
  --config Running_Lab/Soft_MOCOCO/exp/20260803_stage0_50ep_soft_wo_leakage_test/config_eval_runtime.yaml \
  --num-samples 10
```

The command creates:

```text
Running_Lab/Soft_MOCOCO/exp/20260803_stage0_50ep_soft_wo_leakage_test/example/
```

Use a source config and a specific checkpoint when a runtime config does not yet
exist, for example after a new Stage1 sweep completes:

```bash
/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python \
  tools/sample_audio_mkr/make_samples.py \
  --config Running_Lab/Soft_MOCOCO/configs/config_tfgridnet_soft_leakage_20260805_stage0_134ep_test.yaml \
  --checkpoint Running_Lab/Soft_MOCOCO/exp/20260805_stage0_134ep_softleakage_test/stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt \
  --num-samples 10
```

`--start-index 10` selects deterministic items 10 through 19. `--out` changes
the output directory, and `--overwrite` replaces a non-empty output directory.
The default `--backend auto` supports PN_MOCOCO-family labs and the original
`TSE-through-Positive-Negative-Enroll` baseline.
