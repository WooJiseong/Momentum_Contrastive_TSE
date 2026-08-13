# Tools

## Sample Audio Maker

Create a deterministic 10-item TSE audio demo and per-sample YAML metrics in an
experiment-local `example/` directory. See [sample_audio_mkr/README.md](sample_audio_mkr/README.md).

## Watch Experiment Metrics

Render TensorBoard validation scalars to an experiment-local PNG and refresh it
without the TensorBoard web UI.

```bash
/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python tools/watch_experiment_metrics.py \
  --project PN_Indiv_MOCOCO \
  --experiment 20260802_indiv_mococo \
  --stage stage0_moco \
  --watch-seconds 30
```

The default graph name is
`<experiment>/<stage>/<experiment>_<stage>_validation_metrics.png`. Use
`--tags val_loss val_acc` to choose metrics, or `--out <path>` to select a
different PNG path within the experiment directory.

An event file can also be used directly. Its full path is accepted, and a
unique `events.out.tfevents.*` filename is searched under the repository.

```bash
/home1/wjs6800/miniconda3/envs/pnflowtse/bin/python tools/watch_experiment_metrics.py \
  --event events.out.tfevents.1784814643.n038.hpc.1567693.0 \
  --watch-seconds 30
```
