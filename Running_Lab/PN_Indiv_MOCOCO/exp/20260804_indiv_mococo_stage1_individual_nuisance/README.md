# PN_Indiv_MOCOCO Stage1: Individual Nuisance Leakage

This directory is created for Stage1 training initialized from the completed
`20260802_indiv_mococo` Stage0 best PN encoder export (epoch 88).

The runtime script records the resolved configuration, source config, Stage0
encoder checksum, TensorBoard data, full Lightning checkpoints, and logs here.

Submit from `Running_Lab/PN_Indiv_MOCOCO`:

```bash
sbatch script/slurm_tfgridnet_stage1_20260804.sh
```

Evaluate the completed Stage1 checkpoint on 5,000 fixed-count test mixtures:

```bash
sbatch script/slurm_eval_stage1_5000_20260806.sh
```

Results are written to `evaluation/results.json` and `result_summary.md`.
