# PN_MOCOCO

PN_MOCOCO is an isolated experiment repo for replacing the slow PNFlowTSE Flow Matching decoder with the original PN-Enroll pretrained causal TFGridNet extractor.

The project starts from `contrastive_momentum/TSE-through-Positive-Negative-Enroll`, but it reuses the concrete PNFlowTSE data contract:

- speech data: `../../../data/LibriSpeech/{train-clean-360,train-clean-100,dev-clean,test-clean}`
- noise data: `../../../data/wham_noise/{tr,cv,tt}`
- initial teacher/separator checkpoint: `../../../checkpoints/proposed-monaural.pt`
- online mixture implementation: top-level `data/datasets.py`

## Pipeline

```bash
cd contrastive_momentum/Running_Lab/PN_MOCOCO

# Full train -> eval cycle. GPU count comes from YAML ddp.num_gpus.
bash script/train_eval.sh

# Individual stages are also available.
bash script/train_eval.sh moco
bash script/train_eval.sh tfgridnet
bash script/train_eval.sh eval

# Submit evaluation from a login/CPU node, or run on this compute node if CUDA is visible.
bash script/submit_eval_slurm.sh

# Verify a completed eval run.
python script/check_eval_result.py --run-dir exp/YYYYMMDD_pn_mococo_eval
```

## Hypothesis

Task-aware momentum contrastive training of the PN encoder improves the
positive/negative enrollment representation used by the original causal TFGridNet
extractor.

## Baseline

`Running_Lab/TSE-through-Positive-Negative-Enroll`

## Main Difference

- Stage 0 trains a momentum encoder with a negative queue.
- Stage 1 fine-tunes the original causal TFGridNet separator using the exported MoCo PN encoder.
- The Flow Matching decoder is not used in this Lab.

## Controlled Conditions

- Dataset: same top-level LibriSpeech+WHAM online mixer as the baseline Lab.
- Batch size: 2 per GPU, gradient accumulation 4.
- GPUs: 4 by default.
- Optimizer: AdamW.
- TFGridNet training epochs: 200.
- Checkpoint selection: minimum validation loss.

## Evaluation Metrics

- SI-SDR
- SI-SDRi
- SNR
- SNRi

## Main Outputs

Each run creates a new `exp/YYYYMMDD_pn_mococo[_runNN]/` folder containing:

- `config_moco_encoder.yaml`
- `config_tfgridnet_supervised.yaml`
- copied source configs
- `command.txt`
- `environment.txt`
- `stage0_moco/checkpoints/pn_encoder_best.pt`
- `stage1_tfgridnet/checkpoints/tfgridnet_best.ckpt`
- `evaluation/results.json`
- `result_summary.md` after evaluation

The MOCO export keeps the original PN checkpoint format:

```python
torch.load(path)["state_dict"]  # keys: encoder.* and encoder_head.*
```

That means the same encoder can be used by the original TFGridNet path and, later, by the PNFlowTSE Flow decoder if you want to switch back to a generative decoder.

## Configs

- `configs/config_moco_encoder.yaml`
- `configs/config_tfgridnet_supervised.yaml`

The default configs use `ddp.num_gpus: 4`. To use a different GPU count, edit both YAML files:

```yaml
ddp:
  use_ddp: true
  num_gpus: 2
train:
  batch_size: 2
  accumulation_steps: 8
```

Then expose the same number of GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash script/train_eval.sh
```

By default, stage 1 loads the full pretrained separator from:

```text
../../../checkpoints/proposed-monaural.pt
```

and then overrides only `encoder.*` / `encoder_head.*` from:

```text
stage0_moco/checkpoints/pn_encoder_best.pt
```

To train the TFGridNet baseline without MOCO, set:

```yaml
paths:
  encoder_override_ckpt: ""
```

## Notes

- Legacy PN-Enroll training/evaluation entrypoints were removed from this Lab; use `script/` and the YAML configs above.
- The top-level PNFlowTSE online mixer is used so Flow and TFGridNet experiments see the same LibriSpeech+WHAM distribution.
- Evaluation uses `eval.mode: fixed_count`, `eval.test_n: 5000`, and `eval.batch_size: 4`. A true full test-manifest evaluation is not available through the top-level online mixer.
- `script/run_eval.sh` requires a CUDA-visible GPU by default. Set `ALLOW_CPU_EVAL=1` only when intentionally running a small debug eval on CPU.
- Use `script/submit_eval_slurm.sh` for the default 5000-item evaluation. On `gate*` hosts it submits Slurm; on a non-`gate*` compute host it uses the current node when CUDA is visible and otherwise falls back to Slurm.
- Do not install the full `asteroid` package. The original causal TFGridNet code only needs `asteroid-filterbanks`.
