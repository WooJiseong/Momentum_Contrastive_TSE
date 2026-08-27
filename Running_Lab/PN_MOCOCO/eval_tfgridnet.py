from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/pn_mococo_numba_cache_{os.environ.get('USER', 'user')}")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pn_mococo.paths import add_repo_paths, project_path

add_repo_paths()

from data.datasets import build_test_dataset
from pn_mococo.moco_encoder import ensure_channel
from pn_mococo.tfgridnet_model import build_model_from_config, causal_forward
from Base.Code_Snippet.metrics_code import align_output_polarity, reference_metric_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PN_MOCOCO TFGridNet extractor.")
    parser.add_argument("--config", default="configs/config_tfgridnet_supervised.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def parse_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return normalize_paths(cfg)


def normalize_paths(cfg: dict) -> dict:
    cfg = dict(cfg)
    paths = cfg.setdefault("paths", {})
    for key in ("initial_model_ckpt", "encoder_override_ckpt"):
        if paths.get(key):
            paths[key] = project_path(paths[key])
    eval_cfg = cfg.setdefault("eval", {})
    for key in ("checkpoint", "out_json"):
        if eval_cfg.get(key):
            eval_cfg[key] = project_path(eval_cfg[key])
    return cfg


def load_eval_checkpoint(model: torch.nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Eval checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def write_result_summary(result: dict, out_json: str | None, config: dict) -> None:
    if not out_json:
        return
    out_path = Path(out_json)
    run_dir = out_path.parent.parent if out_path.parent.name == "evaluation" else out_path.parent
    eval_cfg = config.get("eval", {})
    experiment = config.get("experiment", {})
    summary = result["summary"]
    lines = [
        "# Result Summary",
        "",
        "## Experiment",
        "",
        str(experiment.get("name", "pn_mococo")),
        "",
        "## Main Settings",
        "",
        f"- Checkpoint: {result.get('checkpoint')}",
        f"- Evaluation mode: {eval_cfg.get('mode', 'fixed_count')}",
        f"- Evaluation items: {result.get('n')}",
        f"- Configured test_n: {eval_cfg.get('test_n')}",
        "- Full eval status: unavailable for the top-level online mixer; fixed-count deterministic evaluation is used.",
        "",
        "## Best Result",
        "",
        f"- SI-SDR: {summary.get('si_sdr'):.6f}",
        f"- SI-SDRi: {summary.get('si_sdri'):.6f}",
        f"- SDR (Base): {summary.get('sdr'):.6f}",
        f"- SDRi (Base): {summary.get('sdri'):.6f}",
        f"- SI-SNR: {summary.get('si_snr'):.6f}",
        f"- SI-SNRi: {summary.get('si_snri'):.6f}",
        f"- SNR: {summary.get('snr'):.6f}",
        f"- SNRi: {summary.get('snri'):.6f}",
        "",
        "## Baseline Comparison",
        "",
        "| Metric | Baseline | PN_MOCOCO | Difference |",
        "|---|---:|---:|---:|",
        "| SI-SDR | | | |",
        "| SI-SDRi | | | |",
        "| SNR | | | |",
        "| SNRi | | | |",
        "",
        "## Issues",
        "",
        "- Full file-manifest evaluation is not available through data.datasets.build_test_dataset.",
        "",
        "## Next Experiment",
        "",
        "- Fill the baseline comparison after running the baseline with the same eval.test_n and dataset config.",
        "",
    ]
    summary_path = run_dir / "result_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {summary_path}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = parse_config(args.config)
    checkpoint = args.checkpoint or config.get("eval", {}).get("checkpoint")
    if checkpoint:
        checkpoint = project_path(checkpoint) if not Path(checkpoint).is_absolute() else checkpoint
    out_json = args.out or config.get("eval", {}).get("out_json")
    if out_json:
        out_json = project_path(out_json) if not Path(out_json).is_absolute() else out_json

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    model = build_model_from_config(config)
    if checkpoint:
        load_eval_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()

    dataset = build_test_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("eval", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("eval", {}).get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    chunk_samples = int(config.get("inference", {}).get("chunk_samples", config["dataset"]["sample_rate"]))
    sign_correction = bool(config.get("eval", {}).get("sign_correction", True))

    metric_keys = ("si_sdr", "si_sdri", "sdr", "sdri", "si_snr", "si_snri", "snr", "snri")
    totals = {key: 0.0 for key in metric_keys}
    count = 0
    rows = []

    for batch in tqdm(loader, unit="batch"):
        mixture = ensure_channel(batch["mixture"]).to(device)
        target = batch["source"].to(device).float()
        pos = ensure_channel(batch["pos_wave"]).to(device)
        neg = ensure_channel(batch["neg_wave"]).to(device)
        est = causal_forward(model, mixture, pos, neg, chunk_samples)
        if sign_correction:
            est, _ = align_output_polarity(est, target)
        mix_wave = mixture.squeeze(1)
        metrics, _, _ = reference_metric_batch(
            est, target, mix_wave, sign_correction=False
        )
        bsz = int(target.shape[0])
        for key in metric_keys:
            value = metrics[key]
            totals[key] += float(value.detach().sum().cpu())
        count += bsz
        utt_ids = batch.get("utt_id", [""] * bsz)
        for i in range(bsz):
            rows.append({
                "utt_id": str(utt_ids[i]),
                "si_sdr": float(metrics["si_sdr"][i].detach().cpu()),
                "si_sdri": float(metrics["si_sdri"][i].detach().cpu()),
                "sdr": float(metrics["sdr"][i].detach().cpu()),
                "sdri": float(metrics["sdri"][i].detach().cpu()),
                "si_snr": float(metrics["si_snr"][i].detach().cpu()),
                "si_snri": float(metrics["si_snri"][i].detach().cpu()),
                "snr": float(metrics["snr"][i].detach().cpu()),
                "snri": float(metrics["snri"][i].detach().cpu()),
            })

    summary = {key: value / max(1, count) for key, value in totals.items()}
    result = {
        "checkpoint": checkpoint,
        "n": count,
        "eval": {
            "mode": config.get("eval", {}).get("mode", "fixed_count"),
            "test_n": config.get("eval", {}).get("test_n"),
            "full_eval": False,
            "full_eval_reason": "The top-level online mixer exposes deterministic fixed-count pseudo-mixtures, not a finite full-test manifest.",
        },
        "summary": summary,
        "items": rows,
    }
    print(json.dumps({"n": count, **summary}, indent=2))

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_json}")
        write_result_summary(result, out_json, config)


if __name__ == "__main__":
    main()
