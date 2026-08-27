#!/usr/bin/env python
"""Create a small, deterministic audio demo set for a completed TSE experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import soundfile as sf
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNING_LAB = ROOT / "Running_Lab"
PNFLOW_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write deterministic TSE audio samples and per-sample metrics into an experiment/example directory."
    )
    parser.add_argument("--config", required=True, help="Runtime or source TFGridNet YAML configuration.")
    parser.add_argument("--checkpoint", help="Override eval.checkpoint from the configuration.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of fixed test items to export.")
    parser.add_argument("--start-index", type=int, default=0, help="First deterministic fixed-test item index.")
    parser.add_argument("--out", help="Output directory. Defaults to <experiment>/example.")
    parser.add_argument(
        "--backend",
        choices=("auto", "pn_mococo", "baseline"),
        default="auto",
        help="Model backend. auto selects baseline only for TSE-through-Positive-Negative-Enroll.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a specific CUDA device.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing example directory. Required to discard prior demo artifacts.",
    )
    return parser.parse_args()


def add_paths(*paths: Path) -> None:
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def lab_dir_for(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if candidate.parent == RUNNING_LAB:
            return candidate
    raise ValueError(
        f"Cannot infer a Running_Lab directory for {config_path}. "
        "Pass a config stored below Running_Lab/<Lab_Name>."
    )


def merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


def read_config(config_path: Path, lab_dir: Path) -> dict[str, Any]:
    override = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(override, dict):
        raise ValueError(f"YAML configuration must contain a mapping: {config_path}")
    if "base_config" not in override:
        return copy.deepcopy(override)

    base_path = Path(override["base_config"])
    if not base_path.is_absolute():
        base_path = (lab_dir / base_path).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"Base configuration must contain a mapping: {base_path}")
    config = copy.deepcopy(base)
    merge(config, override)
    return config


def resolve_path(value: str | Path | None, lab_dir: Path) -> str | None:
    if value is None or value == "":
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (lab_dir / path).resolve())


def resolve_config_paths(config: dict[str, Any], lab_dir: Path) -> None:
    for section, keys in {
        "paths": ("initial_model_ckpt", "encoder_override_ckpt"),
        "train": ("log_dir",),
        "checkpoint": ("dir", "resume"),
        "eval": ("checkpoint", "out_json"),
    }.items():
        for key in keys:
            if config.get(section, {}).get(key):
                config[section][key] = resolve_path(config[section][key], lab_dir)


def default_output_dir(config: dict[str, Any], lab_dir: Path) -> Path:
    log_dir = config.get("train", {}).get("log_dir")
    if not log_dir:
        raise ValueError("config.train.log_dir is required to place example artifacts in the experiment directory.")
    stage_dir = Path(resolve_path(log_dir, lab_dir))
    if stage_dir.name.startswith("stage"):
        return stage_dir.parent / "example"
    return stage_dir / "example"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_backend(name: str, lab_dir: Path) -> tuple[Callable, Callable, Callable, str]:
    if name == "auto":
        name = "baseline" if lab_dir.name == "TSE-through-Positive-Negative-Enroll" else "pn_mococo"

    if name == "pn_mococo":
        pn_mococo_dir = RUNNING_LAB / "PN_MOCOCO"
        add_paths(lab_dir, pn_mococo_dir, ROOT, PNFLOW_ROOT)
        from pn_mococo.moco_encoder import ensure_channel
        from pn_mococo.tfgridnet_model import build_model_from_config, causal_forward

        return build_model_from_config, causal_forward, ensure_channel, name

    baseline_dir = RUNNING_LAB / "TSE-through-Positive-Negative-Enroll"
    add_paths(lab_dir, baseline_dir, ROOT, PNFLOW_ROOT)
    from tse_pn_enroll.tfgridnet_model import build_model_from_config, causal_forward, ensure_channel

    return build_model_from_config, causal_forward, ensure_channel, "baseline"


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if any(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    return device


def batchify(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
        for key, value in item.items()
    }


def one_dimensional(wave: torch.Tensor, name: str) -> torch.Tensor:
    wave = wave.detach().float().cpu().squeeze()
    if wave.ndim != 1:
        raise ValueError(f"{name} must reduce to one waveform, got shape {tuple(wave.shape)}")
    if not torch.isfinite(wave).all():
        raise FloatingPointError(f"{name} contains NaN or Inf and will not be written.")
    return wave


def write_wav(path: Path, wave: torch.Tensor, sample_rate: int) -> None:
    sf.write(path, one_dimensional(wave, path.name).numpy(), sample_rate, subtype="FLOAT")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Example directory already exists and is non-empty: {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")

    config_path = Path(args.config).resolve()
    lab_dir = lab_dir_for(config_path)
    add_paths(lab_dir, ROOT, PNFLOW_ROOT)
    from Base.Code_Snippet.metrics_code import reference_metric_batch

    config = read_config(config_path, lab_dir)
    resolve_config_paths(config, lab_dir)

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else Path(config.get("eval", {}).get("checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")

    output_dir = Path(args.out).resolve() if args.out else default_output_dir(config, lab_dir)
    prepare_output(output_dir, args.overwrite)
    print(f"example_dir={output_dir}", flush=True)

    sample_count = args.start_index + args.num_samples
    config.setdefault("eval", {})["test_n"] = sample_count
    config["eval"]["mode"] = "fixed_count_demo"
    config["eval"]["demo_start_index"] = args.start_index
    config["eval"]["demo_num_samples"] = args.num_samples
    write_yaml(output_dir / "config_resolved.yaml", config)

    print("initializing_model_backend", flush=True)
    build_model, causal_forward, ensure_channel, backend = select_backend(args.backend, lab_dir)
    from data.datasets import build_test_dataset

    device = select_device(args.device)
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(int(config.get("seed", 42)))
    print(f"building_model backend={backend} device={device}", flush=True)
    model = build_model(config)
    print(f"loading_checkpoint={checkpoint}", flush=True)
    load_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()

    print("building_fixed_test_dataset", flush=True)
    dataset = build_test_dataset(config)
    sample_rate = int(config["dataset"]["sample_rate"])
    chunk_samples = int(config.get("inference", {}).get("chunk_samples", sample_rate))
    sign_correction = bool(config.get("eval", {}).get("sign_correction", True))
    metric_keys = (
        "si_sdr", "si_sdri", "sdr", "sdri", "si_snr", "si_snri",
        "snr", "snri", "input_si_sdr", "input_sdr", "input_si_snr", "input_snr",
    )
    totals = {key: 0.0 for key in metric_keys}
    rows: list[dict[str, Any]] = []

    for index in range(args.start_index, sample_count):
        print(f"processing_sample={index}", flush=True)
        item = dataset[index]
        batch = batchify(item)
        mixture = ensure_channel(batch["mixture"]).to(device).float()
        target = batch["source"].to(device).float()
        positive = ensure_channel(batch["pos_wave"]).to(device).float()
        negative = ensure_channel(batch["neg_wave"]).to(device).float()
        estimate = causal_forward(model, mixture, positive, negative, chunk_samples)
        mixture_wave = mixture.squeeze(1)
        metric_batch, estimate, flipped = reference_metric_batch(
            estimate,
            target,
            mixture_wave,
            sign_correction=sign_correction,
        )
        sign_flipped = bool(flipped[0].item())
        values = {
            key: float(metric_batch[key][0].cpu())
            for key in metric_keys
        }
        for key, value in values.items():
            totals[key] += value

        sample_dir = output_dir / f"sample_{index:04d}"
        sample_dir.mkdir()
        write_wav(sample_dir / "mixture.wav", mixture_wave[0], sample_rate)
        write_wav(sample_dir / "target.wav", target[0], sample_rate)
        write_wav(sample_dir / "positive_enrollment.wav", positive[0], sample_rate)
        write_wav(sample_dir / "negative_enrollment.wav", negative[0], sample_rate)
        write_wav(sample_dir / "estimate.wav", estimate[0], sample_rate)

        metrics = {
            "sample_index": index,
            "utt_id": str(item.get("utt_id", index)),
            "sample_rate": sample_rate,
            "duration_seconds": round(float(target.shape[-1]) / sample_rate, 6),
            "checkpoint": str(checkpoint),
            "sign_correction": sign_correction,
            "sign_flipped": sign_flipped,
            "metrics": values,
            "audio": {
                "mixture": "mixture.wav",
                "target": "target.wav",
                "positive_enrollment": "positive_enrollment.wav",
                "negative_enrollment": "negative_enrollment.wav",
                "estimate": "estimate.wav",
            },
        }
        write_yaml(sample_dir / "metrics.yaml", metrics)
        rows.append({"sample_index": index, "utt_id": metrics["utt_id"], **values})
        print(f"sample={index} si_sdri={values['si_sdri']:.4f} snri={values['snri']:.4f}")

    count = len(rows)
    summary = {key: totals[key] / count for key in metric_keys}
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "tools/sample_audio_mkr/make_samples.py",
        "lab_dir": str(lab_dir),
        "backend": backend,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_config": str(config_path),
        "device": str(device),
        "dataset": {
            "split": config["dataset"].get("test_root", "LibriSpeech/test-clean"),
            "sample_rate": sample_rate,
            "fixed_indices": list(range(args.start_index, sample_count)),
        },
        "audio_format": {"container": "WAV", "subtype": "FLOAT", "normalization": "none"},
    }
    write_yaml(output_dir / "manifest.yaml", manifest)
    write_yaml(
        output_dir / "summary_metrics.yaml",
        {"num_samples": count, "metrics_mean": summary, "items": rows},
    )
    (output_dir / "README.md").write_text(
        "# TSE Demo Audio\n\n"
        "Each sample directory contains the input mixture, target reference, positive and negative "
        "enrollment waveforms, the TSE estimate, and metrics.yaml. WAV files use IEEE float samples "
        "without amplitude normalization. summary_metrics.yaml averages only these demo items; it is "
        "not the 5,000-item validation result.\n",
        encoding="utf-8",
    )
    print(f"example_dir={output_dir}")
    print(f"summary={output_dir / 'summary_metrics.yaml'}")


if __name__ == "__main__":
    main()
