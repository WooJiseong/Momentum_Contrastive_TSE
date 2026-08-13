#!/usr/bin/env python
"""Write a fixed-count evaluation summary with an explicit baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


METRICS = ("si_sdr", "si_sdri", "snr", "snri")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Resolved evaluation YAML")
    parser.add_argument("--result-json", required=True, help="Experiment evaluation JSON")
    parser.add_argument("--baseline-json", required=True, help="Baseline evaluation JSON")
    parser.add_argument("--summary", required=True, help="Output result_summary.md")
    parser.add_argument("--baseline-name", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(result: dict, label: str) -> dict[str, float]:
    summary = result.get("summary", {})
    missing = [key for key in METRICS if key not in summary]
    if missing:
        raise SystemExit(f"{label} is missing metrics: {', '.join(missing)}")
    return {key: float(summary[key]) for key in METRICS}


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    result_path = Path(args.result_json).resolve()
    baseline_path = Path(args.baseline_json).resolve()
    summary_path = Path(args.summary).resolve()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = read_json(result_path)
    baseline = read_json(baseline_path)
    current = metric_values(result, "experiment result")
    reference = metric_values(baseline, "baseline result")

    n = int(result.get("n", -1))
    baseline_n = int(baseline.get("n", -1))
    configured_n = int(config.get("eval", {}).get("test_n", n))
    if n != configured_n:
        raise SystemExit(f"experiment count mismatch: configured {configured_n}, got {n}")
    if baseline_n != n:
        raise SystemExit(f"baseline count mismatch: experiment {n}, baseline {baseline_n}")

    experiment = config.get("experiment", {}).get("name", summary_path.parent.name)
    checkpoint = result.get("checkpoint") or config.get("eval", {}).get("checkpoint", "")
    eval_cfg = config.get("eval", {})
    differences = {key: current[key] - reference[key] for key in METRICS}

    lines = [
        "# Result Summary",
        "",
        "## Experiment",
        "",
        str(experiment),
        "",
        "## Main Settings",
        "",
        f"- Checkpoint: {checkpoint}",
        f"- Evaluation mode: {eval_cfg.get('mode', 'fixed_count')}",
        f"- Evaluation items: {n}",
        f"- Configured test_n: {configured_n}",
        "- Full eval status: unavailable for the top-level online mixer; fixed-count deterministic evaluation is used.",
        "",
        "## Best Result",
        "",
        f"- SI-SDR: {fmt(current['si_sdr'])}",
        f"- SI-SDRi: {fmt(current['si_sdri'])}",
        f"- SNR: {fmt(current['snr'])}",
        f"- SNRi: {fmt(current['snri'])}",
        "",
        "## Baseline Comparison",
        "",
        f"- Baseline: `{args.baseline_name}`",
        f"- Baseline source: `{baseline_path}`",
        "- Difference is calculated as `Experiment - Baseline`.",
        "",
        f"| Metric | Baseline | {experiment} | Difference |",
        "|---|---:|---:|---:|",
    ]
    for key, title in (("si_sdr", "SI-SDR"), ("si_sdri", "SI-SDRi"), ("snr", "SNR"), ("snri", "SNRi")):
        lines.append(f"| {title} | {fmt(reference[key])} | {fmt(current[key])} | {fmt(differences[key])} |")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Compared with the same 5000-item fixed-count Baseline, the SI-SDR difference is {fmt(differences['si_sdr'])} dB and the SNR difference is {fmt(differences['snr'])} dB.",
            "",
            "## Issues",
            "",
            "- Full file-manifest evaluation is not available through data.datasets.build_test_dataset.",
            "",
            "## Next Experiment",
            "",
            "- Inspect the Stage1 validation curves and compare checkpoint selection under the same fixed-count protocol.",
            "",
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
