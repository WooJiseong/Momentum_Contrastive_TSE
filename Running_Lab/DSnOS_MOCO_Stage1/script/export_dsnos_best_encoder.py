#!/usr/bin/env python3
"""Export the true DSnOS Stage0 best student encoder for Stage1."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)

    prefix = "learner.student_encoder."
    exported = {
        key[len(prefix):]: value.detach().cpu()
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not exported:
        raise RuntimeError(f"No student encoder weights found in {source}")
    if not all(
        key.startswith(("encoder.", "encoder_head."))
        for key in exported
    ):
        raise RuntimeError("Exported state contains non-PN encoder keys")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": exported,
            "meta": {
                "format": "PNEncodePath encoder.* + encoder_head.*",
                "source": str(source),
                "source_epoch": int(checkpoint.get("epoch", -1)),
                "source_global_step": int(checkpoint.get("global_step", -1)),
            },
        },
        output,
    )
    print(f"exported={output}")
    print(f"source_epoch={checkpoint.get('epoch')}")
    print(f"source_global_step={checkpoint.get('global_step')}")
    print(f"tensor_count={len(exported)}")


if __name__ == "__main__":
    main()
