"""Improved USEF PN enrollment path used by the final experiment.

The Hugging Face improved checkpoint is a complete USEF-TFGridNet separator.
For Stage0 and Flow enrollment we keep only its ``siamese`` and
``encoder_head`` branches.  The output contract remains [B, 64, T, 65], so the
existing Attn_MOCOCO projection and SIA Flow Generator can consume it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from model.tfgridnet_encoder import TFGridNet_encoder
from improved_model.GridnetAttnHead import GridNetBlock_attnhead


class ImprovedPNEncodePath(nn.Module):
    """USEF improved siamese encoder plus its positive/negative fusion head."""

    def __init__(self, binaural: bool = False):
        super().__init__()
        self.encoder = TFGridNet_encoder(
            num_ch=2,
            n_fft=128,
            stride=64,
            num_blocks=1,
            binaural=binaural,
        )
        self.encoder_head = GridNetBlock_attnhead(
            layer_num=2,
            pooling_size=1,
            stride=1,
            return_clean_dvec=False,
            out_dim=0,
            refine_layer_num=2,
            fusion_shortcut=[0, 1],
            cut_pos=True,
        )


def _state_dict(ckpt_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def filtered_improved_pn_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map either a full improved checkpoint or an exported Stage0 checkpoint."""
    if any(key.startswith("encoder.") for key in state):
        encoder = {
            key: value for key, value in state.items()
            if key.startswith("encoder.") or key.startswith("encoder_head.")
        }
        if encoder:
            return encoder

    if any(key.startswith("siamese.") for key in state):
        return {
            ("encoder." + key[len("siamese."):]) if key.startswith("siamese.")
            else key: value
            for key, value in state.items()
            if key.startswith("siamese.") or key.startswith("encoder_head.")
        }

    prefixes = (
        "student_encoder.",
        "learner.student_encoder.",
        "model.student_encoder.",
    )
    for prefix in prefixes:
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix + "encoder.")
            or key.startswith(prefix + "encoder_head.")
        }
        if selected:
            return selected

    raise RuntimeError(
        "Improved checkpoint has no encoder/encoder_head or siamese/encoder_head keys."
    )


def load_trainable_pn_encoder(
    ckpt_path: str,
    num_blocks: int = 1,
    head_layers: int = 2,
    binaural: bool = False,
    strict: bool = True,
) -> ImprovedPNEncodePath:
    """Load the improved PN path for the Attn_MOCOCO Stage0 learner."""
    if int(num_blocks) != 1 or int(head_layers) != 2:
        raise ValueError(
            "Improved USEF PN path requires encoder.num_blocks=1 and head_layers=2."
        )
    model = ImprovedPNEncodePath(binaural=binaural)
    missing, unexpected = model.load_state_dict(
        filtered_improved_pn_state(_state_dict(ckpt_path)),
        strict=strict,
    )
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Improved PN encoder load mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model

