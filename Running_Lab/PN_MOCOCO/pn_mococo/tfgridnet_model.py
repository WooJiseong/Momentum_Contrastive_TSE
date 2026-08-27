from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from pn_mococo.moco_encoder import ensure_channel, filtered_pn_state
from pn_mococo.paths import add_repo_paths

add_repo_paths()

from Base.Code_Snippet.metrics_code import (
    fast_training_metrics,
    negative_si_sdr_loss,
    reference_metrics,
)

from model.GridnetAttnHead import GridNetBlock_attnhead
from model.tfgridnet_KVfusion import TFGridNet_KVfusion
from model.tfgridnet_crossattn_causal_single_emb import TFGridNet_origcrossattn_causal_single_emb
from model.tfgridnet_encoder import TFGridNet_encoder


def build_causal_tfgridnet(config: dict) -> TFGridNet_origcrossattn_causal_single_emb:
    mcfg = config.get("model", {})
    ecfg = config.get("encoder", {})
    encoder = TFGridNet_encoder(
        num_ch=2,
        n_fft=int(ecfg.get("n_fft", 128)),
        stride=int(ecfg.get("stride", 64)),
        num_blocks=int(ecfg.get("num_blocks", 3)),
        binaural=bool(ecfg.get("binaural", False)),
    )
    encoder_head = GridNetBlock_attnhead(
        layer_num=int(ecfg.get("head_layers", 2)),
        pooling_size=1,
        stride=1,
    )
    return TFGridNet_origcrossattn_causal_single_emb(
        n_fft=int(mcfg.get("n_fft", 128)),
        stride=int(mcfg.get("stride", 64)),
        n_layers=int(mcfg.get("n_layers", 3)),
        lstm_hidden_units=int(mcfg.get("lstm_hidden_units", 64)),
        emb_dim=int(mcfg.get("emb_dim", 64)),
        emb_ks=int(mcfg.get("emb_ks", 1)),
        model_normalize=bool(mcfg.get("model_normalize", True)),
        Fusion_class=TFGridNet_KVfusion,
        pooling_size=int(mcfg.get("pooling_size", 40)),
        fusion_stride=int(mcfg.get("fusion_stride", 40)),
        encoder=encoder,
        encoder_head=encoder_head,
        train_encoder=bool(config.get("trainable", {}).get("encoder", False)),
        train_encoder_head=bool(config.get("trainable", {}).get("encoder_head", False)),
        fusion_layer=list(mcfg.get("fusion_layer", [0, 1])),
        binaural=bool(mcfg.get("binaural", False)),
    )


def _load_state(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def load_full_separator(
    model: nn.Module,
    ckpt_path: str | None,
    strict: bool = True,
) -> None:
    if not ckpt_path:
        return
    state = _load_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Full separator checkpoint mismatch for {ckpt_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_encoder_override(
    model: nn.Module,
    ckpt_path: str | None,
    strict: bool = True,
) -> None:
    if not ckpt_path:
        return
    state = filtered_pn_state(_load_state(ckpt_path))
    own_state = model.state_dict()
    own_state.update(state)
    missing, unexpected = model.load_state_dict(own_state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Encoder override checkpoint mismatch for {ckpt_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def set_module_trainable(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def configure_trainability(model: TFGridNet_origcrossattn_causal_single_emb, config: dict) -> None:
    trainable = config.get("trainable", {})
    enc_enabled = bool(trainable.get("encoder", False))
    head_enabled = bool(trainable.get("encoder_head", False))
    main_enabled = bool(trainable.get("separator", True))
    model.train_encoder = enc_enabled
    model.train_encoder_head = head_enabled
    set_module_trainable(model.encoder, enc_enabled)
    set_module_trainable(model.encoder_head, head_enabled)
    for module in (model.enc, model.dec, model.conv, model.blocks, model.fusions, model.deconv):
        set_module_trainable(module, main_enabled)


def build_model_from_config(config: dict) -> TFGridNet_origcrossattn_causal_single_emb:
    model_type = config.get("model", {}).get("type", "tfgridnet_causal")
    if model_type != "tfgridnet_causal":
        raise ValueError("Only model.type=tfgridnet_causal is currently wired for PN_MOCOCO.")
    model = build_causal_tfgridnet(config)
    paths = config.get("paths", {})
    load_full_separator(
        model,
        paths.get("initial_model_ckpt"),
        strict=bool(paths.get("strict_initial_model", True)),
    )
    load_encoder_override(
        model,
        paths.get("encoder_override_ckpt"),
        strict=bool(paths.get("strict_encoder_override", True)),
    )
    configure_trainability(model, config)
    return model


def causal_forward(
    model: TFGridNet_origcrossattn_causal_single_emb,
    mixture: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    chunk_samples: int,
) -> torch.Tensor:
    """Run original causal TFGridNet over fixed chunks and return [B, T]."""
    mixture = ensure_channel(mixture)
    pos = ensure_channel(pos)
    neg = ensure_channel(neg)
    cond_emb = model.encode(pos, neg)
    init_state = model.init_buffers(mixture.shape[0], mixture.device)
    outs = []
    for chunk in torch.split(mixture, int(chunk_samples), dim=-1):
        if chunk.shape[-1] == 0:
            continue
        out_chunk, init_state = model(chunk, cond_emb, init_state)
        outs.append(out_chunk)
    if not outs:
        raise RuntimeError("No chunks were produced for causal TFGridNet forward.")
    est = torch.cat(outs, dim=-1)
    est = est.squeeze(1) if est.ndim == 3 and est.shape[1] == 1 else est
    return est[..., :mixture.shape[-1]]


def neg_si_sdr_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return negative_si_sdr_loss(est, target)


@torch.no_grad()
def separation_metrics(
    est: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    *,
    reference: bool = True,
) -> dict[str, torch.Tensor]:
    if reference:
        return reference_metrics(est, target, mixture)
    return fast_training_metrics(est, target, mixture)


def checkpoint_exists(path: str | None) -> bool:
    return bool(path) and Path(path).is_file()
