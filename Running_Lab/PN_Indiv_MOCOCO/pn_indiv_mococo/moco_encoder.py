from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F

from pn_indiv_mococo.paths import add_repo_paths

add_repo_paths()

from model.GridnetAttnHead import GridNetBlock_attnhead
from model.tfgridnet_encoder import TFGridNet_encoder


class PNEncodePath(nn.Module):
    def __init__(self, num_blocks: int = 3, head_layers: int = 2, binaural: bool = False):
        super().__init__()
        self.encoder = TFGridNet_encoder(2, 128, 64, num_blocks, binaural)
        self.encoder_head = GridNetBlock_attnhead(head_layers, 1, 1)


def ensure_channel(wave: torch.Tensor) -> torch.Tensor:
    return wave.unsqueeze(1) if wave.ndim == 2 else wave


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def set_tfgridnet_encode_path_requires_grad(module: nn.Module, enabled: bool) -> None:
    set_requires_grad(module, False)
    if not enabled:
        return
    for name in ("enc", "conv", "blocks"):
        child = getattr(module, name, None)
        if child is not None:
            set_requires_grad(child, True)


def filtered_pn_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = state.get("state_dict", state)
    for prefix in ("", "student_encoder.", "learner.student_encoder.", "model.student_encoder."):
        out = {}
        for key, value in state.items():
            if key.startswith(prefix + "encoder.") or key.startswith(prefix + "encoder_head."):
                out[key[len(prefix):]] = value
        if out:
            return out
    raise RuntimeError("No encoder.* / encoder_head.* weights found in checkpoint")


def encode_fused(model: PNEncodePath, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    pos = ensure_channel(pos)
    neg = ensure_channel(neg)
    pos_emb = model.encoder(pos.transpose(1, 2))
    neg_emb = model.encoder(neg.transpose(1, 2))
    return model.encoder_head(pos_emb, neg_emb)[:, :, :pos_emb.shape[2]]


class ProjectionHead(nn.Module):
    def __init__(self, in_ch=64, in_freq=65, hidden_dim=2048, emb_dim=256, nlayers=3, use_bn=False):
        super().__init__()
        if nlayers <= 1:
            layers = [nn.Linear(in_ch * in_freq, emb_dim)]
        else:
            layers = [nn.Linear(in_ch * in_freq, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(max(0, nlayers - 2)):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.nan_to_num(x.float())
        return F.normalize(self.net(x.mean(dim=2).flatten(1)), dim=-1)


class IndividualNegativeMoCo(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ccfg = cfg["contrastive"]
        ecfg = cfg.get("encoder", {})
        ckpt = ccfg["initial_pn_ckpt"]
        self.student = PNEncodePath(int(ecfg.get("num_blocks", 3)), int(ecfg.get("head_layers", 2)), bool(ecfg.get("binaural", False)))
        self.student.load_state_dict(filtered_pn_state(torch.load(ckpt, map_location="cpu")), strict=True)
        self.momentum = copy.deepcopy(self.student)
        pcfg = ccfg.get("projection", {})
        self.embedding_dim = int(pcfg.get("emb_dim", 256))
        self.projection = ProjectionHead(**pcfg)
        self.momentum_projection = copy.deepcopy(self.projection)
        set_tfgridnet_encode_path_requires_grad(self.student.encoder, bool(ccfg.get("train_encoder", True)))
        set_requires_grad(self.student.encoder_head, bool(ccfg.get("train_encoder_head", True)))
        for p in self.momentum.parameters():
            p.requires_grad = False
        for p in self.momentum_projection.parameters():
            p.requires_grad = False
        self.temperature = float(ccfg.get("temperature", 0.1))
        self.ema_momentum = float(ccfg.get("ema_momentum", 0.997))
        self.negative_aggregation = str(ccfg.get("negative_aggregation", "mean")).lower()
        if self.negative_aggregation not in {"mean", "sum"}:
            raise ValueError(
                "contrastive.negative_aggregation must be 'mean' or 'sum', "
                f"got {self.negative_aggregation!r}"
            )

    def student_embedding(self, pos, neg):
        return self.projection(encode_fused(self.student, pos, neg))

    @torch.no_grad()
    def momentum_embedding(self, pos, neg):
        self.momentum.eval()
        self.momentum_projection.eval()
        return self.momentum_projection(encode_fused(self.momentum, pos, neg))

    def student_negative_embeddings(self, pos, neg_items):
        bsz, nneg = neg_items.shape[:2]
        pos_rep = pos[:, None].expand(-1, nneg, -1, -1).reshape(bsz * nneg, *pos.shape[1:])
        neg_rep = neg_items.reshape(bsz * nneg, *neg_items.shape[2:])
        return self.projection(encode_fused(self.student, pos_rep, neg_rep)).view(bsz, nneg, -1)

    @torch.no_grad()
    def momentum_negative_embeddings(self, pos, neg_items, valid_mask=None):
        bsz, nneg = neg_items.shape[:2]
        pos_rep = pos[:, None].expand(-1, nneg, -1, -1).reshape(bsz * nneg, *pos.shape[1:])
        neg_rep = neg_items.reshape(bsz * nneg, *neg_items.shape[2:])
        if valid_mask is None:
            return self.momentum_projection(encode_fused(self.momentum, pos_rep, neg_rep)).view(bsz, nneg, -1)

        flat_valid = valid_mask.reshape(-1).bool()
        out = neg_items.new_zeros((bsz * nneg, self.embedding_dim))
        if flat_valid.any():
            emb = self.momentum_projection(
                encode_fused(self.momentum, pos_rep[flat_valid], neg_rep[flat_valid])
            )
            out[flat_valid] = emb
        return out.view(bsz, nneg, -1)

    def loss(self, query, positive, negative, valid_mask):
        query_raw = query.float()
        positive_raw = positive.detach().float()
        negative_raw = negative.detach().float()
        finite_query = torch.isfinite(query_raw).all(dim=-1)
        finite_positive = torch.isfinite(positive_raw).all(dim=-1)
        finite_negative = torch.isfinite(negative_raw).all(dim=-1)
        query = F.normalize(torch.nan_to_num(query_raw), dim=-1)
        positive = F.normalize(torch.nan_to_num(positive_raw), dim=-1)
        negative = F.normalize(torch.nan_to_num(negative_raw), dim=-1)
        valid_mask = valid_mask.bool() & finite_negative
        pos_logit = (query * positive).sum(-1)
        neg_logits = torch.einsum("bd,bnd->bn", query, negative)
        pos_logit = torch.nan_to_num(pos_logit, nan=0.0, posinf=1.0, neginf=-1.0)
        neg_logits = torch.nan_to_num(neg_logits, nan=-1.0e4, posinf=1.0, neginf=-1.0)
        n = valid_mask.sum(dim=1).clamp_min(1).to(neg_logits.dtype)
        weighted = neg_logits.masked_fill(~valid_mask, -1.0e4)
        if self.negative_aggregation == "mean":
            weighted = weighted - n.log().unsqueeze(1)
        neg_logit = torch.logsumexp(weighted, dim=1)
        logits = torch.stack([pos_logit, neg_logit], dim=1) / self.temperature
        logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            valid_neg_values = neg_logits[valid_mask]
            neg_sim = valid_neg_values.mean() if valid_neg_values.numel() else neg_logits.new_tensor(0.0)
            return loss, {
                "acc": (logits.argmax(1) == 0).float().mean(),
                "pos_sim": pos_logit.mean(),
                "neg_sim": neg_sim,
                "neg_logmean_sim" if self.negative_aggregation == "mean" else "neg_logsum_sim": neg_logit.mean(),
                "n_negative": n.mean(),
                "finite_query_ratio": finite_query.float().mean(),
                "finite_positive_ratio": finite_positive.float().mean(),
                "valid_negative_ratio": valid_mask.float().mean(),
            }

    @torch.no_grad()
    def update_momentum(self):
        m = self.ema_momentum
        for ps, pm in zip(self.student.parameters(), self.momentum.parameters()):
            pm.data.mul_(m).add_(ps.data, alpha=1.0 - m)
        for ps, pm in zip(self.projection.parameters(), self.momentum_projection.parameters()):
            pm.data.mul_(m).add_(ps.data, alpha=1.0 - m)
