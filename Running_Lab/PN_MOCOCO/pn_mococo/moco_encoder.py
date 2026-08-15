from __future__ import annotations

import copy
import math
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from pn_mococo.paths import add_repo_paths

add_repo_paths()

from model.GridnetAttnHead import GridNetBlock_attnhead
from model.tfgridnet_encoder import TFGridNet_encoder


class PNEncodePath(nn.Module):
    """Original PN enrollment encode path: Siamese TFGridNet encoder + PN attention head."""

    def __init__(
        self,
        num_blocks: int = 3,
        head_layers: int = 2,
        binaural: bool = False,
    ):
        super().__init__()
        self.encoder = TFGridNet_encoder(
            num_ch=2,
            n_fft=128,
            stride=64,
            num_blocks=num_blocks,
            binaural=binaural,
        )
        self.encoder_head = GridNetBlock_attnhead(
            layer_num=head_layers,
            pooling_size=1,
            stride=1,
        )


def filtered_pn_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return state_dict keys compatible with PNEncodePath."""
    if any(k.startswith("encoder.") or k.startswith("encoder_head.") for k in state):
        return {
            k: v
            for k, v in state.items()
            if k.startswith("encoder.") or k.startswith("encoder_head.")
        }

    prefixes = (
        "student_encoder.",
        "learner.student_encoder.",
        "model.student_encoder.",
    )
    for prefix in prefixes:
        out = {}
        for key, value in state.items():
            if key.startswith(prefix + "encoder.") or key.startswith(prefix + "encoder_head."):
                out[key[len(prefix):]] = value
        if out:
            return out

    raise RuntimeError(
        "Could not find PN encoder weights. Expected keys beginning with "
        "'encoder.'/'encoder_head.' or a student_encoder.* Lightning checkpoint."
    )


def load_trainable_pn_encoder(
    ckpt_path: str,
    num_blocks: int = 3,
    head_layers: int = 2,
    binaural: bool = False,
    strict: bool = True,
) -> PNEncodePath:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model = PNEncodePath(num_blocks=num_blocks, head_layers=head_layers, binaural=binaural)
    missing, unexpected = model.load_state_dict(filtered_pn_state(state), strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"PN encoder load mismatch: missing={missing}, unexpected={unexpected}")
    return model


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def set_tfgridnet_encode_path_requires_grad(module: nn.Module, enabled: bool) -> None:
    """Train only modules used by TFGridNet_encoder.forward()."""
    set_requires_grad(module, False)
    if not enabled:
        return
    for name in ("enc", "conv", "blocks"):
        child = getattr(module, name, None)
        if child is not None:
            set_requires_grad(child, True)


def ensure_channel(wave: torch.Tensor) -> torch.Tensor:
    return wave.unsqueeze(1) if wave.ndim == 2 else wave


def encode_fused_grad(model: PNEncodePath, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Differentiable PNEncodePath.encode(pos, neg)."""
    pos = ensure_channel(pos)
    neg = ensure_channel(neg)
    pos_t = pos.transpose(1, 2)
    neg_t = neg.transpose(1, 2)
    pos_emb = model.encoder(pos_t)
    neg_emb = model.encoder(neg_t)
    cond = model.encoder_head(pos_emb, neg_emb)
    return cond[:, :, :pos_emb.shape[2]]


@torch.no_grad()
def concat_all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x.detach()
    x = x.contiguous()
    gathered = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0).detach()


class ContrastiveProjectionHead(nn.Module):
    """Pool dense PN TF embeddings and project them to a normalized contrastive vector."""

    def __init__(
        self,
        in_ch: int = 64,
        in_freq: int = 65,
        hidden_dim: int = 2048,
        emb_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = False,
    ):
        super().__init__()
        in_dim = in_ch * in_freq
        layers: list[nn.Module] = []
        if nlayers <= 1:
            layers.append(nn.Linear(in_dim, emb_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb.mean(dim=2).flatten(1)
        return F.normalize(self.net(x), dim=-1)


class MomentumContrastivePNLearner(nn.Module):
    """Task-aware momentum contrastive learner for the original TFGridNet PN encode path."""

    def __init__(self, cfg: dict):
        super().__init__()
        ccfg = cfg["contrastive"]
        init_ckpt = ccfg.get("initial_pn_ckpt") or cfg.get("paths", {}).get("initial_pn_ckpt")
        if not init_ckpt:
            raise ValueError("contrastive.initial_pn_ckpt or paths.initial_pn_ckpt is required")

        enc_cfg = cfg.get("encoder", {})
        self.student_encoder = load_trainable_pn_encoder(
            init_ckpt,
            num_blocks=int(enc_cfg.get("num_blocks", 3)),
            head_layers=int(enc_cfg.get("head_layers", 2)),
            binaural=bool(enc_cfg.get("binaural", False)),
            strict=bool(ccfg.get("strict_init", True)),
        )
        self.momentum_encoder = copy.deepcopy(self.student_encoder)

        pcfg = ccfg.get("projection", {})
        self.student_head = ContrastiveProjectionHead(**pcfg)
        self.momentum_head = copy.deepcopy(self.student_head)

        train_encoder = bool(ccfg.get("train_encoder", True))
        train_encoder_head = bool(ccfg.get("train_encoder_head", True))
        set_tfgridnet_encode_path_requires_grad(self.student_encoder.encoder, train_encoder)
        set_requires_grad(self.student_encoder.encoder_head, train_encoder_head)
        set_requires_grad(self.momentum_encoder, False)
        set_requires_grad(self.momentum_head, False)

        self.temperature = float(ccfg.get("temperature", 0.1))
        self.ema_momentum_base = float(ccfg.get("ema_momentum_base", 0.997))
        self.ema_momentum_final = float(ccfg.get("ema_momentum_final", 1.0))
        self.queue_size = int(ccfg.get("queue_size", 4096))
        self.use_inbatch_positive_as_negative = bool(
            ccfg.get("use_inbatch_positive_as_negative", False)
        )
        self.speaker_aware_queue = bool(ccfg.get("speaker_aware_queue", False))

        emb_dim = int(pcfg.get("emb_dim", 256))
        if self.queue_size > 0:
            self.register_buffer("queue", torch.empty(self.queue_size, emb_dim))
            nn.init.normal_(self.queue)
            self.queue = F.normalize(self.queue, dim=-1)
        else:
            self.register_buffer("queue", torch.empty(0, emb_dim))
        # -1 means that the entry predates speaker-aware queue metadata. Such
        # entries remain usable, but cannot be safely masked by speaker ID.
        self.register_buffer(
            "queue_speaker_ids",
            torch.full((max(0, self.queue_size),), -1, dtype=torch.long),
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from (p for p in self.student_encoder.parameters() if p.requires_grad)
        yield from self.student_head.parameters()

    def ema_momentum(self, step: int, total_steps: int) -> float:
        if self.ema_momentum_final <= self.ema_momentum_base:
            return self.ema_momentum_base
        progress = min(1.0, max(0.0, float(step) / float(max(1, total_steps))))
        return self.ema_momentum_final - (
            self.ema_momentum_final - self.ema_momentum_base
        ) * (math.cos(math.pi * progress) + 1.0) / 2.0

    @torch.no_grad()
    def update_momentum_encoder(self, momentum: float) -> None:
        for ps, pk in zip(self.student_encoder.parameters(), self.momentum_encoder.parameters()):
            pk.data.mul_(momentum).add_(ps.detach().data, alpha=1.0 - momentum)
        for ps, pk in zip(self.student_head.parameters(), self.momentum_head.parameters()):
            pk.data.mul_(momentum).add_(ps.detach().data, alpha=1.0 - momentum)
        for bs, bk in zip(self.student_encoder.buffers(), self.momentum_encoder.buffers()):
            bk.data.copy_(bs.detach().data)
        for bs, bk in zip(self.student_head.buffers(), self.momentum_head.buffers()):
            bk.data.copy_(bs.detach().data)

    def student_embedding(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        return self.student_head(encode_fused_grad(self.student_encoder, pos, neg))

    @torch.no_grad()
    def momentum_embedding(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        self.momentum_encoder.eval()
        self.momentum_head.eval()
        emb = encode_fused_grad(self.momentum_encoder, pos, neg)
        return self.momentum_head(emb)

    def contrastive_loss(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: list[torch.Tensor],
        use_queue: bool,
        update_queue: bool,
        query_speaker_ids: torch.Tensor | None = None,
        negative_speaker_ids: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not negative_keys:
            raise ValueError("At least one explicit negative key is required.")
        if self.speaker_aware_queue:
            if query_speaker_ids is None:
                raise ValueError("speaker_aware_queue requires query_speaker_ids.")
            if negative_speaker_ids is None or len(negative_speaker_ids) != len(negative_keys):
                raise ValueError(
                    "speaker_aware_queue requires one negative_speaker_ids tensor per negative key."
                )

        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key.detach(), dim=-1)
        explicit_neg = torch.cat(
            [concat_all_gather_no_grad(F.normalize(k, dim=-1)) for k in negative_keys],
            dim=0,
        )
        explicit_neg_speaker_ids = None
        if self.speaker_aware_queue:
            explicit_neg_speaker_ids = torch.cat(
                [
                    concat_all_gather_no_grad(ids.to(query.device, dtype=torch.long).reshape(-1))
                    for ids in negative_speaker_ids
                ],
                dim=0,
            )

        neg_logit_parts = [query @ explicit_neg.t()]
        if self.use_inbatch_positive_as_negative:
            all_pos = concat_all_gather_no_grad(positive_key)
            all_pos_speaker_ids = None
            if self.speaker_aware_queue:
                all_pos_speaker_ids = concat_all_gather_no_grad(
                    query_speaker_ids.to(query.device, dtype=torch.long).reshape(-1)
                )
            if all_pos.shape[0] > query.shape[0]:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                start = rank * query.shape[0]
                mask = torch.ones(all_pos.shape[0], dtype=torch.bool, device=all_pos.device)
                mask[start:start + query.shape[0]] = False
                all_pos = all_pos[mask]
                if all_pos_speaker_ids is not None:
                    all_pos_speaker_ids = all_pos_speaker_ids[mask]
            else:
                all_pos = all_pos[:0]
                if all_pos_speaker_ids is not None:
                    all_pos_speaker_ids = all_pos_speaker_ids[:0]
            if all_pos.numel() > 0:
                all_pos_logits = query @ all_pos.t()
                if all_pos_speaker_ids is not None:
                    same_speaker = query_speaker_ids.to(query.device, dtype=torch.long).reshape(-1, 1) == (
                        all_pos_speaker_ids.to(query.device, dtype=torch.long).reshape(1, -1)
                    )
                    all_pos_logits = all_pos_logits.masked_fill(same_speaker, float("-inf"))
                neg_logit_parts.append(all_pos_logits)

        queue_len = int(self.queue_filled.item())
        queue_masked_ratio = query.new_tensor(0.0)
        if use_queue and queue_len > 0:
            queue_logits = query @ self.queue[:queue_len].detach().t()
            if self.speaker_aware_queue:
                query_speaker_ids = query_speaker_ids.to(query.device, dtype=torch.long).reshape(-1)
                queue_ids = self.queue_speaker_ids[:queue_len].to(query.device, dtype=torch.long)
                known_queue_ids = queue_ids.ge(0).unsqueeze(0)
                same_speaker = query_speaker_ids.unsqueeze(1).eq(queue_ids.unsqueeze(0))
                queue_mask = same_speaker & known_queue_ids
                queue_logits = queue_logits.masked_fill(queue_mask, float("-inf"))
                queue_masked_ratio = queue_mask.float().mean()
            neg_logit_parts.append(queue_logits)

        pos_logits = torch.sum(query * positive_key, dim=-1, keepdim=True)
        neg_logits = torch.cat(neg_logit_parts, dim=1)
        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            finite_neg = neg_logits[torch.isfinite(neg_logits)]
            neg_sim = finite_neg.mean() if finite_neg.numel() else query.new_tensor(0.0)
            finite_neg_by_row = torch.where(
                torch.isfinite(neg_logits), neg_logits, neg_logits.new_tensor(-1.0e4)
            )
            metrics = {
                "loss": loss.detach(),
                "acc": (pred == 0).float().mean(),
                "pos_sim": pos_logits.mean(),
                "neg_sim": neg_sim,
                "neg_sim_max": finite_neg_by_row.max(dim=1).values.mean(),
                "queue_len": torch.tensor(float(queue_len), device=query.device),
                "queue_masked_ratio": queue_masked_ratio,
            }
            if update_queue and self.queue_size > 0:
                self._dequeue_and_enqueue(explicit_neg, explicit_neg_speaker_ids)
        return loss, metrics

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        keys: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
    ) -> None:
        if self.queue_size <= 0 or keys.numel() == 0:
            return
        if self.speaker_aware_queue:
            if speaker_ids is None or speaker_ids.shape[0] != keys.shape[0]:
                raise ValueError("speaker_aware_queue requires one speaker ID per queued key.")
            speaker_ids = speaker_ids.to(keys.device, dtype=torch.long).reshape(-1)
        keys = F.normalize(keys.detach(), dim=-1)
        if keys.shape[0] >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size:])
            if speaker_ids is not None:
                self.queue_speaker_ids.copy_(speaker_ids[-self.queue_size:])
            self.queue_ptr.zero_()
            self.queue_filled.fill_(self.queue_size)
            return

        ptr = int(self.queue_ptr.item())
        n = keys.shape[0]
        end = ptr + n
        if end <= self.queue_size:
            self.queue[ptr:end].copy_(keys)
            if speaker_ids is not None:
                self.queue_speaker_ids[ptr:end].copy_(speaker_ids)
        else:
            first = self.queue_size - ptr
            self.queue[ptr:].copy_(keys[:first])
            self.queue[:end - self.queue_size].copy_(keys[first:])
            if speaker_ids is not None:
                self.queue_speaker_ids[ptr:].copy_(speaker_ids[:first])
                self.queue_speaker_ids[:end - self.queue_size].copy_(speaker_ids[first:])
        self.queue_ptr[0] = end % self.queue_size
        self.queue_filled[0] = min(self.queue_size, int(self.queue_filled.item()) + n)
