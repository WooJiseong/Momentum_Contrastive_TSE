# =====================================================================================
# [이 파일이 하는 일 — 친절한 한국어 설명]
# "enrollment(목표 화자를 알려주는 참고 음성)"를 어떤 방식으로 모델에 넣을지 고르는 스위치예요.
# 여기서 만든 prefix(앞에 붙이는 조건 토큰)를 flow 모델(UDiT)이 받아서 "이 화자만 뽑아라"라고
# 이해합니다. 아래 세 가지 방식(provider) 중 config의 enroll.provider 로 하나만 고릅니다.
#
#   - V1 (v1_raw)            : 깨끗한 목표 음성 1개의 STFT를 그대로 넣음 (가장 단순, 학습 파라미터 0개).
#   - V2a (v2a_posneg_concat): 화자 인코더 없이 positive(목표)+negative(방해자) STFT를 시간축으로 이어 붙임
#                              (+ pos/neg 구분용 작은 type embedding 학습).
#   - V2b (v2b_pn_encoder)   : 얼려둔(frozen) PN-Enroll 화자 인코더의 임베딩을 작은 adapter로 변환해 넣음.
#
# ★ 기본=v2b 사용 ★ — 이 프로젝트의 표준은 v2b(PN 인코더)예요. 특별한 이유가 없으면 v2b로 두세요.
#
# ✏️ 바꿔도 되는 곳: config의 enroll.provider 값(v1_raw / v2a_posneg_concat / v2b_pn_encoder),
#    그리고 각 provider의 옵션(use_type_emb, mode, use_pos_emb 등).
# ⚠️ 만지지 마세요: PN 인코더 자체는 얼려져 있고(학습 안 함) 여기서 만들지 않습니다. 출력 shape은
#    반드시 [B, 512, T_enroll] 이어야 UDiT가 그대로 받습니다(이 약속 깨지면 모델이 안 돌아감).
# =====================================================================================
"""Enrollment conditioning for PNNoisyFlowTSE — the single V1<->V2 switch point.

Every provider maps a collated ``batch`` dict to a pseudo-STFT enrollment prefix of shape
``[B, 512, T_enroll]``. That is EXACTLY the tensor MeanFlow-TSE's ``UDiT.forward`` expects as
``enrollment``: UDiT runs its shared ``input_proj`` over it, concatenates it as a time-PREFIX to the
noisy-state tokens, attends jointly (one shared self-attention over ``[prefix | state]``), then slices
the prefix off after all blocks. Because UDiT reads ``enrollment.shape[2]`` dynamically, ANY
``T_enroll`` is a drop-in and **UDiT stays byte-identical** across V1 / V2a / V2b.

Providers (selected by ``config['enroll']['provider']``):
  - ``v1_raw``            : MeanFlow-native. enrollment = STFT of ONE clean target utterance (the
                           dataset already computed it as ``batch['enroll_spec']``). ZERO params.
  - ``v2a_posneg_concat`` : NO speaker encoder. Concatenate the positive (target) and negative
                           (distractor) enrollment STFTs along time. MeanFlow marks the prefix only
                           IMPLICITLY-BY-POSITION (no type/segment embedding, verified in
                           udit_meanflow.py), so it cannot tell pos from neg for free — we ADD a
                           learned 2-way type embedding (mirrors our PN head's segment embedding).
  - ``v2b_pn_encoder``    : OUR frozen PN-Enroll encoder. The LightningModule precomputes
                           ``batch['spk_emb'] = pn_encoder.encode(pos, neg) -> [B, 64, Tpn, 65]``;
                           this provider folds 64*65=4160 features and projects them to 512 channels
                           with a learned adapter, yielding a ``[B, 512, M]`` pseudo-STFT prefix.

Only the small adapters/embeddings defined here are trainable. The PN encoder itself is frozen and
held by the LightningModule, so its parameters stay OUT of the optimizer via the
``requires_grad`` filter in ``configure_optimizers``.
"""
from __future__ import annotations

import torch
from torch import nn


class EnrollmentConditioner(nn.Module):
    """Base class. ``forward(batch) -> Tensor[B, 512, T_enroll]`` (a pseudo-STFT enrollment prefix)."""

    out_channels = 512

    def forward(self, batch: dict) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class RawEnrollProvider(EnrollmentConditioner):
    """V1: pass the dataset's single clean-utterance enrollment STFT straight through. No parameters."""

    def forward(self, batch: dict) -> torch.Tensor:
        return batch["enroll_spec"]  # [B, 512, T_enroll]


class PosNegConcatProvider(EnrollmentConditioner):
    """V2a: prefix = [pos_spec | neg_spec] on time, plus a learned 2-way type embedding (pos=0, neg=1).

    The type embedding is a per-segment bias on the 512 channels; it survives UDiT.input_proj just
    like our PN head's ``segment_embedding``. Set ``use_type_emb=False`` to ablate back to pure
    ordering (pos-then-neg) with no marker.
    """

    def __init__(self, use_type_emb: bool = True):
        super().__init__()
        self.use_type_emb = use_type_emb
        self.type_emb = nn.Embedding(2, self.out_channels) if use_type_emb else None

    def forward(self, batch: dict) -> torch.Tensor:
        pos = batch["pos_spec"]  # [B, 512, Tp]
        neg = batch["neg_spec"]  # [B, 512, Tn]
        if self.use_type_emb and self.type_emb is not None:
            pos = pos + self.type_emb.weight[0].view(1, -1, 1)
            neg = neg + self.type_emb.weight[1].view(1, -1, 1)
        return torch.cat([pos, neg], dim=-1)  # [B, 512, Tp+Tn]


class PNEncoderProvider(EnrollmentConditioner):
    """V2b: fold the frozen PN embedding [B,64,Tpn,65] and project to a [B,512,M] pseudo-STFT prefix.

    ``mode='full'`` keeps all Tpn tokens (richest, but inflates the joint self-attention length to
    Tpn+T); ``mode='pool'`` mean-pools to a single token (M=1, cheapest). The 4160->512 ``adapter`` is
    the only trainable projection. A zero-init learned positional embedding is added when
    ``use_pos_emb`` (MeanFlow adds NONE to its prefix; this is an optional ablation knob).
    """

    def __init__(self, in_ch: int = 64, in_freq: int = 65, mode: str = "full",
                 use_pos_emb: bool = True, max_tokens: int = 4096):
        super().__init__()
        assert mode in ("full", "pool"), mode
        self.mode = mode
        self.adapter = nn.Linear(in_ch * in_freq, self.out_channels)
        self.use_pos_emb = use_pos_emb
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, self.out_channels)) if use_pos_emb else None

    def forward(self, batch: dict) -> torch.Tensor:
        e = batch["spk_emb"]  # [B, 64, Tpn, 65] (frozen PN encoder output, precomputed upstream)
        if self.mode == "pool":
            e = e.mean(dim=2, keepdim=True)  # [B, 64, 1, 65]
        b, c, t, f = e.shape
        e = e.permute(0, 2, 1, 3).reshape(b, t, c * f)  # [B, M, 4160]
        x = self.adapter(e)  # [B, M, 512]
        if self.use_pos_emb and self.pos_emb is not None:
            x = x + self.pos_emb[:, :t]
        return x.transpose(1, 2).contiguous()  # [B, 512, M]


def build_enrollment_conditioner(config: dict) -> EnrollmentConditioner:
    """Factory: pick the provider from ``config['enroll']``. Defaults to V1 (RawEnrollProvider)."""
    # [공장(factory) 함수] config의 enroll.provider 로 위 세 방식 중 하나를 골라 만듭니다.
    # 코드상 '값이 아예 없으면' 안전하게 V1로 떨어지지만, 이 프로젝트의 표준(기본)은 v2b 입니다.
    # → config에 enroll.provider: v2b_pn_encoder 를 적어 두고 쓰세요.
    ecfg = (config or {}).get("enroll", {}) or {}
    provider = ecfg.get("provider", "v1_raw")
    if provider == "v1_raw":
        return RawEnrollProvider()
    if provider == "v2a_posneg_concat":
        return PosNegConcatProvider(use_type_emb=ecfg.get("v2a_use_type_emb", True))
    if provider == "v2b_pn_encoder":
        return PNEncoderProvider(mode=ecfg.get("time", "full"),
                                 use_pos_emb=ecfg.get("v2b_use_pos_emb", True))
    raise ValueError(f"Unknown enroll.provider '{provider}'. "
                     f"Use one of: v1_raw, v2a_posneg_concat, v2b_pn_encoder.")
