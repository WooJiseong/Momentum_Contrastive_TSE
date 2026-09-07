# ============================================================================
# 이 파일이 하는 일 (pn/encoder.py)
# ----------------------------------------------------------------------------
# "고정(frozen) PN 인코더 = 학습 안 하는 화자 인코더"의 실제 신경망 구조(껍데기가 아니라
# 알맹이)가 들어 있는 파일입니다. 위층 pn_encoder.py 가 이 안의 PNEncodePath 를 만들고
# proposed-monaural.pt 에서 'encoder.*' 와 'encoder_head.*' 가중치만 채워 넣습니다.
#
# 구성(세 덩어리):
#   1) TFGridNet_encoder : 음성 파형 -> 시간-주파수 임베딩 [B, 64, T, 65] (샴 구조의 한 가지)
#      - 샴(Siamese): 같은 인코더로 pos(목표 예시)와 neg(비목표 예시)를 따로 통과시킴.
#   2) GridNetBlock_attn / GridNetBlock_attnhead : pos·neg 임베딩을 어텐션으로 융합(fusion).
#   3) PNEncodePath : 위 둘을 'encoder', 'encoder_head' 라는 정확한 이름으로 묶은 껍데기.
#      encode(pos, neg) -> 화자 임베딩 [B, 64, T, 65] 를 내줍니다.
#
# 의존성 참고: 이 인코더는 ESPnet(espnet2)의 TFGridNet 위에 세워져 있습니다 -> espnet 설치 필수.
#
# ⚠️ 만지지 마세요(이 파일 전체):
#   - 서브모듈 이름('encoder', 'encoder_head'), 하이퍼파라미터(n_fft=128, emb_dim=64,
#     n_freqs=65, num_blocks 등), forward 안의 모든 텐서 연산.
#   - 이름·숫자 하나만 바뀌어도 proposed-monaural.pt 의 가중치가 strict=True 로 안 맞아
#     로드가 실패합니다. 이 파일은 "사전학습된 인코더와 모양을 정확히 맞추는" 코드라서
#     구조를 바꾸면 안 됩니다. (검증된 코드: 273 + 131 = 404개 텐서가 정확히 일치)
# ✏️ 바꿔도 되는 곳: 없음(주석만).
# ============================================================================

# Minimal PN-Enroll encode path: TFGridNet Siamese encoder + attn fusion head + encode().
"""Frozen PN-Enroll speaker encoder (encode-path only) for NoisyFlowTSE.

COPY-ADAPT of sia_fm_tse/pnenroll/model/{tfgridnet_encoder.py, GridnetAttnHead.py} and the
encode() method of tfgridnet_crossattn_causal_single_emb.py. Only the encode() path is needed:
the proposed-monaural checkpoint's encode() consumes ONLY `encoder.*` (TFGridNet_encoder) and
`encoder_head.*` (GridNetBlock_attnhead) — verified: 273 + 131 = 404 state-dict tensors, the
separator's enc/dec/conv/blocks/fusions/deconv are NOT used by encode().

DEPENDENCIES (pre-installed env packages the FROZEN encoder fundamentally requires; these are
NOT imports from reference/ or sia_fm_tse/ — the checkpoint's TFGridNet is built on ESPnet):
  espnet2.enh.separator.tfgridnet_separator.TFGridNet
  espnet2.torch_utils.get_layer_from_string.get_layer

The wrapper PNEncodePath registers the two submodules under the EXACT names `encoder` and
`encoder_head`, so a filtered checkpoint state dict (keys 'encoder.*' + 'encoder_head.*') loads
strict=True.

encode() tensor contract (verified against the live model):
  encode(pos: float[B, 1, NW], neg: float[B, 1, NW]) -> cond_emb float[B, 64, T, 65]
  (emb_dim = 64, n_freqs = n_fft//2+1 = 65 with n_fft=128; T tracks the pos length.)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from espnet2.enh.separator.tfgridnet_separator import TFGridNet as TFGridNet_original
from espnet2.torch_utils.get_layer_from_string import get_layer


# ----------------------------------------------------------------------------
# Siamese encoder branch (copy of pnenroll/model/tfgridnet_encoder.py).
# Subclasses ESPnet TFGridNet; reuses its self.enc / self.conv / self.blocks, drops the
# decoder/pooling. Strips the waveform reconstruction.
# ----------------------------------------------------------------------------
class TFGridNet_encoder(TFGridNet_original):
    """One branch of the Siamese encoder: waveform -> TF embedding [B, 64, T, 65]."""

    def __init__(self, num_ch: int, n_fft: int, stride: int, num_blocks: int, binaural: bool):
        super().__init__(
            input_dim=None, n_fft=n_fft, stride=stride, n_imics=num_ch, n_srcs=1,
            lstm_hidden_units=64, n_layers=num_blocks, emb_dim=64,
        )
        self.binaural = binaural

    def forward(self, input: torch.Tensor, ilens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """input: float[B, NW, M] (mono duplicated to binaural internally) -> float[B, 64, T, 65]."""
        if ilens is None:
            ilens = torch.tensor([_.shape[0] for _ in input])

        # 모노(1채널) 입력을 같은 신호 2개로 복제해서 가짜 양이(binaural, 2채널)로 만듦.
        # (사전학습 모델이 2채널 입력을 기대하기 때문. 모양 맞추기용.)
        if not self.binaural:
            input = torch.concat([input, input], dim=2)  # duplicate mono to fake binaural

        # RMS 정규화: 입력 소리 크기를 표준편차로 나눠 음량 차이를 없앰(크게/작게 녹음돼도 동일).
        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization

        batch = self.enc(input, ilens)[0]  # [B, T, M, F]
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]

        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        return batch


# ----------------------------------------------------------------------------
# Fusion head over [pos ; neg] (copy of pnenroll/model/GridnetAttnHead.py).
# ----------------------------------------------------------------------------
class LayerNormalization4DCF(nn.Module):
    """Channel-frequency LayerNorm over [B, C, T, F] used inside the attn head."""

    def __init__(self, input_dimension, eps: float = 1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, F] -> normalized [B, C, T, F]."""
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,1]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class GridNetBlock_attn(nn.Module):
    """Multi-head TF self-attention block (Q/K/V conv heads + concat proj)."""

    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, emb_dim: int, emb_ks: int, emb_hs: int, n_freqs: int,
                 n_head: int = 4, approx_qk_dim: int = 512, activation: str = "prelu",
                 eps: float = 1e-5):
        super().__init__()
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module("attn_conv_Q_%d" % ii, nn.Sequential(
                nn.Conv2d(emb_dim, E, 1), get_layer(activation)(),
                LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module("attn_conv_K_%d" % ii, nn.Sequential(
                nn.Conv2d(emb_dim, E, 1), get_layer(activation)(),
                LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module("attn_conv_V_%d" % ii, nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim // n_head, 1), get_layer(activation)(),
                LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps)))
        self.add_module("attn_concat_proj", nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1), get_layer(activation)(),
            LayerNormalization4DCF((emb_dim, n_freqs), eps=eps)))
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, Q] -> [B, C, T, Q] (multi-head TF self-attention)."""
        B, C, old_T, old_Q = x.shape
        batch = x

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim ** 0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q]
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q]
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q]
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q]

        return batch


class GridNetBlock_attnhead(nn.Module):
    """Fuse pos & neg TF embeddings (concat over time + segment embedding) -> [B, 64, T, 65]."""

    def __init__(self, layer_num: int, pooling_size: int, stride: int,
                 return_clean_dvec: bool = False, out_dim: int = 0):
        super().__init__()
        self.pooling_size = pooling_size
        self.stride = stride
        self.segment_embedding = nn.Embedding(2, 64 * 65)
        self.model = nn.ModuleList(
            [GridNetBlock_attn(emb_dim=64, emb_ks=1, emb_hs=1, n_freqs=65, n_head=4, eps=1.0e-5)
             for _ in range(layer_num)]
        )
        # return_clean_dvec / out_dim heads are unused by encode(); kept off (no embed_proj) so the
        # checkpoint keys match exactly (proposed-monaural has no embed_proj).

    def forward(self, pos_cond: torch.Tensor, neg_cond: torch.Tensor) -> torch.Tensor:
        """pos_cond, neg_cond: float[B, 64, T, 65] -> fused float[B, 64, T_pos+T_neg, 65]."""
        B, C, T_pos, Fr = pos_cond.shape
        B, C, T_neg, Fr = neg_cond.shape

        # pos(목표 예시)와 neg(비목표 예시) 임베딩을 시간 축으로 이어 붙임.
        x = torch.concat([pos_cond, neg_cond], dim=2)  # [B, C, 2T', F]

        # 세그먼트 인덱스: 앞쪽(pos)은 0, 뒤쪽(neg)은 1 로 표시 -> 모델이 둘을 구분하게 함.
        seg_idx = torch.concat(
            [torch.zeros((B, T_pos), device=pos_cond.device),
             torch.ones((B, T_neg), device=pos_cond.device)], dim=1
        )
        # 0/1 표시를 학습된 임베딩으로 바꿔(pos/neg 구분 신호) 본 신호에 더해 줌.
        seg_emb = self.segment_embedding(seg_idx.to(torch.int32))  # [B, 2T', C*F]
        seg_emb = seg_emb.unflatten(dim=2, sizes=(C, Fr)).permute((0, 2, 1, 3))  # [B, C, 2T', F]

        x = x + seg_emb

        for layer in self.model[:-1]:
            x = x + layer(x)
        x = self.model[-1](x)
        return x


# ----------------------------------------------------------------------------
# Slim wrapper holding ONLY the encode() submodules (encoder + encoder_head).
# ----------------------------------------------------------------------------
class PNEncodePath(nn.Module):
    """Frozen PN encode path: encode(pos, neg) -> speaker embedding [B, 64, T, 65].

    Holds `self.encoder` (TFGridNet_encoder) and `self.encoder_head` (GridNetBlock_attnhead)
    under those EXACT attribute names so a filtered proposed-monaural state dict (keys
    'encoder.*' + 'encoder_head.*') loads strict=True. Replicates the encode() method of
    pnenroll's TFGridNet_origcrossattn_causal_single_emb (train_encoder/train_encoder_head=False
    -> no_grad + detach inside).
    """

    def __init__(self):
        super().__init__()
        # 'encoder'/'encoder_head' 라는 이름이 체크포인트 키와 정확히 일치해야 로드됨.(이름·숫자 고정)
        self.encoder = TFGridNet_encoder(num_ch=2, n_fft=128, stride=64, num_blocks=3, binaural=False)
        self.encoder_head = GridNetBlock_attnhead(layer_num=2, pooling_size=1, stride=1)

    # @torch.no_grad(): encode 안에서는 절대 미분을 계산하지 않음(고정 인코더라 학습 안 함).
    @torch.no_grad()
    def encode(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """pos, neg: float[B, 1, NW] -> cond_emb float[B, 64, T, 65].

        Mirrors pnenroll encode(): transpose to [B, NW, 1], run each branch through the
        frozen encoder, fuse with encoder_head, then slice cond_emb to the pos time length.
        (Original ran with train_encoder/train_encoder_head=False -> no_grad + detach.)
        """
        # 축 순서를 인코더가 기대하는 모양으로 바꿈.
        pos = pos.transpose(1, 2)  # [B, NW, 1]
        neg = neg.transpose(1, 2)  # [B, NW, 1]

        # 같은 인코더로 pos/neg 각각 임베딩 추출(샴 구조). detach=미분 끊기(고정).
        pos_emb = self.encoder(pos).detach()  # [B, 64, T_pos, 65]
        neg_emb = self.encoder(neg).detach()  # [B, 64, T_neg, 65]

        # pos·neg 임베딩을 어텐션 헤드로 융합해 화자 조건(condition) 임베딩 생성.
        cond_emb = self.encoder_head(pos_emb, neg_emb).detach()  # [B, 64, T_pos+T_neg, 65]

        # 융합 결과에서 pos 길이만큼만 잘라서 반환(neg 쪽은 보조 정보였음).
        cond_emb = cond_emb[:, :, :pos_emb.shape[2]]  # slice back to pos length -> [B, 64, T_pos, 65]
        return cond_emb
