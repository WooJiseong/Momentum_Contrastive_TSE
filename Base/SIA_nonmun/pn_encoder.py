# ============================================================================
# 이 파일이 하는 일 (pn_encoder.py)
# ----------------------------------------------------------------------------
# "고정(frozen) PN 인코더" = 학습하지 않는 화자 인코더(speaker encoder) 를 불러오는
# 짧은 로더(loader) 파일입니다.
#   - PN 인코더: 등록(enrollment) 음성(목표 화자 예시 pos, 비목표 예시 neg)을 듣고
#     "이 사람이 누구인지"를 나타내는 임베딩(embedding, 화자 특징 벡터)을 뽑는 모듈.
#   - frozen(고정): 이 인코더는 절대 학습하지 않습니다. 미리 학습된 가중치를 그대로 씀.
#   - 체크포인트 checkpoints/proposed-monaural.pt 에서 'encoder.*' 와 'encoder_head.*'
#     로 시작하는 가중치만 골라 로드합니다(분리기 separator 가중치는 encode()가 안 씀).
#   - 불러온 뒤 eval() + requires_grad=False 로 완전히 얼립니다.
#
# ✏️ 바꿔도 되는 곳: 거의 없음. (호출 쪽에서 ckpt 경로/device 만 넘겨 주면 됨)
# ⚠️ 만지지 마세요: 가중치 필터 조건('encoder.'/'encoder_head.'로 시작), strict=True 로드,
#    eval()/requires_grad=False(=고정). 여기를 바꾸면 사전학습 화자 인코더가 깨집니다.
# ============================================================================

# Frozen PN-Enroll speaker encoder facade: load proposed-monaural.pt, freeze, expose encode().
"""PN encoder loader for NoisyFlowTSE.

Builds the minimal frozen encode path (pn.encoder.PNEncodePath), loads ONLY the
'encoder.*' + 'encoder_head.*' tensors out of checkpoints/proposed-monaural.pt (the
separator weights are unused by encode()), freezes (eval + requires_grad=False), and
returns a module exposing encode(pos, neg) -> [B, 64, T, 65].

The encoder is ALWAYS frozen and run under no_grad; bf16 autocast is applied by the caller
(the train/eval scripts) for speed — the encode-path RNN/attn tolerate bf16 (verified in the FlowSE port).
"""

from __future__ import annotations

import torch
from torch import nn

from pn.encoder import PNEncodePath


def load_pn_encoder(ckpt_path: str, device: torch.device) -> nn.Module:
    """Construct PNEncodePath, load the filtered checkpoint, freeze, move to device.

    Args:
        ckpt_path: path to checkpoints/proposed-monaural.pt (a dict with key 'state_dict').
        device: target device.
    Returns:
        a frozen (eval, requires_grad=False) module with .encode(pos, neg) -> float[B,64,T,65].
    Contract:
        load the subset of state_dict whose keys start with 'encoder.' or 'encoder_head.'
        into PNEncodePath with strict=True (must match exactly, no missing/unexpected).
    """
    # 빈 인코더 뼈대를 먼저 만들고(아래에서 가중치를 채워 넣습니다).
    model = PNEncodePath()

    # 체크포인트 파일을 CPU로 읽습니다. 'state_dict' 안에 모든 가중치가 들어 있음.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    # 체크포인트에는 분리기(separator) 전체가 들어 있지만, encode()에 필요한 두 부분
    # ('encoder.*' 와 'encoder_head.*')만 골라냅니다. 나머지는 버립니다.
    # The checkpoint holds the FULL separator; keep only the two encode-path submodules.
    enc_state = {
        k: v for k, v in state.items()
        if k.startswith("encoder.") or k.startswith("encoder_head.")
    }
    # strict=True: 키가 하나라도 안 맞으면 에러. 정확히 일치해야만 통과(안전장치).
    model.load_state_dict(enc_state, strict=True)

    # 여기서부터 완전히 고정(frozen): 평가 모드 + 미분(역전파) 끄기.
    model.eval()  # FROZEN: eval (LSTM/attn deterministic, no dropout state)
    for p in model.parameters():
        p.requires_grad = False

    return model.to(device)
