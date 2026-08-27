from typing import Dict, Mapping, Tuple, Union

import torch
from torch import Tensor


def build_target_orthogonal_leakage_kwargs(
    loss_cfg: Mapping[str, object],
) -> Dict[str, object]:
    """Translate YAML loss options into shared loss-call arguments."""
    return {
        "window_size": int(loss_cfg.get("orthogonal_window_size", 2048)),
        "hop_size": int(loss_cfg.get("orthogonal_hop_size", 1024)),
        "gate_threshold": float(loss_cfg.get("orthogonal_gate_threshold", 0.1)),
        "eps": float(loss_cfg.get("orthogonal_eps", 1e-8)),
        "normalize_residual_energy": bool(
            loss_cfg.get("normalize_residual_energy", True)
        ),
        "activity_gate": bool(loss_cfg.get("activity_gate", True)),
        "target_activity_threshold": float(
            loss_cfg.get("target_activity_threshold", 0.01)
        ),
        "nuisance_activity_threshold": float(
            loss_cfg.get("nuisance_activity_threshold", 0.01)
        ),
        "gate_mode": str(loss_cfg.get("gate_mode", "waveform_orthogonal")),
        "stft_n_fft": int(loss_cfg.get("stft_n_fft", 510)),
        "stft_hop_length": int(loss_cfg.get("stft_hop_length", 128)),
        "stft_win_length": int(loss_cfg.get("stft_win_length", 510)),
        "stft_center": bool(loss_cfg.get("stft_center", False)),
        "stft_magnitude_overlap_threshold": float(
            loss_cfg.get("stft_magnitude_overlap_threshold", 0.85)
        ),
    }


def _stft_magnitude_overlap(
    target_windows: Tensor,
    interferer_windows: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool,
    eps: float,
) -> Tensor:
    """Return [B, J, W] cosine overlap of STFT magnitudes."""
    batch_size, window_count, window_samples = target_windows.shape
    source_count = interferer_windows.shape[1]
    if not 1 <= win_length <= n_fft <= window_samples:
        raise ValueError(
            "STFT requires 1 <= win_length <= n_fft <= waveform window size; "
            f"got win_length={win_length}, n_fft={n_fft}, window={window_samples}"
        )
    if hop_length <= 0:
        raise ValueError("stft_hop_length must be positive")

    stft_window = torch.hann_window(
        win_length, device=target_windows.device, dtype=target_windows.dtype
    )
    target_mag = torch.stft(
        target_windows.reshape(batch_size * window_count, window_samples),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=stft_window,
        center=center,
        return_complex=True,
    ).abs()
    nuisance_mag = torch.stft(
        interferer_windows.reshape(batch_size * source_count * window_count, window_samples),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=stft_window,
        center=center,
        return_complex=True,
    ).abs()

    target_vec = target_mag.reshape(batch_size, window_count, -1)
    nuisance_vec = nuisance_mag.reshape(batch_size, source_count, window_count, -1)
    target_norm = target_vec.square().sum(dim=-1).sqrt().unsqueeze(1)
    nuisance_norm = nuisance_vec.square().sum(dim=-1).sqrt()
    overlap = (
        (nuisance_vec * target_vec.unsqueeze(1)).sum(dim=-1)
        / (nuisance_norm * target_norm + eps)
    )
    return overlap.clamp(0.0, 1.0)


def target_orthogonal_leakage_loss(
    est_targets: Tensor,
    targets: Tensor,
    interferers: Tensor,
    window_size: int = 2048,
    hop_size: int = 1024,
    gate_threshold: float = 0.1,
    eps: float = 1e-8,
    normalize_residual_energy: bool = True,
    activity_gate: bool = True,
    target_activity_threshold: float = 0.01,
    nuisance_activity_threshold: float = 0.01,
    gate_mode: str = "waveform_orthogonal",
    stft_n_fft: int = 510,
    stft_hop_length: int = 128,
    stft_win_length: int = 510,
    stft_center: bool = False,
    stft_magnitude_overlap_threshold: float = 0.85,
    return_stats: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """
    Target-orthogonal per-source leakage loss.

    Args:
        est_targets:
            추출된 목표 음성.
            Shape: [B, T] 또는 [B, 1, T]

        targets:
            정답 목표 음성.
            Shape: [B, T] 또는 [B, 1, T]

        interferers:
            mixture를 구성한 개별 방해 음원.
            Shape: [B, J, T]
            J는 nuisance source 개수.

        window_size:
            프레임 길이. 16 kHz에서 2048 samples = 128 ms.

        hop_size:
            프레임 이동 크기. 16 kHz에서 1024 samples = 64 ms.

        gate_threshold:
            방해 음원이 목표 화자와 충분히 직교하는 window만
            loss에 포함하기 위한 threshold.

        eps:
            수치 안정성을 위한 작은 값.

        normalize_residual_energy:
            True이면 residual energy를 분모에 추가하여 residual과
            nuisance의 정규화된 방향 유사도(cosine-squared)를 계산한다.

        activity_gate:
            True이면 저에너지 target/nuisance window를 leakage 계산에서
            제외한다. 각 sample/source 내부 최대 window energy 대비 상대
            threshold를 사용한다.

        target_activity_threshold:
            target window energy가 sample 내부 최대값의 이 비율보다 작으면
            비활성 구간으로 제외한다.

        nuisance_activity_threshold:
            nuisance window energy가 source 내부 최대값의 이 비율보다 작으면
            비활성 구간으로 제외한다.

        gate_mode:
            waveform_orthogonal은 기존 waveform projection Gate를 사용한다.
            stft_magnitude_overlap은 target/nuisance waveform Window의
            STFT magnitude cosine overlap이 threshold 이하인 Window만 사용한다.

        stft_n_fft, stft_hop_length, stft_win_length, stft_center:
            STFT magnitude Gate의 파라미터.

        stft_magnitude_overlap_threshold:
            STFT magnitude overlap 상한. overlap이 이 값 이하인 Window만
            Gate를 통과한다. waveform threshold와 의미가 다르다.

        return_stats:
            True이면 gate ratio 등의 통계도 함께 반환.

    Returns:
        loss 또는 (loss, stats)
    """

    if not 0.0 <= float(target_activity_threshold) <= 1.0:
        raise ValueError("target_activity_threshold must be in [0, 1].")
    if not 0.0 <= float(nuisance_activity_threshold) <= 1.0:
        raise ValueError("nuisance_activity_threshold must be in [0, 1].")
    gate_mode = str(gate_mode).lower().strip()
    if gate_mode not in {"waveform_orthogonal", "stft_magnitude_overlap"}:
        raise ValueError(
            "gate_mode must be 'waveform_orthogonal' or "
            f"'stft_magnitude_overlap', got {gate_mode!r}"
        )
    if not 0.0 <= float(stft_magnitude_overlap_threshold) <= 1.0:
        raise ValueError("stft_magnitude_overlap_threshold must be in [0, 1].")

    # ---------------------------------------------------------
    # 1. 입력 차원 정리
    # ---------------------------------------------------------
    if est_targets.ndim == 3:
        est_targets = est_targets.squeeze(1)

    if targets.ndim == 3:
        targets = targets.squeeze(1)

    if interferers.ndim == 2:
        interferers = interferers.unsqueeze(1)

    if est_targets.ndim != 2:
        raise ValueError(
            f"est_targets must be [B, T], got {est_targets.shape}"
        )

    if targets.ndim != 2:
        raise ValueError(
            f"targets must be [B, T], got {targets.shape}"
        )

    if interferers.ndim != 3:
        raise ValueError(
            f"interferers must be [B, J, T], got {interferers.shape}"
        )

    batch_size = est_targets.shape[0]

    if targets.shape[0] != batch_size:
        raise ValueError("Batch size mismatch between estimate and target.")

    if interferers.shape[0] != batch_size:
        raise ValueError(
            "Batch size mismatch between estimate and interferers."
        )

    # bf16/fp16에서 projection 연산이 불안정할 수 있어 fp32 사용
    est_targets = est_targets.float()
    targets = targets.float()
    interferers = interferers.float()

    # 길이가 다를 경우 공통 길이만 사용
    time_length = min(
        est_targets.shape[-1],
        targets.shape[-1],
        interferers.shape[-1],
    )

    est_targets = est_targets[..., :time_length] # 가장 짧은 길이에 맞춰 자르기
    targets = targets[..., :time_length]
    interferers = interferers[..., :time_length]

    if time_length < window_size:
        raise ValueError(
            f"Audio length {time_length} is shorter than "
            f"window_size {window_size}."
        )

    # ---------------------------------------------------------
    # 2. Local window 생성
    # ---------------------------------------------------------
    # [B, W, L] Ex : [1,45,2048] (window_size=2048, hop_size=1024, T=48000)
    est_windows = est_targets.unfold(
        dimension=-1,
        size=window_size,
        step=hop_size,
    )

    target_windows = targets.unfold(
        dimension=-1,
        size=window_size,
        step=hop_size,
    )

    # [B, J, W, L] (J = nuisance source (부정화자 + Noise)개수)
    interferer_windows = interferers.unfold(
        dimension=-1,
        size=window_size,
        step=hop_size,
    )

    # ---------------------------------------------------------
    # 3. 추정 음성에서 target 방향 제거
    #
    # r = est - proj_target(est)
    # ---------------------------------------------------------
    # [B, W, 1] 각 Window 크기 구함 (L 방향으로 제곱합) -> 정규화 위해 사용
    target_energy = target_windows.square().sum(
        dim=-1,
        keepdim=True,
    )
    # [B, W, 1] 각 Window에서 est와 target의 내적 (L 방향 -> 추론 음성과 target이 유사하면 최대)
    est_target_inner = (
        est_windows * target_windows
    ).sum(
        dim=-1,
        keepdim=True,
    )
    # 범위 : [0, 1] (est와 target이 유사하면 1, 직교하면 0)
    est_projection_scale = (
        est_target_inner / (target_energy + eps)
    )

    # [B, W, L] Broadcast하여 est에서 target 방향 제거 (GT와 다른 부분을 강조)
    residual = (
        est_windows
        - est_projection_scale * target_windows
    )

    # ---------------------------------------------------------
    # 4. 각 nuisance에서도 target 방향 제거
    #
    # b_perp = b - proj_target(b)
    # ---------------------------------------------------------
    # [B, 1, W, L]
    target_windows_expanded = target_windows.unsqueeze(1)

    # [B, 1, W, 1]
    target_energy_expanded = target_energy.unsqueeze(1)

    # [B, J, W, 1]
    nuisance_target_inner = (
        interferer_windows * target_windows_expanded
    ).sum(
        dim=-1,
        keepdim=True,
    )
    # 범위 : [0, 1] (nuisance와 target이 유사하면 1, 직교하면 0) 크기 : [B, J, W, 1]
    nuisance_projection_scale = (
        nuisance_target_inner
        / (target_energy_expanded + eps)
    )

    # [B, J, W, L] nuisance에서 target 방향 제거 (이유 : Target 음원에 orthogonal한 음원 성분만 leakage로 계산하기 위함)
    nuisance_orthogonal = (
        interferer_windows
        - nuisance_projection_scale * target_windows_expanded
    )

    # ---------------------------------------------------------
    # 5. Local leakage 계산
    #
    # psi_jw =
    # |<r_w, b_perp_jw>|^2
    # --------------------------------
    # (||b_perp_jw||^2 + eps)
    # (||s_w||^2 + eps)
    # ---------------------------------------------------------
    residual_expanded = residual.unsqueeze(1)

    # [B, J, W]
    residual_nuisance_inner = (
        residual_expanded * nuisance_orthogonal
    ).sum(dim=-1)
    # [B, J, W] nuisance_orthogonal의 에너지 Nuisance orthogonal과 residual의 코사인 유사도 계산을 위해 분모에 사용
    nuisance_orthogonal_energy = (
        nuisance_orthogonal.square().sum(dim=-1)
    )

    target_energy_flat = target_energy.squeeze(-1).unsqueeze(1)

    residual_energy = residual.square().sum(dim=-1)
    residual_energy_expanded = residual_energy.unsqueeze(1)
    normalization_energy = ( # normalize_residual_energy가 True이면 residual energy를 분모 추가, residual과 nuisance의 정규화된 방향 유사도 계산한다.
        residual_energy_expanded
        if normalize_residual_energy
        else target_energy_flat
    )
    # [B, J, W] 각 source/window별 leakage 계산 -> residual (est_target에 target 방향 제거)와 nuisance_orthogonal (nuisance에서 target 방향 제거) 사이의 코사인 유사도 계산
    local_leakage = residual_nuisance_inner.square() / (
        (nuisance_orthogonal_energy + eps)
        * (normalization_energy + eps)
    )

    # ---------------------------------------------------------
    # 6. Orthogonality gate
    #
    # q_jw = ||b_perp||^2 / (||b||^2 + eps)
    # g_jw = 1[q_jw >= threshold]
    # ---------------------------------------------------------
    nuisance_energy = ( # 방해화자 정규화 에너지
        interferer_windows.square().sum(dim=-1)
    )

    orthogonal_ratio = ( # [B, J, W], 범위 [0,1] 커지면 nuisance가 target과 orthogonal, 작으면 target과 유사
        nuisance_orthogonal_energy
        / (nuisance_energy + eps)
    )
    # nuisance가 target과 직교했을 때 q_jw가 1로 수렴 (Threshold가 0.1이라서, 너무 낮았던 Issue 존재 / Ratio 중앙값 0.9997058510780334)
    spectral_overlap = None
    if gate_mode == "stft_magnitude_overlap":
        spectral_overlap = _stft_magnitude_overlap(
            target_windows,
            interferer_windows,
            n_fft=int(stft_n_fft),
            hop_length=int(stft_hop_length),
            win_length=int(stft_win_length),
            center=bool(stft_center),
            eps=eps,
        )
        orthogonal_gate = spectral_overlap <= float(
            stft_magnitude_overlap_threshold
        )
    else:
        orthogonal_gate = orthogonal_ratio >= gate_threshold
    # reference_energy는 sample 내부에서 가장 큰 window energy (target/nuisance 각각) Threshold 비율만큼의 window만 activity gate 통과
    target_window_energy = target_energy.squeeze(-1)
    target_reference_energy = target_window_energy.amax(dim=1, keepdim=True)
    target_active = (
        (target_reference_energy > eps)
        & (target_window_energy >= target_reference_energy * target_activity_threshold)
    )

    nuisance_reference_energy = nuisance_energy.amax(dim=-1, keepdim=True)
    nuisance_active = (
        (nuisance_reference_energy > eps)
        & (nuisance_energy >= nuisance_reference_energy * nuisance_activity_threshold)
    )

    if activity_gate:
        gate = (
            orthogonal_gate
            & target_active.unsqueeze(1)
            & nuisance_active
        ).to(local_leakage.dtype)
    else:
        gate = orthogonal_gate.to(local_leakage.dtype)

    # ---------------------------------------------------------
    # 7. Source/window별 leakage의 정규화 평균
    #
    # L =
    # sum_jw g_jw * psi_jw
    # --------------------
    # sum_jw g_jw + eps
    # ---------------------------------------------------------
    valid_count_per_sample = gate.sum(dim=(1, 2)) # Gate 통화 안하면 안 셈

    leakage_per_sample = (
        (gate * local_leakage).sum(dim=(1, 2))
        / (valid_count_per_sample + eps)
    )

    loss = leakage_per_sample.mean()

    if not return_stats:
        return loss

    total_valid = gate.sum()

    stats = {
        # 전체 window 중 gate를 통과한 비율
        "gate_ratio": gate.mean().detach(),

        # nuisance가 target과 얼마나 직교하는지
        "orthogonal_ratio": orthogonal_ratio.mean().detach(),
        "stft_magnitude_overlap": (
            spectral_overlap.mean().detach()
            if spectral_overlap is not None
            else orthogonal_ratio.new_tensor(float("nan"))
        ),
        "stft_magnitude_gate_ratio": (
            orthogonal_gate.float().mean().detach()
            if spectral_overlap is not None
            else orthogonal_ratio.new_tensor(float("nan"))
        ),

        # activity gate를 통과한 target/nuisance window 비율
        "target_activity_ratio": target_active.float().mean().detach(),
        "nuisance_activity_ratio": nuisance_active.float().mean().detach(),

        # gate를 통과한 window에서의 평균 leakage
        "local_leakage": (
            (gate * local_leakage).sum()
            / (total_valid + eps)
        ).detach(),

        # 샘플당 유효 source-window 개수
        "valid_windows": (
            valid_count_per_sample.float().mean().detach()
        ),
    }

    return loss, stats
