from typing import Dict, Tuple, Union
from torch import Tensor
# Vendored: asteroid.losses.singlesrc_neg_sisdr -> torchmetrics (asteroid intentionally NOT installed).
# asteroid's singlesrc_neg_sisdr returns the per-utterance NEGATIVE SI-SDR (it zero-means internally),
# and MeanFlow's wrapper means it over the batch. torchmetrics' functional SI-SDR with zero_mean=True
# is numerically the same quantity (preds first, target second); we negate and mean to match exactly.
import torch
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as _si_sdr


def neg_sisdr_loss_wrapper(est_targets, targets):
    """Mean negative SI-SDR over the batch. est_targets/targets are (B, T) waveforms."""
    return (-_si_sdr(est_targets, targets, zero_mean=True)).mean()


def mse_loss(est_targets, targets):
    return torch.mean((est_targets - targets) ** 2)


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

        return_stats:
            True이면 gate ratio 등의 통계도 함께 반환.

    Returns:
        loss 또는 (loss, stats)
    """

    if not 0.0 <= float(target_activity_threshold) <= 1.0:
        raise ValueError("target_activity_threshold must be in [0, 1].")
    if not 0.0 <= float(nuisance_activity_threshold) <= 1.0:
        raise ValueError("nuisance_activity_threshold must be in [0, 1].")

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

    est_targets = est_targets[..., :time_length]
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
    # [B, W, L]
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

    # [B, J, W, L]
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
    # [B, W, 1]
    target_energy = target_windows.square().sum(
        dim=-1,
        keepdim=True,
    )

    est_target_inner = (
        est_windows * target_windows
    ).sum(
        dim=-1,
        keepdim=True,
    )

    est_projection_scale = (
        est_target_inner / (target_energy + eps)
    )

    # [B, W, L]
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

    nuisance_projection_scale = (
        nuisance_target_inner
        / (target_energy_expanded + eps)
    )

    # [B, J, W, L]
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

    nuisance_orthogonal_energy = (
        nuisance_orthogonal.square().sum(dim=-1)
    )

    target_energy_flat = target_energy.squeeze(-1).unsqueeze(1)

    residual_energy = residual.square().sum(dim=-1)
    residual_energy_expanded = residual_energy.unsqueeze(1)
    normalization_energy = (
        residual_energy_expanded
        if normalize_residual_energy
        else target_energy_flat
    )

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
    nuisance_energy = (
        interferer_windows.square().sum(dim=-1)
    )

    orthogonal_ratio = (
        nuisance_orthogonal_energy
        / (nuisance_energy + eps)
    )

    orthogonal_gate = orthogonal_ratio >= gate_threshold

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
    valid_count_per_sample = gate.sum(dim=(1, 2))

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
