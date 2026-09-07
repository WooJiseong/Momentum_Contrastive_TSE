# ============================================================================
# 이 파일이 하는 일 (meanflow.py)
# ----------------------------------------------------------------------------
# 이 파일은 "흐름 매칭(flow matching) 학습의 손실(loss) 계산기" 입니다.
# 우리 모델은 잡음 섞인 소리(mixture/background, 시작점 t=0)에서 깨끗한 목표 화자
# 소리(source, 도착점 t=1)로 "흘러가는 속도(velocity)" 를 배웁니다. 이 파일은
# 그 속도를 어떻게 맞춰야 하는지(=손실)를 여러 방식으로 정의해 둔 곳입니다.
#
# 핵심 개념 (처음 보면 헷갈리는 단어들):
#   - flow matching(흐름 매칭): "어디로 얼마나 빨리 움직여야 하는가(속도)"를
#     신경망이 배우는 방식. 한 번에 목표로 점프(1-step)할 수 있게 해 줍니다.
#   - t (시간 변수): 0이면 잡음 섞인 시작, 1이면 깨끗한 목표. 그 사이 값은 중간 지점.
#   - alpha(알파) 스케줄: 학습 초반엔 "정직한 궤적 따라가기(trajectory flow, alpha=1)",
#     후반엔 "한 방에 도착하기 연습(consistency, alpha→0)"으로 서서히 바꾸는 커리큘럼.
#   - MR-jitter(엠알 지터): 추론 때 시작 시점 t를 t-predicter가 살짝 틀리게 줘도
#     1-step 결과가 깨끗하게 착지하도록 일부러 시작 시점을 흔들어(jitter) 훈련하는 것.
#
# ✏️ 바꿔도 되는 곳:
#   - MeanFlowTSE 생성자의 하이퍼파라미터 기본값(flow_ratio, alpha_schedule_*,
#     alpha_gamma, alpha_min, gamma). 보통은 config 파일에서 넘겨주니 그쪽을 권장.
# ⚠️ 만지지 마세요:
#   - 손실 수식(보간 z, 목표 속도 v, target 계산, adaptive_l2_loss 안의 가중치 w).
#     수식 한 줄만 틀려도 학습이 통째로 망가집니다. (검증된 코드)
# ============================================================================

import torch
from einops import rearrange
import numpy as np


# stopgrad: "여기서 역전파(미분) 멈춰" 표시. target 쪽 값이 학습되지 않게 끊습니다.
def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.0, c=1e-3, alpha=None):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    
    Args:
        error: Error tensor (B, C, T) or (B, C, H, W)
        gamma: Exponent parameter (default: 0.0)
        c: Small constant for numerical stability (default: 1e-3)
        alpha: Optional alpha value for alpha-dependent weighting (default: None)
               When provided, uses w = alpha / (delta_sq + c)^p instead of w = 1 / (delta_sq + c)^p
    """
    # error(예측 속도 - 정답 속도)의 제곱을 샘플마다 평균 -> 샘플별 오차 크기 delta_sq.
    if error.ndim == 3:
        delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    elif error.ndim == 4:
        delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {error.ndim}D")

    p = 1.0 - gamma

    # 가중치 w 계산: 오차가 큰 샘플의 영향을 적당히 눌러서(자기 정규화) 학습을 안정화.
    # alpha가 주어지면 alpha를 곱해 "정직한 궤적(alpha=1) ~ 한방 도착(alpha→0)" 비중을 조절.
    # Apply alpha-dependent weighting if alpha is provided
    if alpha is not None:
        # For alpha-Flow: w = alpha / (delta_sq + c)^p
        w = alpha**p / (delta_sq + c).pow(p)
    else:
        # Standard weighting: w = 1 / (delta_sq + c)^p
        w = 1.0 / (delta_sq + c).pow(p)
    
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlowTSE:
    """
    MeanFlowTSE: Target speaker extraction with flow matching objectives.
    Includes alpha scheduling for curriculum learning from trajectory flow matching
    to consistency training.
    
    Convention:
    - t=0: background/mixture (noisy starting point)
    - t=1: source (clean target)
    - r ≥ t always (r is the target timestep, moving forward in time)
    """
    
    def __init__(
        self,
        flow_ratio=0.50,
        use_enrollment=True,
        data_dim='1d',
        alpha_schedule_start=0,  # Iteration to start transitioning from alpha=1
        alpha_schedule_end=None,  # Iteration to finish transitioning to alpha=0
        alpha_gamma=25.0,  # Temperature parameter for sigmoid schedule
        alpha_min=5e-3,  # Minimum alpha value (clamping)
        gamma=0.0,  # Gamma parameter for adaptive loss weighting
        time_sampling_mode='full',
        tau_margin=0.1,
    ):
        super().__init__()
        self.use_enrollment = use_enrollment
        self.data_dim = data_dim
        self.flow_ratio = flow_ratio
        self.gamma = gamma

        # Time-support sampling:
        #   full          -> original [0, 1] sampling
        #   tau_truncated -> [max(0, tau - margin), 1]
        self.time_sampling_mode = str(time_sampling_mode)
        self.tau_margin = float(tau_margin)

        valid_sampling_modes = {
            'full',
            'tau_truncated',
        }

        if self.time_sampling_mode not in valid_sampling_modes:
            raise ValueError(
                f"Unknown time_sampling_mode: "
                f"{self.time_sampling_mode}"
            )

        if self.tau_margin < 0.0:
            raise ValueError(
                f"tau_margin must be non-negative, "
                f"got {self.tau_margin}"
            )
        
        # Alpha scheduling parameters
        self.alpha_schedule_start = alpha_schedule_start
        self.alpha_schedule_end = alpha_schedule_end
        self.alpha_gamma = alpha_gamma
        self.alpha_min = alpha_min
        
        # Track current training iteration
        self.current_iteration = 0

    def _rescale_time_samples(
        self,
        samples,
        mixing_ratio=None,
    ):
        """Rescale unit-interval samples to the configured support.

        For sample i:

            lower_i = max(0, tau_i - tau_margin)
            mapped  = lower_i + (1 - lower_i) * samples

        Therefore:
            full          -> samples remain in [0, 1]
            tau_truncated -> samples lie in [lower_i, 1]
        """
        if self.time_sampling_mode == 'full':
            return samples

        if mixing_ratio is None:
            raise ValueError(
                "mixing_ratio is required for "
                "tau_truncated sampling."
            )

        batch_size = samples.shape[0]

        tau = torch.as_tensor(
            mixing_ratio,
            device=samples.device,
            dtype=samples.dtype,
        )

        if tau.ndim == 0:
            tau = tau.expand(batch_size)
        elif tau.shape[0] == batch_size:
            tau = tau.reshape(batch_size, -1).mean(dim=1)
        else:
            tau = tau.reshape(-1)

            if tau.numel() == 1:
                tau = tau.expand(batch_size)
            elif tau.numel() != batch_size:
                raise ValueError(
                    "mixing_ratio shape mismatch: "
                    f"tau={tuple(tau.shape)}, "
                    f"batch_size={batch_size}"
                )

        lower = (
            tau - self.tau_margin
        ).clamp(
            min=0.0,
            max=1.0 - 1e-5,
        )

        # [B] 또는 [B, K] 모두 지원
        if samples.ndim > 1:
            lower = lower.view(
                batch_size,
                *([1] * (samples.ndim - 1)),
            )

        return lower + (1.0 - lower) * samples

    def get_alpha(self, iteration=None, return_raw=False):
        """
        Compute alpha value based on the current training iteration using sigmoid schedule.
        
        Args:
            iteration: Current training iteration (uses self.current_iteration if None)
            return_raw: If True, return (clipped_alpha, raw_alpha_before_clipping)
        
        Returns:
            alpha: Value in (0, 1], with 1 = trajectory flow matching, 0 = consistency training
            If return_raw=True: tuple of (clipped_alpha, raw_alpha)
        """
        if iteration is None:
            iteration = self.current_iteration

        # 스케줄(언제부터 언제까지 alpha를 1->0으로 바꿀지)이 안 정해져 있으면
        # 항상 alpha=1 = "정직한 궤적 따라가기"만 합니다.
        # If no schedule defined, use alpha=1 (pure trajectory flow matching mode)
        if self.alpha_schedule_end is None:
            return (1.0, 1.0) if return_raw else 1.0
        
        # Before schedule starts, use alpha=1 (trajectory flow matching pretraining)
        if iteration < self.alpha_schedule_start:
            return (1.0, 1.0) if return_raw else 1.0
        
        # After schedule ends, use alpha≈0 (consistency training fine-tuning)
        if iteration >= self.alpha_schedule_end:
            # Compute raw alpha even after schedule ends
            k_s = self.alpha_schedule_start
            k_e = self.alpha_schedule_end
            scale = 1.0 / (k_e - k_s)
            offset = -(k_s + k_e) / 2.0 / (k_e - k_s)
            x = (scale * iteration + offset) * self.alpha_gamma
            raw_alpha = 1.0 - torch.sigmoid(torch.tensor(x)).item()
            return (self.alpha_min, raw_alpha) if return_raw else self.alpha_min
        
        # 전환 구간: 학습이 진행될수록 alpha를 1 -> 0 으로 부드럽게(S자 곡선=sigmoid) 낮춥니다.
        # scale/offset은 현재 iteration을 -1~1 범위로 옮겨 주는 자(척도)이고,
        # alpha_gamma(온도)가 클수록 더 급격하게(가파른 S자) 전환됩니다.
        # During transition: sigmoid schedule
        k_s = self.alpha_schedule_start
        k_e = self.alpha_schedule_end
        scale = 1.0 / (k_e - k_s)
        offset = -(k_s + k_e) / 2.0 / (k_e - k_s)

        # alpha = 1 - sigmoid((scale * k + offset) * gamma)
        x = (scale * iteration + offset) * self.alpha_gamma
        raw_alpha = 1.0 - torch.sigmoid(torch.tensor(x)).item()
        
        # 클램핑(clamping, 값 자르기): 너무 1에 가까우면 그냥 1로, 너무 0에 가까우면
        # alpha_min으로 딱 잘라서 위상(phase) 판정을 깔끔하게 만듭니다.
        # Apply clamping
        if raw_alpha > (1.0 - self.alpha_min):
            clipped_alpha = 1.0
        elif raw_alpha < self.alpha_min:
            clipped_alpha = self.alpha_min
        else:
            clipped_alpha = raw_alpha

        return (clipped_alpha, raw_alpha) if return_raw else clipped_alpha

    def loss_rectified_flow(
        self,
        model,
        source,
        background,
        enrollment,
        alpha=None,
        mixing_ratio=None,
    ):
        """
        Rectified flow loss with alpha-dependent weighting.
        Uses sigmoid(randn) for t sampling and sets r = t.
        
        Flow direction: background (t=0) -> source (t=1)
        
        Args:
            model: The neural network model
            source: Clean source (t=1)
            background: Background/noise (t=0)
            enrollment: Enrollment/reference signals
            alpha: Optional alpha value for weighting. If None, uses 1.0 (standard weighting).
        """
        batch_size = source.shape[0]
        device = source.device

        # 시간 t를 무작위로 뽑습니다(정규분포 -> sigmoid 로 0~1 사이로 변환).
        # Sample time using sigmoid of normal
        nt = torch.randn(batch_size, device=device)
        t = torch.sigmoid(nt)

        t = self._rescale_time_samples(
            t,
            mixing_ratio=mixing_ratio,
        )

        # rectified flow(직선 흐름)에서는 도착 시점 r을 시작 시점 t와 같게 둡니다.
        # For rectified flow, r = t
        r = t.clone()

        # 시간을 텐서 모양(b 1 1...)으로 바꿔서 배경/소스와 곱할 수 있게 맞춥니다.
        # Reshape time variables
        if self.data_dim == '1d':
            t_ = rearrange(t, "b -> b 1 1")
        else:
            t_ = rearrange(t, "b -> b 1 1 1")

        # 배경(t=0)과 소스(t=1)를 t 비율로 직선 보간 -> 중간 지점 z 를 만듭니다.
        # Interpolate: z = (1-t) * background + t * source
        z = (1 - t_) * background + t_ * source

        # 정답 속도 v: 배경에서 소스로 가는 방향(소스 - 배경). 직선이라 항상 일정.
        # Target velocity: from background to source
        v = source - background

        # 모델이 예측한 속도 u.
        # Model prediction
        u = model(z, t, r, enrollment=enrollment)

        # 손실: 예측 속도 u 가 정답 속도 v 와 같아지도록 (alpha 가중치 적용).
        # Compute loss with alpha-dependent adaptive weighting
        error = u - v
        loss_alpha = alpha if alpha is not None else 1.0
        loss = adaptive_l2_loss(error, gamma=self.gamma, alpha=loss_alpha)
        mse_val = (error ** 2).mean()

        return loss, mse_val

    def loss_mr_jitter(
        self,
        model,
        source,
        mixture,
        mixing_ratio,
        enrollment,
        sigma,
        return_stats=False,
    ):
        """MR-ROBUST training (continue-training objective): make the 1-step Euler FROM THE MIXTURE land on
        the clean target even when given a JITTERED, possibly-wrong start time m_hat. The mixture sits at
        z(m_true) on the background->source path; we tell the model a Gaussian-jittered time
        m_hat = clamp(m_true + N(0, sigma)) and regress its velocity onto the MEAN velocity to the endpoint
        target = (source - mixture)/(1 - m_hat). Then inference x1 = mixture + (1-m_hat)*model(mixture,m_hat,1)
        ~= source for ANY m_hat -> robust to t-predictor error. At sigma->0 (m_hat=m_true) the target reduces
        EXACTLY to the rectified velocity source-background, so this is a smooth generalization of training."""
        # -- 왜 jitter(흔들기)가 필요한가? --------------------------------------
        # 추론은 단 한 번(1-step)에 끝납니다: mixture(섞인 소리)에서 출발해 깨끗한 소스로
        # 점프하죠. 이때 "출발 시점 t" 는 t-predicter(섞인 비율 추정기)가 추정해 주는데,
        # 그 추정값이 살짝 틀릴 수 있습니다. 만약 정확한 t에서만 훈련하면, 추론 때 t가
        # 조금만 틀려도 결과가 망가집니다.
        # 그래서 일부러 시작 시점을 가우시안 잡음으로 흔든 값 m_hat 을 모델에 알려 주고,
        # "어떤 m_hat 이 와도 1-step 으로 깨끗한 소스에 착지" 하도록 훈련합니다.
        # 결과: t-predicter 가 조금 틀려도 깔끔하게 도착(robust, 견고).
        # sigma=0(흔들기 0)이면 m_hat=m_true 라서 그냥 직선 흐름(rectified flow)과 똑같아짐
        # -> 즉 이 손실은 직선 흐름의 부드러운 확장판입니다.
        # ----------------------------------------------------------------------
        batch_size = source.shape[0]
        device = source.device
        # m: 실제 섞인 비율(=섞인 소리가 놓인 진짜 시작 시점). 0/1 끝값은 살짝 안쪽으로 자름.
        m = torch.as_tensor(
            mixing_ratio,
            device=device,
            dtype=torch.float32,
        ).reshape(batch_size).clamp(
            min=1e-3,
            max=1.0 - 1e-3,
        )
        # m_hat: m 에 가우시안 잡음(sigma 크기)을 더해 "틀린 시작 시점"을 흉내 낸 값.
        raw_m_hat = (
            m
            + torch.randn(
                batch_size,
                device=device,
                dtype=m.dtype,
            ) * sigma
        )

        if self.time_sampling_mode == 'tau_truncated':
            # 각 샘플의 실제 mixture 위치 tau=m을 기준으로
            # MR-jitter 시작 시간이 tau-margin보다 아래로 내려가지 않게 한다.
            support_lower = (
                m - self.tau_margin
            ).clamp(
                min=1e-3,
                max=1.0 - 1e-3,
            )

            m_hat = torch.maximum(
                raw_m_hat,
                support_lower,
            ).clamp(
                max=1.0 - 1e-3,
            )

        elif self.time_sampling_mode == 'full':
            # 기존 v1 MR-jitter와 동일한 동작
            support_lower = torch.full_like(
                m,
                1e-3,
            )

            m_hat = raw_m_hat.clamp(
                min=1e-3,
                max=1.0 - 1e-3,
            )

        else:
            raise ValueError(
                f"Unknown time_sampling_mode: "
                f"{self.time_sampling_mode}"
            )
        r = torch.ones(batch_size, device=device)                        # 도착 시점은 항상 끝(t=1)
        if self.data_dim == '1d':
            denom = rearrange(1.0 - m_hat, "b -> b 1 1")
        else:
            denom = rearrange(1.0 - m_hat, "b -> b 1 1 1")
        # 목표 속도: m_hat 에서 소스(끝)까지 한 번에 가려면 가져야 할 평균 속도.
        #   x1 = mixture + (1 - m_hat) * 속도  가 소스가 되도록 -> 속도 = (source - mixture)/(1 - m_hat)
        target = (source - mixture) / denom                              # mean velocity mixture->source over [m_hat,1]
        pred = model(mixture, m_hat, r, enrollment=enrollment)
        error = pred - target
        loss = adaptive_l2_loss(error, gamma=self.gamma, alpha=1.0)
        mse_val = (error ** 2).mean()

        stats = {
            'mr_tau_mean': m.detach().mean(),
            'mr_time_lower_mean': (
                support_lower.detach().mean()
            ),
            'mr_time_mean': m_hat.detach().mean(),
            'mr_time_min': m_hat.detach().amin(),
            'mr_time_max': m_hat.detach().amax(),

            # clamp 전 Gaussian sample 중 하한보다 낮았던 비율
            'mr_raw_below_lower_ratio': (
                raw_m_hat.detach()
                < support_lower.detach()
            ).float().mean(),

            # 최종 m_hat이 하한을 위반한 비율: 반드시 0이어야 함
            'mr_time_below_lower_ratio': (
                m_hat.detach()
                < support_lower.detach() - 1e-7
            ).float().mean(),

            'mr_tau_truncated_enabled': (
                m_hat.detach().new_tensor(
                    float(
                        self.time_sampling_mode
                        == 'tau_truncated'
                    )
                )
            ),
        }

        if return_stats:
            return loss, mse_val, stats

        return loss, mse_val

    def loss_alpha_flow(
        self,
        model,
        source,
        background,
        enrollment,
        alpha,
        mixing_ratio=None,
    ):
        """
        Alpha-Flow loss for alpha ∈ (0, 1).
        Interpolates between trajectory flow matching (alpha=1) and
        consistency training (alpha→0).

        The intermediate point is advanced along the ground-truth trajectory
        (velocity v = source - background), and the target mixes that true
        velocity with the model's bootstrapped prediction from s to r:
            target = alpha * v + (1 - alpha) * u_sr

        Args:
            model: The neural network model
            source: Clean source (t=1)
            background: Background/noise (t=0)
            enrollment: Enrollment/reference signals
            alpha: Consistency step ratio in (0, 1]
        """
        batch_size = source.shape[0]
        device = source.device

        # 시작 t 와 도착 r 두 시점을 무작위로 뽑습니다(logistic 분포).
        # Sample t and r from logistic distribution
        mu, sigma = -0.4, 1.0
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))  # sigmoid

        # 둘 중 작은 값을 t, 큰 값을 r 로 정해 항상 r >= t (도착이 시작보다 앞) 보장.
        # Ensure r >= t (r is the target, ahead in flow)
        t_np = np.minimum(samples[:, 0], samples[:, 1])
        r_np = np.maximum(samples[:, 0], samples[:, 1])

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)

        # 같은 affine mapping을 t와 r에 적용하므로
        # 기존의 t <= r 순서가 그대로 유지된다.
        tr = torch.stack(
            [t, r],
            dim=1,
        )

        tr = self._rescale_time_samples(
            tr,
            mixing_ratio=mixing_ratio,
        )

        t = tr[:, 0]
        r = tr[:, 1]

        # 중간 시점 s: t 와 r 사이를 alpha 비율로 나눈 지점(자기 예측을 부트스트랩할 자리).
        # Intermediate time s = alpha * r + (1 - alpha) * t
        s = alpha * r + (1 - alpha) * t

        # Reshape time variables
        if self.data_dim == '1d':
            t_ = rearrange(t, "b -> b 1 1")
            s_ = rearrange(s, "b -> b 1 1")
        else:
            t_ = rearrange(t, "b -> b 1 1 1")
            s_ = rearrange(s, "b -> b 1 1 1")

        # t 시점의 중간 지점 x_t 와 정답 속도 v(소스-배경).
        # Point at time t and the ground-truth velocity
        x_t = (1 - t_) * background + t_ * source
        v = source - background

        # t -> r 로 바로 예측한 속도 u_tr (이게 우리가 학습시킬 대상).
        # Direct prediction from t to r
        u_tr = model(x_t, t, r, enrollment=enrollment)

        # 정답 궤적을 따라 s 까지 한 칸 이동한 뒤, s -> r 속도를 모델 스스로 예측(부트스트랩).
        # Advance to s along the true trajectory, then predict from s to r
        z_s = x_t + (s_ - t_) * v
        u_sr = model(z_s, s, r, enrollment=enrollment)

        # 목표 속도: 정답 속도 v 와 모델 자기 예측 u_sr 을 alpha 비율로 섞음.
        # alpha=1 이면 순수 정답(궤적), alpha→0 이면 자기 예측(한방 도착 연습) 쪽으로.
        # Target: mix true velocity with the bootstrapped prediction
        target_u = alpha * v + (1 - alpha) * u_sr

        # Compute loss
        error = u_tr - stopgrad(target_u)
        loss = adaptive_l2_loss(error, gamma=self.gamma, alpha=alpha)
        mse_val = (error ** 2).mean()

        return loss, mse_val

    def loss_alpha_flow_alpha1(
        self,
        model,
        source,
        background,
        enrollment,
        alpha,
        mixing_ratio=None,
    ):
        """
        Special case of Alpha-Flow when alpha=1 (reduces to trajectory flow matching).
        Since s = r, the bootstrap term drops out and a single forward pass
        regresses the model onto the ground-truth velocity v = source - background.

        Args:
            model: The neural network model
            source: Clean source (t=1)
            background: Background/noise (t=0)
            enrollment: Enrollment/reference signals
            alpha: Consistency step ratio (should be 1.0)
        """
        batch_size = source.shape[0]
        device = source.device

        # Sample t and r from logistic distribution
        mu, sigma = -0.4, 1.0
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))  # sigmoid

        # Ensure r >= t
        t_np = np.minimum(samples[:, 0], samples[:, 1])
        r_np = np.maximum(samples[:, 0], samples[:, 1])

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)

        # 같은 affine mapping을 t와 r에 적용하므로
        # 기존의 t <= r 순서가 그대로 유지된다.
        tr = torch.stack(
            [t, r],
            dim=1,
        )

        tr = self._rescale_time_samples(
            tr,
            mixing_ratio=mixing_ratio,
        )

        t = tr[:, 0]
        r = tr[:, 1]

        # Reshape time variables
        if self.data_dim == '1d':
            t_ = rearrange(t, "b -> b 1 1")
        else:
            t_ = rearrange(t, "b -> b 1 1 1")

        # Point at time t
        x_t = (1 - t_) * background + t_ * source

        # Direct prediction from t to r, regressed onto the true velocity
        u_tr = model(x_t, t, r, enrollment=enrollment)
        v = source - background

        # Compute loss
        error = u_tr - v
        loss = adaptive_l2_loss(error, gamma=self.gamma, alpha=alpha)
        mse_val = (error ** 2).mean()

        return loss, mse_val
    
    def loss(
        self,
        model,
        source,
        background,
        enrollment,
        iteration=None,
        mixing_ratio=None,
    ):
        """
        Combined loss function with alpha scheduling for curriculum learning.
        
        Training phases:
        1. Trajectory flow matching pretraining (alpha=1)
        2. Alpha-Flow transition (alpha ∈ (0,1))
        3. Consistency training fine-tuning (alpha→0)
        
        Convention:
        - background: at t=0 (starting point, noisy mixture)
        - source: at t=1 (target, clean)
        - r ≥ t always (r is the target timestep)
        
        Args:
            model: The neural network model
            source: Clean source (t=1) - unnormalized spectrograms (B, C, T)
            background: Background/noise (t=0) - unnormalized spectrograms (B, C, T)
            enrollment: Enrollment/reference signals (B, C, T_enroll)
            iteration: Current training iteration (optional, uses self.current_iteration if None)
        
        Returns:
            loss: Scalar loss value
            mse_val: MSE value for logging
            alpha: Current alpha value (for logging)
            raw_alpha: Raw alpha value before clipping (for logging)
        """
        # Update iteration counter
        if iteration is not None:
            self.current_iteration = iteration
        
        # 현재 학습 진행도에 맞는 alpha 값을 스케줄에서 가져옵니다.
        # Get current alpha value based on schedule
        alpha, raw_alpha = self.get_alpha(return_raw=True)

        # alpha 값(=학습 위상)에 따라 어떤 손실을 쓸지 고릅니다. 매 스텝 flow_ratio 확률로
        # 직선 흐름(loss_rectified_flow) vs 알파 흐름(loss_alpha_flow) 둘 중 하나를 섞어 씁니다.
        # Select loss function based on alpha and flow_ratio
        if alpha >= (1.0 - self.alpha_min):
            # 위상 1: 정직한 궤적 따라가기 (alpha = 1). 학습 초반 기초 다지기.
            # Phase 1: Trajectory flow matching (alpha = 1)
            if torch.rand(1).item() < self.flow_ratio:
                loss, mse_val = self.loss_rectified_flow(model, source, background, enrollment, alpha=1.0, mixing_ratio=mixing_ratio)
            else:
                loss, mse_val = self.loss_alpha_flow_alpha1(model, source, background, enrollment, alpha=1.0, mixing_ratio=mixing_ratio)
        elif alpha <= self.alpha_min:
            # 위상 3: 한방 도착 연습 (alpha ≈ 0). 학습 후반 1-step 추론 정밀도 끌어올리기.
            # Phase 3: Consistency training (alpha ≈ 0)
            if torch.rand(1).item() < self.flow_ratio:
                loss, mse_val = self.loss_rectified_flow(model, source, background, enrollment, alpha=self.alpha_min, mixing_ratio=mixing_ratio)
            else:
                loss, mse_val = self.loss_alpha_flow(model, source, background, enrollment, alpha=self.alpha_min, mixing_ratio=mixing_ratio)
        else:
            # 위상 2: 둘 사이 전환 구간 (0 < alpha < 1). 궤적->한방으로 서서히 넘어가는 중.
            # Phase 2: Alpha-Flow transition (0 < alpha < 1)
            if torch.rand(1).item() < self.flow_ratio:
                loss, mse_val = self.loss_rectified_flow(model, source, background, enrollment, alpha=alpha, mixing_ratio=mixing_ratio)
            else:
                loss, mse_val = self.loss_alpha_flow(model, source, background, enrollment, alpha=alpha, mixing_ratio=mixing_ratio)
        
        return loss, mse_val, alpha, raw_alpha

