"""
Training script for MeanFlowTSE with alpha-flow scheduling and periodic checkpointing.

============================================================================
[이 파일이 하는 일 — 쉽게 설명]
----------------------------------------------------------------------------
이 파일은 "flow(흐름) 모델"을 학습시키는 메인 스크립트입니다.
우리 목표는 TSE(Target Speaker Extraction, 목표 화자 추출): 여러 사람 목소리가
섞인 소리에서 "내가 원하는 한 사람" 목소리만 깨끗하게 뽑아내는 것입니다.

핵심 구성품 4가지:
  1) UDiT  : flow(흐름)를 따라 "섞인 소리 -> 깨끗한 소리"로 한 발짝에 밀어주는 본체 모델.
  2) PN-encoder(Positive/Negative encoder, 등록 화자 인코더) : "원하는 사람(pos)"과
     "원하지 않는 사람(neg)" 목소리를 비교해 화자 정보를 뽑는 부분.
     => 학습하지 않고 그대로 씁니다(frozen, 고정). 동봉된 checkpoints/proposed-monaural.pt 사용.
  3) t-predicter(섞임 비율 예측기) : "지금 소리가 얼마나 섞여 있나(0~1)"를 맞추는 작은 모델.
     검증(validation)할 때만 불러와서 출발 지점 t를 정하는 데 씁니다. 역시 frozen.
  4) MR-jitter(Mixing-Ratio jitter, 섞임비율 흔들기) : 출발 t를 일부러 살짝 흔들어
     학습시켜, t-predicter가 조금 틀려도 1-step 추론이 버티게 만드는 강건성(robustness) 기법.

추론/검증은 전부 NFE=1 (한 발짝, 1-step) 으로 통일합니다.

[재학습 순서]
  보통은 동봉된 체크포인트(flow_best.ckpt 등)로 바로 추론만 하면 됩니다(아래 파일은 안 돌려도 됨).
  꼭 다시 학습한다면: (1) t-predicter 먼저 -> (2) 이 flow 학습(MR-jitter 켜고).

----------------------------------------------------------------------------
[✏️ 바꿔도 되는 곳]
  - YAML config 파일의 값들 (batch size, accumulation, num_gpus, precision, 학습률 등).
    48GB(A6000) 기준 예시 — global batch 256 유지:
      1GPU: batch16 / accum16 / num_gpus1
      2GPU: batch16 / accum8  / num_gpus2
      4GPU: batch16 / accum4  / num_gpus4   (num_workers 4~6, precision bf16-mixed)
  - mr_jitter.sigma (흔들기 세기) 같은 하이퍼파라미터.
  - 로그/체크포인트 저장 경로, 로그 주기 등.

[⚠️ 만지지 마세요]
  - MeanFlowModelWrapper / LightningModule 안의 학습·검증 로직 (loss 계산, NFE=1 적분 등).
  - DDP 설정 (find_unused_parameters / broadcast_buffers) — 멀티 GPU에서 멈춤(deadlock) 방지용.
  - frozen 모듈(PN encoder, t-predicter)을 optimizer/state_dict에서 빼두는 부분.
============================================================================
"""
import speedups  # noqa: F401
import argparse
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    Callback,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from utils.optim import make_optimizer  # vendored AdamW shim (asteroid intentionally not installed)
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR

from meanflow import MeanFlowTSE
from utils import neg_sisdr_loss_wrapper
from utils.losses import (
    target_orthogonal_leakage_loss,
    target_dominance_retention_loss,
)
from data.datasets import get_dataloaders
from conditioning import build_enrollment_conditioner  # the single V1<->V2 enrollment switch
from val_logging import batch_metrics, mel_image, peak_norm  # val metrics (SI-SDR/SI-SDRi/PESQ/eSTOI) + TB audio/mel

from torchmetrics.functional import (
    signal_noise_ratio as _tb_snr,
    scale_invariant_signal_noise_ratio as _tb_si_snr,
)
from models.udit_meanflow.udit_meanflow import UDiT
from utils.transforms import istft_torch
from flow_matching.path import CondOTProbPath
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper


# --- config(설정) 로딩 -------------------------------------------------------
# 실행할 때 --config 로 YAML 설정 파일 경로를 받습니다.
# 예: python train_meanflow.py --config config/config_PNNoisyFlow_v2b_crossattn_jitter.yaml
def parse_args():
    parser = argparse.ArgumentParser(description='Training script for MeanFlowTSE')
    parser.add_argument('--config', default='config/config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    return args


# YAML 파일을 읽어서 파이썬 dict(딕셔너리)로 바꿔줍니다. 거의 모든 설정값이 여기서 나옵니다.
def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# UDiT를 flow_matching 라이브러리의 ODESolver(미분방정식 풀이기)에 끼워 맞추는 어댑터(연결부).
# ODESolver는 (x, t, **extras) 형태로 모델을 부르는데, 우리 UDiT는 (x, t, r, enrollment)가
# 필요해서 이 래퍼가 그 신호 모양을 맞춰줍니다. (지금 코드 경로에서는 거의 참고용)
class MeanFlowModelWrapper(ModelWrapper):
    """
    Wrapper to make UDiT compatible with ODESolver.
    ODESolver expects a model that takes (x, t, **extras).
    UDiT expects (x, t, r, enrollment).
    """
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        
    def forward(self, x, t, **extras):
        """
        Args:
            x: (B, C, T) tensor
            t: (B,) or scalar tensor of current timestep
            **extras: Must contain 'r' and 'enrollment'
        """
        r = extras.get('r', t)  # If r not provided, use t (rectified flow mode)
        enrollment = extras['enrollment']
        return self.model(x, t, r, enrollment)


class MetricPrinterCallback(Callback):
    """Custom callback to print metrics at the end of each epoch."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Print training metrics at the end of each training epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # Collect metrics to print
        train_loss = metrics.get('train_loss', None)
        train_mse = metrics.get('train_mse', None)
        train_alpha = metrics.get('train_alpha_epoch', None)
        lr = None
        
        # Get learning rate
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]['lr']
        
        # Build print string
        print_str = f"Epoch {epoch}"
        if train_loss is not None:
            print_str += f" | Train Loss: {train_loss:.4f}"
        if train_mse is not None:
            print_str += f" | Train MSE: {train_mse:.4f}"
        if train_alpha is not None:
            print_str += f" | Alpha: {train_alpha:.4f}"
        if lr is not None:
            print_str += f" | LR: {lr:.6f}"
        
        print(print_str)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Print validation metrics (NFE sweep) at the end of each validation epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        val_loss = metrics.get('val_loss', None)
        nfe_list = pl_module.config.get('validation', {}).get('nfe_list', [1, 4, 16])
        parts = []
        for nfe in nfe_list:
            v = metrics.get(f'val_loss_nfe{nfe}', None)
            if v is not None:
                parts.append(f"NFE{nfe}={float(v):.4f}")

        if val_loss is not None:
            line = f"Epoch {epoch} | Val Loss(best)={float(val_loss):.4f}"
            if parts:
                line += " | " + " ".join(parts)
            print(line)


# PyTorch Lightning 모듈 — 학습/검증의 모든 "한 스텝" 동작이 여기 모여 있습니다.
class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # 본체 flow 모델(UDiT). 모델 구조 하이퍼파라미터는 전부 config['model']에서 옵니다.
        self.model = UDiT(
            **config['model']
        )
        self.config = config
        self.save_hyperparameters(config)
        self.neg_si_sdr = neg_sisdr_loss_wrapper
        
        # MeanFlowTSE: 실제 loss(손실) 계산을 담당하는 도우미. alpha(알파) 스케줄링으로
        # 학습 진행에 따라 흐름 경로의 세기를 서서히 바꿉니다(예열 -> 본 학습).
        meanflow_config = config.get('meanflow', {})

        self.meanflow = MeanFlowTSE(
            flow_ratio=meanflow_config.get('flow_ratio', 0.50),
            use_enrollment=True,
            data_dim='1d',
            alpha_schedule_start=meanflow_config.get('alpha_schedule_start_epoch', 0),
            alpha_schedule_end=meanflow_config.get('alpha_schedule_end_epoch', None),
            alpha_gamma=meanflow_config.get('alpha_gamma', 25.0),
            alpha_min=meanflow_config.get('alpha_min', 5e-3),
            gamma=config.get('loss', {}).get('gamma', 0.0),
            time_sampling_mode=meanflow_config.get(
                'time_sampling_mode',
                'full',
            ),
            tau_margin=meanflow_config.get(
                'tau_margin',
                0.1,
            ),
        )
        
        # alpha 스케줄을 "에폭"이 아니라 "iteration(반복 스텝)" 단위로 계산하려면 1에폭이
        # 몇 스텝인지 알아야 합니다. 그 값은 학습 시작 시점(on_train_start)에 채웁니다.
        self.steps_per_epoch = None

        # --- enrollment(등록 화자 조건) 만들기 -------------------------------
        # "원하는 화자"가 누군지 모델에 알려주는 조건 정보를 만드는 부분입니다.
        # V1<->V2를 바꾸는 단 하나의 스위치(config['enroll']['provider'])로 결정됩니다.
        # 학습되는 어댑터(adapter)는 이 모듈 안에 들어 있어 자동으로 optimizer/DDP에 포함됩니다.
        # 반대로 frozen(고정) PN encoder(V2b)는 따로 보관해서 optimizer에서 빠집니다(아래 requires_grad 필터).
        # Enrollment conditioning (the single V1<->V2 switch). V1=RawEnrollProvider (no params);
        # V2a=PosNegConcatProvider; V2b=PNEncoderProvider. Trainable adapters live INSIDE this module,
        # so they join self.parameters()/DDP automatically. The frozen PN encoder (V2b only) is held
        # separately and stays OUT of the optimizer via the requires_grad filter (configure_optimizers).
        self.enroll_cond = build_enrollment_conditioner(config)
        # V2b일 때만 frozen PN encoder를 불러옵니다(espnet2 의존성도 이때만 import).
        # 동봉된 가중치 경로: config['paths']['pn_ckpt'] (= checkpoints/proposed-monaural.pt).
        self.pn_encoder = None
        if config.get('enroll', {}).get('provider') == 'v2b_pn_encoder':
            from pn_encoder import load_pn_encoder  # lazy: only V2b pulls in the espnet2/PN stack
            self.pn_encoder = load_pn_encoder(config['paths']['pn_ckpt'], device='cpu')  # Lightning moves it

        # --- frozen t-predicter(섞임비율 예측기) 불러오기 ----------------------
        # 검증할 때 "출발 t"를 예측해 주는 작은 모델입니다(학습 안 함, 고정).
        # config['paths']['t_predicter_ckpt']가 있으면 그 체크포인트(= t_predicter_best.ckpt)를 읽어옵니다.
        # 핵심 트릭: object.__setattr__ 로 넣어서 "정식 서브모듈로 등록되지 않게" 합니다.
        #   -> state_dict / DDP / optimizer 어디에도 안 들어가고, flow 체크포인트에도 안 섞입니다.
        tp_ck = config.get('paths', {}).get('t_predicter_ckpt')
        tp = None
        if tp_ck:
            from models.t_predicter import TPredicterPN
            ck = torch.load(tp_ck, map_location='cpu')
            # 체크포인트 안에 저장된 모델 설정을 꺼내 같은 구조로 만든 뒤, 'model.' 접두어만 떼고 가중치를 채웁니다.
            mcfg = (ck.get('hyper_parameters', {}) or {}).get('model', {'C': 1024})
            tp = TPredicterPN(**mcfg)
            tp.load_state_dict({k[len('model.'):]: v for k, v in ck['state_dict'].items() if k.startswith('model.')},
                               strict=True)
            tp.eval()
            for p in tp.parameters():   # 모든 파라미터 학습 끄기 -> 완전히 고정(frozen)
                p.requires_grad = False
        object.__setattr__(self, '_t_predicter', tp)

    # 학습이 막 시작될 때 1번 호출됩니다. 1에폭이 몇 스텝인지 세고, 에폭 단위로 적힌 alpha 스케줄을
    # iteration(반복 스텝) 단위로 환산해 둡니다. (학습 로직 자체는 건드리지 않음)
    def on_train_start(self):
        """Called when training starts - compute steps per epoch and adjust alpha schedule."""
        if self.steps_per_epoch is None:
            # Get dataloader to compute steps per epoch
            train_dataloader = self.trainer.train_dataloader
            self.steps_per_epoch = len(train_dataloader)
            
            # Convert epoch-based schedule to iteration-based schedule
            alpha_start_epoch = self.config.get('meanflow', {}).get('alpha_schedule_start_epoch', 0)
            alpha_end_epoch = self.config.get('meanflow', {}).get('alpha_schedule_end_epoch', None)
            
            self.meanflow.alpha_schedule_start = alpha_start_epoch * self.steps_per_epoch
            
            if alpha_end_epoch is not None:
                self.meanflow.alpha_schedule_end = alpha_end_epoch * self.steps_per_epoch
            
            print(f"Alpha schedule: iterations {self.meanflow.alpha_schedule_start} to {self.meanflow.alpha_schedule_end}")
            print(f"Steps per epoch: {self.steps_per_epoch}")

    def forward(self, x, t, r, enrollment):
        return self.model(x, t, r, enrollment)

    # --- enrollment(등록 화자 조건) 한 번에 만들기 -----------------------------
    # "원하는 화자가 누구다"라는 조건 정보를 만들어 반환합니다. provider 종류와 무관하게 같은 함수로 처리.
    #   V1  : 등록 음성의 raw STFT를 그대로 사용.
    #   V2b : 먼저 frozen PN encoder를 GPU에서 no_grad + bf16(메모리 절약)로 돌려 화자 임베딩을
    #         batch['spk_emb']에 넣어두고, 그걸 conditioner가 접어 넣습니다.
    # 학습(training_step)과 검증(validation_step)에서 똑같이 불려서, UDiT 입력 모양이 항상 동일합니다.
    def _enrollment(self, batch):
        """Provider-agnostic enrollment prefix [B, 512, T_enroll].

        V1 returns the raw single-utt enroll STFT. V2b first runs the FROZEN PN encoder on-device
        under no_grad + bf16 (mirrors our train.py), stashing batch['spk_emb'], then the conditioner
        folds it. The same call works in training_step and validation_step, so UDiT stays byte-identical.
        """
        if self.pn_encoder is not None:
            self.pn_encoder.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                batch['spk_emb'] = self.pn_encoder.encode(batch['pos_wave'], batch['neg_wave']).float()
        return self.enroll_cond(batch)

    # === MULTI_ENDPOINT_AUX_HELPERS ===
    def _endpoint_aux_waveforms(self, batch, enrollment, aux_batch_size):
        """NFE=1 endpoint와 target/nuisance waveform을 반환한다."""
        batch_size = batch['mixture_spec'].size(0)
        aux_batch_size = max(
            1,
            min(int(aux_batch_size), batch_size),
        )

        mixture = batch['mixture_spec'][:aux_batch_size]
        enrollment_aux = enrollment[:aux_batch_size]
        nuisance_sources = batch['nuisance_sources'][:aux_batch_size]

        # mixture가 background→source 경로에서 위치한 실제 시간
        m = (
            batch['mixing_ratio'][:aux_batch_size]
            .reshape(aux_batch_size)
            .float()
            .clamp(1e-3, 1.0 - 1e-3)
        )

        r = torch.ones(
            aux_batch_size,
            device=mixture.device,
            dtype=m.dtype,
        )

        velocity = self.model(
            mixture,
            m,
            r,
            enrollment=enrollment_aux,
        )

        dt = (
            (1.0 - m)
            .to(dtype=mixture.dtype)
            .view(aux_batch_size, 1, 1)
        )

        endpoint_spec = mixture + dt * velocity

        endpoint_wave = istft_torch(
            endpoint_spec.float(),
            n_fft=self.config['dataset']['n_fft'],
            hop_length=self.config['dataset']['hop_length'],
            win_length=self.config['dataset']['win_length'],
            length=batch['source'][:aux_batch_size].shape[-1],
        )

        target_wave = batch['source'][:aux_batch_size]

        return endpoint_wave, target_wave, nuisance_sources

    def _multi_resolution_stft_loss(
        self,
        estimate,
        target,
        resolutions,
        eps=1.0e-7,
    ):
        """Spectral convergence + log-magnitude MR-STFT loss."""
        estimate = estimate.float()
        target = target.float()

        total_loss = estimate.new_zeros(())

        if not resolutions:
            raise ValueError("MR-STFT resolutions가 비어 있습니다.")

        for resolution in resolutions:
            if len(resolution) != 3:
                raise ValueError(
                    "각 MR-STFT resolution은 "
                    "[n_fft, hop_length, win_length] 형식이어야 합니다."
                )

            n_fft, hop_length, win_length = map(int, resolution)

            window = torch.hann_window(
                win_length,
                device=estimate.device,
                dtype=torch.float32,
            )

            estimate_spec = torch.stft(
                estimate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )

            target_spec = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )

            estimate_mag = estimate_spec.abs()
            target_mag = target_spec.abs()

            diff_norm = (
                (estimate_mag - target_mag)
                .pow(2)
                .sum(dim=(-2, -1))
                .sqrt()
            )

            target_norm = (
                target_mag
                .pow(2)
                .sum(dim=(-2, -1))
                .sqrt()
            )

            spectral_convergence = (
                diff_norm / (target_norm + eps)
            ).mean()

            log_magnitude = (
                torch.log(estimate_mag + eps)
                - torch.log(target_mag + eps)
            ).abs().mean()

            total_loss = (
                total_loss
                + spectral_convergence
                + log_magnitude
            )

        return total_loss / float(len(resolutions))

    def training_step(self, batch, batch_idx):
        """
        Training step using MeanFlowTSE with alpha scheduling.
        
        Convention:
        - t=0: background/mixture (noisy)
        - t=1: source (clean target)
        
        The model works directly with unnormalized spectrograms.
        """
        # 지금이 전체 학습에서 몇 번째 스텝인지 계산 (alpha 스케줄을 진행도에 맞춰 쓰기 위함).
        current_iteration = self.current_epoch * self.steps_per_epoch + batch_idx

        # 데이터 꺼내기. 흐름의 양 끝을 이렇게 약속합니다:
        #   t=0 : background(섞인 소리, 노이즈) / t=1 : source(깨끗한 목표 화자)
        background = batch['background_rescaled_spec']  # t=0 (noisy mixture)
        source = batch['source_rescaled_spec']  # t=1 (clean target)
        enrollment = self._enrollment(batch)  # V1 raw / V2a posneg / V2b PN-encoder prefix
        mixing_ratio = batch['mixing_ratio']
        batch_size = source.size(0)

        if self.meanflow.time_sampling_mode == 'tau_truncated':
            tau_for_log = (
                mixing_ratio
                .float()
                .reshape(batch_size, -1)
                .mean(dim=1)
            )

            time_lower = (
                tau_for_log - self.meanflow.tau_margin
            ).clamp(
                min=0.0,
                max=1.0,
            )

            self.log(
                'train_tau_mean',
                tau_for_log.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

            self.log(
                'train_time_lower_mean',
                time_lower.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

            # true tau가 현재 background/source 경로와 일치하는지 확인.
            # 이 값은 0에 매우 가까워야 한다.
            mixture_spec = batch.get('mixture_spec')

            if (
                mixture_spec is not None
                and mixture_spec.shape == source.shape
            ):
                with torch.no_grad():
                    tau_view = tau_for_log.view(
                        batch_size,
                        *([1] * (source.ndim - 1)),
                    )

                    reconstructed_mixture = (
                        (1.0 - tau_view) * background
                        + tau_view * source
                    )

                    tau_path_mae = (
                        reconstructed_mixture
                        - mixture_spec
                    ).abs().mean()

                self.log(
                    'train_tau_path_mae',
                    tau_path_mae,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

        # === MR-jitter(섞임비율 흔들기) 분기 ====================================
        # config의 mr_jitter.enabled가 켜져 있으면 이 강건성(robustness) 학습으로 "완전히 대체"됩니다.
        # 아이디어: 실제 섞인 소리에서 깨끗한 소리로 한 발짝(1-step) 점프하는 걸 배우되,
        #   출발 시각 t를 가우시안 노이즈로 sigma만큼 일부러 흔들어 학습합니다.
        # 효과: 추론 때 t-predicter가 t를 살짝 틀려도 결과가 무너지지 않게 됩니다.
        # (아래 일반 loss 경로와 섞지 않고, 켜지면 이 경로만 사용)
        mj = self.config.get('mr_jitter', {}) or {}
        mr_jitter_enabled = bool(mj.get('enabled', False))

        # MR-jitter 또는 일반 MeanFlow Loss를 Base Flow Loss로 계산한다.
        # MR-jitter가 켜져 있어도 여기서 반환하지 않고,
        # 아래 Endpoint Auxiliary Loss 계산까지 진행한다.
        if mr_jitter_enabled:
            flow_loss, mse_val, mr_stats = (
                self.meanflow.loss_mr_jitter(
                    model=self.model,
                    source=source,
                    mixture=batch['mixture_spec'],
                    mixing_ratio=batch['mixing_ratio'],
                    enrollment=enrollment,
                    sigma=float(
                        mj.get('sigma', 0.25)
                    ),
                    return_stats=True,
                )
            )

            for metric_name, metric_value in (
                mr_stats.items()
            ):
                self.log(
                    f'train_{metric_name}',
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            # 일반 MeanFlow 경로에서만 계산되는 로깅 값이다.
            alpha = flow_loss.detach().new_zeros(())
            raw_alpha = flow_loss.detach().new_zeros(())

        else:
            flow_loss, mse_val, alpha, raw_alpha = self.meanflow.loss(
                model=self.model,
                source=source,
                background=background,
                enrollment=enrollment,
                iteration=current_iteration,
                mixing_ratio=mixing_ratio,
            )

        # Endpoint SI-SDR/MR-STFT Loss가 아래에서 여기에 추가된다.
        loss = flow_loss

        # === MULTI_ENDPOINT_AUX ===
        # 공통 endpoint waveform을 한 번만 생성한 뒤,
        # config에 따라 SI-SDR 및 MR-STFT를 선택적으로 적용한다.
        aux_cfg = (
            (self.config.get('loss', {}) or {})
            .get('endpoint_aux', {})
            or {}
        )

        sisdr_loss = flow_loss.detach().new_zeros(())
        mrstft_loss = flow_loss.detach().new_zeros(())
        leak_loss = flow_loss.detach().new_zeros(())
        tdrl_loss = flow_loss.detach().new_zeros(())
        tdrl_dom_loss = flow_loss.detach().new_zeros(())
        tdrl_ret_loss = flow_loss.detach().new_zeros(())
        tdrl_switch_ratio = flow_loss.detach().new_zeros(())
        tdrl_retention_violation_ratio = flow_loss.detach().new_zeros(())
        tdrl_target_gain = flow_loss.detach().new_zeros(())
        tdrl_speech_gain = flow_loss.detach().new_zeros(())
        tdrl_valid_windows = flow_loss.detach().new_zeros(())

        # TLIS Leakage Loss 진단값
        leak_gate_ratio = flow_loss.detach().new_zeros(())
        leak_orthogonal_ratio = flow_loss.detach().new_zeros(())
        leak_valid_windows = flow_loss.detach().new_zeros(())
        leak_target_activity_ratio = flow_loss.detach().new_zeros(())
        leak_nuisance_activity_ratio = flow_loss.detach().new_zeros(())

        sisdr_weight = float(
            aux_cfg.get('sisdr_weight', 0.0)
        )
        mrstft_weight = float(
            aux_cfg.get('mrstft_weight', 0.0)
        )
        leak_weight = float(
            aux_cfg.get('leak_weight', 0.0)
        )
        tdrl_weight = float(
            aux_cfg.get('tdrl_weight', 0.0)
        )

        effective_sisdr_weight = 0.0
        effective_mrstft_weight = 0.0
        effective_leak_weight = 0.0
        effective_tdrl_weight = 0.0

        aux_enabled = (
            aux_cfg.get('enabled', False)
            and (
                sisdr_weight != 0.0
                or mrstft_weight != 0.0
                or leak_weight != 0.0
                or tdrl_weight != 0.0
            )
        )

        if aux_enabled:
            endpoint_wave, target_wave, nuisance_sources = (
                self._endpoint_aux_waveforms(
                    batch=batch,
                    enrollment=enrollment,
                    aux_batch_size=aux_cfg.get(
                        'aux_batch_size',
                        1,
                    ),
                )
            )

            warmup_epochs = max(
                0,
                int(aux_cfg.get('warmup_epochs', 5)),
            )

            if warmup_epochs > 0:
                weight_ratio = min(
                    1.0,
                    (float(self.current_epoch) + 1.0)
                    / float(warmup_epochs),
                )
            else:
                weight_ratio = 1.0

            # Leakage Loss는 필요하면 일정 epoch 이후부터 천천히 켤 수 있다.
            leak_start_epoch = max(
                0,
                int(aux_cfg.get('leak_start_epoch', 0)),
            )
            leak_warmup_epochs = max(
                0,
                int(
                    aux_cfg.get(
                        'leak_warmup_epochs',
                        aux_cfg.get('warmup_epochs', 5),
                    )
                ),
            )

            if self.current_epoch < leak_start_epoch:
                leak_weight_ratio = 0.0
            elif leak_warmup_epochs > 0:
                leak_weight_ratio = min(
                    1.0,
                    (
                        float(self.current_epoch)
                        - float(leak_start_epoch)
                        + 1.0
                    )
                    / float(leak_warmup_epochs),
                )
            else:
                leak_weight_ratio = 1.0

            tdrl_start_epoch = max(
                0,
                int(aux_cfg.get('tdrl_start_epoch', 0)),
            )
            tdrl_warmup_epochs = max(
                0,
                int(aux_cfg.get('tdrl_warmup_epochs', 3)),
            )
            if self.current_epoch < tdrl_start_epoch:
                tdrl_weight_ratio = 0.0
            elif tdrl_warmup_epochs > 0:
                tdrl_weight_ratio = min(
                    1.0,
                    (
                        float(self.current_epoch)
                        - float(tdrl_start_epoch)
                        + 1.0
                    ) / float(tdrl_warmup_epochs),
                )
            else:
                tdrl_weight_ratio = 1.0

            if sisdr_weight != 0.0:
                sisdr_loss = self.neg_si_sdr(
                    endpoint_wave,
                    target_wave,
                )

                sisdr_loss = torch.nan_to_num(
                    sisdr_loss,
                    nan=0.0,
                    posinf=50.0,
                    neginf=-50.0,
                ).clamp(min=-50.0, max=50.0)

                effective_sisdr_weight = (
                    sisdr_weight * weight_ratio
                )

                loss = (
                    loss
                    + effective_sisdr_weight
                    * sisdr_loss
                )

            if mrstft_weight != 0.0:
                resolutions = aux_cfg.get(
                    'mrstft_resolutions',
                    [
                        [256, 64, 256],
                        [512, 128, 512],
                        [1024, 256, 1024],
                    ],
                )

                mrstft_loss = (
                    self._multi_resolution_stft_loss(
                        endpoint_wave,
                        target_wave,
                        resolutions,
                    )
                )

                mrstft_loss = torch.nan_to_num(
                    mrstft_loss,
                    nan=0.0,
                    posinf=50.0,
                    neginf=0.0,
                ).clamp(min=0.0, max=50.0)

                effective_mrstft_weight = (
                    mrstft_weight * weight_ratio
                )

                loss = (
                    loss
                    + effective_mrstft_weight
                    * mrstft_loss
                )

            if leak_weight != 0.0:
                leak_loss, leak_stats = (
                    target_orthogonal_leakage_loss(
                        est_targets=endpoint_wave,
                        targets=target_wave,
                        interferers=nuisance_sources,
                        window_size=int(
                            aux_cfg.get(
                                'leak_window_size',
                                2048,
                            )
                        ),
                        hop_size=int(
                            aux_cfg.get(
                                'leak_hop_size',
                                1024,
                            )
                        ),
                        gate_threshold=float(
                            aux_cfg.get(
                                'leak_gate_threshold',
                                0.1,
                            )
                        ),
                        eps=float(
                            aux_cfg.get(
                                'leak_eps',
                                1.0e-8,
                            )
                        ),
                        normalize_residual_energy=bool(
                            aux_cfg.get(
                                'normalize_residual_energy',
                                True,
                            )
                        ),
                        activity_gate=bool(
                            aux_cfg.get(
                                'activity_gate',
                                True,
                            )
                        ),
                        target_activity_threshold=float(
                            aux_cfg.get(
                                'target_activity_threshold',
                                0.01,
                            )
                        ),
                        nuisance_activity_threshold=float(
                            aux_cfg.get(
                                'nuisance_activity_threshold',
                                0.01,
                            )
                        ),
                        return_stats=True,
                    )
                )

                leak_loss = torch.nan_to_num(
                    leak_loss,
                    nan=0.0,
                    posinf=50.0,
                    neginf=0.0,
                ).clamp(min=0.0, max=50.0)

                leak_gate_ratio = leak_stats['gate_ratio']
                leak_orthogonal_ratio = (
                    leak_stats['orthogonal_ratio']
                )
                leak_valid_windows = (
                    leak_stats['valid_windows']
                )
                leak_target_activity_ratio = (
                    leak_stats['target_activity_ratio']
                )
                leak_nuisance_activity_ratio = (
                    leak_stats['nuisance_activity_ratio']
                )

                effective_leak_weight = (
                    leak_weight * leak_weight_ratio
                )

                loss = (
                    loss
                    + effective_leak_weight
                    * leak_loss
                )

            if tdrl_weight != 0.0:
                tdrl_loss, tdrl_stats = target_dominance_retention_loss(
                    est_targets=endpoint_wave,
                    targets=target_wave,
                    interferers=nuisance_sources,
                    window_size=int(aux_cfg.get('tdrl_window_size', 2048)),
                    hop_size=int(aux_cfg.get('tdrl_hop_size', 1024)),
                    speech_interferer_count=int(
                        aux_cfg.get(
                            'tdrl_speech_interferer_count',
                            max(int(self.config.get('dataset', {}).get('source_num', 3)) - 1, 0),
                        )
                    ),
                    ridge_scale=float(aux_cfg.get('tdrl_ridge_scale', 1.0e-6)),
                    target_activity_threshold=float(aux_cfg.get('tdrl_target_activity_threshold', 0.01)),
                    speech_activity_threshold=float(aux_cfg.get('tdrl_speech_activity_threshold', 0.01)),
                    dominance_margin=float(aux_cfg.get('tdrl_dominance_margin', 0.0)),
                    retention_weight=float(aux_cfg.get('tdrl_retention_weight', 1.0)),
                    retention_floor=float(aux_cfg.get('tdrl_retention_floor', 0.5)),
                    reference_quantile=float(aux_cfg.get('tdrl_reference_quantile', 0.75)),
                    gain_floor=float(aux_cfg.get('tdrl_gain_floor', 1.0e-3)),
                    eps=float(aux_cfg.get('tdrl_eps', 1.0e-10)),
                    return_stats=True,
                )

                tdrl_loss = torch.nan_to_num(
                    tdrl_loss, nan=0.0, posinf=50.0, neginf=0.0
                ).clamp(min=0.0, max=50.0)
                tdrl_dom_loss = tdrl_stats['dominance_loss']
                tdrl_ret_loss = tdrl_stats['retention_loss']
                tdrl_switch_ratio = tdrl_stats['switch_ratio']
                tdrl_retention_violation_ratio = tdrl_stats['retention_violation_ratio']
                tdrl_target_gain = tdrl_stats['target_gain_mean']
                tdrl_speech_gain = tdrl_stats['speech_gain_mean']
                tdrl_valid_windows = tdrl_stats['valid_windows']

                effective_tdrl_weight = tdrl_weight * tdrl_weight_ratio
                loss = loss + effective_tdrl_weight * tdrl_loss

        self.log(
            'train_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_flow_loss',
            flow_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_sisdr_loss',
            sisdr_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_mrstft_loss',
            mrstft_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_leak_loss',
            leak_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_leak_gate_ratio',
            leak_gate_ratio,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_leak_orthogonal_ratio',
            leak_orthogonal_ratio,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_leak_valid_windows',
            leak_valid_windows,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_leak_target_activity_ratio',
            leak_target_activity_ratio,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_leak_nuisance_activity_ratio',
            leak_nuisance_activity_ratio,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        for _name, _value in (
            ('train_endpoint_tdrl_loss', tdrl_loss),
            ('train_endpoint_tdrl_dominance_loss', tdrl_dom_loss),
            ('train_endpoint_tdrl_retention_loss', tdrl_ret_loss),
            ('train_endpoint_tdrl_switch_ratio', tdrl_switch_ratio),
            ('train_endpoint_tdrl_retention_violation_ratio', tdrl_retention_violation_ratio),
            ('train_endpoint_tdrl_target_gain', tdrl_target_gain),
            ('train_endpoint_tdrl_speech_gain', tdrl_speech_gain),
            ('train_endpoint_tdrl_valid_windows', tdrl_valid_windows),
        ):
            self.log(
                _name, _value,
                on_step=False, on_epoch=True,
                prog_bar=False, sync_dist=True,
                batch_size=batch_size,
            )

        self.log(
            'train_endpoint_tdrl_weight',
            loss.detach().new_tensor(effective_tdrl_weight),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_sisdr_weight',
            loss.detach().new_tensor(
                effective_sisdr_weight
            ),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_mrstft_weight',
            loss.detach().new_tensor(
                effective_mrstft_weight
            ),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_endpoint_leak_weight',
            loss.detach().new_tensor(
                effective_leak_weight
            ),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_mse',
            mse_val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_alpha',
            alpha,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            'train_alpha_raw',
            raw_alpha,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss


    def on_fit_start(self):
        """
        Fail-fast guard:
        config가 4-GPU DDP를 요구하는데 실제 Trainer world_size가 4가 아니면
        장기 학습을 시작하지 않고 즉시 중단한다.
        """
        ddp_cfg = self.config.get("ddp", {}) or {}
        use_ddp = bool(ddp_cfg.get("use_ddp", False))

        expected_world_size = (
            int(ddp_cfg.get("num_gpus", 1))
            if use_ddp else 1
        )

        actual_world_size = int(
            getattr(self.trainer, "world_size", 1)
        )
        global_rank = int(
            getattr(self.trainer, "global_rank", 0)
        )
        local_rank = int(
            getattr(self.trainer, "local_rank", 0)
        )

        micro_batch = int(
            self.config["train"]["batch_size"]
        )
        accum = int(
            self.config["train"]["accumulation_steps"]
        )
        actual_global_batch = (
            micro_batch * accum * actual_world_size
        )

        print(
            f"[DDP_GUARD] "
            f"expected_world_size={expected_world_size} "
            f"actual_world_size={actual_world_size} "
            f"global_rank={global_rank} "
            f"local_rank={local_rank} "
            f"micro_batch={micro_batch} "
            f"accum={accum} "
            f"actual_global_batch={actual_global_batch}",
            flush=True,
        )

        if actual_world_size != expected_world_size:
            raise RuntimeError(
                "[DDP_GUARD] WORLD SIZE MISMATCH: "
                f"config expects {expected_world_size} GPU processes "
                f"but Trainer has {actual_world_size}. "
                "ABORTING before training."
            )

    def on_after_backward(self):
        # DDP_VERIFY=1: prove gradients are genuinely all-reduced (synchronized) across ranks.
        # Each rank sees DIFFERENT data (DistributedSampler) -> different LOCAL grads; only DDP's
        # gradient all-reduce makes them identical. So if every rank's grad-checksum matches, the
        # all-reduce ran (real multi-GPU training). broadcast_buffers=False does NOT touch this path.
        import os
        if not os.environ.get('DDP_VERIFY'):
            return
        ws = getattr(self.trainer, 'world_size', 1)
        if ws <= 1 or self.global_step > 3:
            return
        import torch.distributed as dist
        s = torch.zeros(1, device=self.device, dtype=torch.float64)
        n = 0
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                s += p.grad.detach().double().abs().sum()
                n += 1
        gathered = [torch.zeros_like(s) for _ in range(ws)]
        dist.all_gather(gathered, s)
        vals = [g.item() for g in gathered]
        spread = max(vals) - min(vals)
        rel = spread / (abs(vals[0]) + 1e-12)
        if getattr(self.trainer, 'is_global_zero', True):
            verdict = 'ALL-REDUCED ✓ (real multi-GPU sync)' if rel < 1e-9 else 'NOT SYNCED ✗'
            print(f"[DDP_VERIFY step{self.global_step}] grad-|sum| per rank={vals} "
                  f"rel-spread={rel:.2e} over {n} trainable grads -> {verdict}", flush=True)

    # --- 검증(validation) 한 스텝 -------------------------------------------
    # 실제로 소리를 복원해 보고 음질 점수(SI-SDR 등)를 매깁니다. 추론은 NFE=1(한 발짝)이 핵심.
    #   t=0(섞인 소리)에서 출발 -> t=1(깨끗한 소리)로 한 발짝 점프하는 게 우리가 쓰는 방식.
    def validation_step(self, batch, batch_idx):
        """
        Validation step - generate samples and compute SI-SDR.
        Uses ODESolver for inference.

        Convention:
        - t=0: background/mixture (noisy) - starting point
        - t=1: source (clean) - target
        - mixing_ratio: actual ratio used by LibriMix to create the mixture
        """

        enrollment = self._enrollment(batch)  # V1 raw / V2a posneg / V2b PN-encoder prefix
        mixture = batch['mixture_spec']  # The actual mixture spectrogram
        mixing_ratio = batch['mixing_ratio']  # The actual ratio used by LibriMix
        if mixing_ratio.ndim > 1:
            mixing_ratio = mixing_ratio.squeeze()  # Remove extra dimensions if present

        batch_size = mixture.size(0)
        # --- 출발 지점 t(= 섞임비율 m) 정하기 --------------------------------
        # 각 샘플마다 "지금 얼마나 섞여 있나(m)"를 정해서, 거기서부터 t=1까지 점프합니다.
        #   t-predicter가 있으면: frozen t-predicter가 (섞인 파형 + pos/neg PN 임베딩)으로 m을 "예측".
        #     -> 이게 실제 추론 상황과 같은, 우리가 진짜로 좋아져야 할 숫자입니다(정답 m 아님!).
        #   t-predicter가 없으면: 정답 섞임비율(oracle m)로 대체(참고용 상한 느낌).
        #   clamp(1e-3, 1-1e-3): m이 정확히 0이나 1이 되어 계산이 깨지는 걸 막는 안전장치.
        if getattr(self, '_t_predicter', None) is not None:
            tp = self._t_predicter
            tp.to(self.device).eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                enroll_emb = self.pn_encoder.encode(batch['pos_wave'], batch['neg_wave']).float()
                m = tp(batch['mixture'], enroll_emb).reshape(batch_size).float().clamp(1e-3, 1.0 - 1e-3)
        else:
            m = mixing_ratio.reshape(batch_size).clamp(1e-3, 1.0 - 1e-3)   # per-sample oracle fallback
        ones = torch.ones(batch_size, device=mixture.device)
        # --- NFE 스윕(여러 발짝 수로 비교) ----------------------------------
        # 각 NFE는 t=m에서 t=1까지 정확히 nfe번의 오일러(Euler) 적분 스텝으로 도달합니다.
        # 우리는 NFE=1(한 발짝)을 가장 중요하게 보지만, 비교용으로 4/16도 같이 재 봅니다.
        nfe_list = self.config.get('validation', {}).get('nfe_list', [1, 4, 16])

        per_nfe_loss = {}
        first_source_hat = None
        with torch.no_grad():
            for nfe in nfe_list:
                x = mixture.clone()
                dt = (1.0 - m) / nfe                                   # 샘플별 한 스텝 크기 [B]
                # nfe번 반복하며 x를 t=1 쪽으로 밀어줍니다. NFE=1이면 이 반복이 딱 한 번 = 한 발짝 점프.
                for k in range(nfe):
                    t_cur = (m + k * dt).clamp(0.0, 1.0)
                    v = self.model(x, t_cur, ones, enrollment)         # r=1.0 (도착 목표 시각)
                    x = x + dt.view(batch_size, 1, 1) * v              # x_new = x + (스텝크기) * (속도 v)
                # 스펙트로그램 x를 다시 파형(waveform)으로 되돌립니다(역 STFT). 그래야 점수를 잴 수 있음.
                source_hat = istft_torch(
                    x.float(),
                    n_fft=self.config['dataset']['n_fft'],
                    hop_length=self.config['dataset']['hop_length'],
                    win_length=self.config['dataset']['win_length'],
                    length=batch['source'].shape[-1],
                )
                # neg SI-SDR: checkpoint 기준은 기존과 동일하게 유지
                loss_n = self.neg_si_sdr(
                    source_hat,
                    batch['source'],
                )

                # --------------------------------------------------
                # Paper-compatible evaluation waveform
                #
                # Official PN evaluator evaluates both +est and -est
                # and selects the polarity with lower target MSE.
                # No amplitude rescaling is applied.
                # --------------------------------------------------
                est_metric = source_hat.float().reshape(batch_size, -1)
                tgt_metric = batch['source'].float().reshape(batch_size, -1)
                mix_metric = batch['mixture'].float().reshape(batch_size, -1)

                mse_pos = (
                    (est_metric - tgt_metric)
                    .square()
                    .sum(dim=-1)
                )

                mse_neg = (
                    (-est_metric - tgt_metric)
                    .square()
                    .sum(dim=-1)
                )

                flip = mse_neg < mse_pos

                est_metric = torch.where(
                    flip.unsqueeze(-1),
                    -est_metric,
                    est_metric,
                )

                # Existing metrics
                metrics_n = batch_metrics(
                    est_metric,
                    tgt_metric,
                    mix_metric,
                )

                # PN paper metrics
                out_snr = _tb_snr(
                    est_metric,
                    tgt_metric,
                ).mean()

                input_snr = _tb_snr(
                    mix_metric,
                    tgt_metric,
                ).mean()

                out_si_snr = _tb_si_snr(
                    est_metric,
                    tgt_metric,
                ).mean()

                input_si_snr = _tb_si_snr(
                    mix_metric,
                    tgt_metric,
                ).mean()

                metrics_n['snr'] = out_snr
                metrics_n['input_snr'] = input_snr
                metrics_n['snri'] = out_snr - input_snr

                metrics_n['si_snr'] = out_si_snr
                metrics_n['input_si_snr'] = input_si_snr
                metrics_n['si_snri'] = (
                    out_si_snr - input_si_snr
                )
                per_nfe_loss[nfe] = loss_n
                self.log(f'val_loss_nfe{nfe}', loss_n, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                # Core waveform metrics.
                # SNRi / SI-SNRi follow PN-paper definitions.
                for _k in (
                    'snr',
                    'input_snr',
                    'snri',
                    'si_snr',
                    'input_si_snr',
                    'si_snri',
                    'si_sdr',
                    'si_sdri',
                ):
                    self.log(
                        f'val_{_k}_nfe{nfe}',
                        metrics_n[_k],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )
                # pesq/stoi can be NaN on some ranks -> sync_dist=False avoids the rank-divergent all-reduce -> no DDP val hang
                for _k in ('pesq', 'stoi'):
                    if torch.isfinite(metrics_n[_k]):
                        self.log(f'val_{_k}_nfe{nfe}', metrics_n[_k], on_step=False, on_epoch=True, sync_dist=False, batch_size=batch_size)
                if first_source_hat is None:
                    first_source_hat = source_hat

        # 체크포인트 선택 기준이 되는 'val_loss'는 NFE=1 점수로 잡습니다.
        # 이유: MR-jitter는 "한 발짝 점프"만 직접 학습하므로, 우리가 실제로 쓰는 1-step 성능을 추적해야 함.
        # (혹시 nfe_list에 1이 없으면 스윕 중 best로 대체)
        val_loss = per_nfe_loss.get(1, min(per_nfe_loss.values()))
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)

        # Validation media is a rank-0-only artifact.  Scalar metrics above
        # remain distributed, but only the global-zero process should copy
        # waveforms to CPU and hand them to TensorBoard at epoch end.
        if (
            batch_idx == 0
            and first_source_hat is not None
            and getattr(self.trainer, 'is_global_zero', True)
        ):  # stash first-batch samples (first NFE) for TB audio/mel
            k = min(3, first_source_hat.shape[0])
            self._val_samples = {
                'est': first_source_hat[:k].detach().cpu(),
                'target': batch['source'][:k].detach().cpu(),
                'mixture': batch['mixture'][:k].detach().cpu(),
            }
            if 'pos_wave' in batch:   # V2b: keep enroll waves to also run the PN-Enroll baseline for comparison
                self._val_samples['pos_wave'] = batch['pos_wave'][:k].detach().cpu()
                self._val_samples['neg_wave'] = batch['neg_wave'][:k].detach().cpu()

        return val_loss

    def on_validation_epoch_end(self):
        """rank0: log the PN-Enroll baseline as flat reference lines + a few audio/mel-spec samples to TB."""
        if not getattr(self.trainer, 'is_global_zero', True):
            self._val_samples = None
            return
        writer = getattr(self.logger, 'experiment', None)
        step = self.global_step
        baseline = self.config.get('baseline', {}) or {}
        if writer is not None:
            # PN-Enroll baseline as flat reference lines (measured offline by baseline_pnenroll.py)
            for _k in ('si_sdr', 'si_sdri', 'pesq', 'stoi'):
                if baseline.get(_k) is not None:
                    writer.add_scalar(f'baseline/{_k}', float(baseline[_k]), step)
            # audio + mel-spectrogram for a few fixed val items (est vs target vs mixture)
            s = getattr(self, '_val_samples', None)
            if s is not None:
                for i in range(s['est'].shape[0]):
                    for name in ('est', 'target', 'mixture'):
                        wav = s[name][i]
                        writer.add_audio(f'audio{i}/{name}', peak_norm(wav), step, sample_rate=16000)
                        writer.add_image(f'mel{i}/{name}', mel_image(wav), step)
                # PN-Enroll baseline OUTPUT audio + mel: run proposed-monaural on the SAME items for comparison
                if 'pos_wave' in s and self.pn_encoder is not None:
                    sep = self._get_baseline_separator()
                    if sep is not None:
                        for i in range(s['est'].shape[0]):
                            b = self._baseline_estimate(
                                sep, s['mixture'][i:i + 1].to(self.device), s['pos_wave'][i:i + 1].to(self.device),
                                s['neg_wave'][i:i + 1].to(self.device), s['target'][i:i + 1].to(self.device))[0].cpu()
                            writer.add_audio(f'audio{i}/baseline', peak_norm(b), step, sample_rate=16000)
                            writer.add_image(f'mel{i}/baseline', mel_image(b), step)
        self._val_samples = None

    def _get_baseline_separator(self):
        """Lazily load the frozen proposed-monaural full separator (rank0, once) for baseline audio/mel."""
        if getattr(self, '_baseline_sep', None) is None:
            try:
                from baseline_pnenroll import load_pn_full
                # store as a PLAIN attribute (object.__setattr__), NOT a registered submodule, so the
                # rank-0-only baseline separator never enters state_dict (resume) or buffers (broadcast).
                object.__setattr__(self, '_baseline_sep', load_pn_full(self.device))
                print('[val] loaded PN-Enroll baseline separator for TB audio/mel comparison', flush=True)
            except Exception as exc:  # baseline audio is optional; never block validation
                print(f'[val] baseline separator unavailable ({exc!r}); skipping baseline audio', flush=True)
                object.__setattr__(self, '_baseline_sep', False)
        return self._baseline_sep or None

    # 체크포인트를 불러올 때(이어서 학습/추론) 호출됩니다. 옛날 체크포인트에는 baseline 분리기가
    # 실수로 함께 저장되어 있어 로딩이 실패할 수 있는데, 그 찌꺼기 키를 미리 지워 깔끔하게 불러옵니다.
    def on_load_checkpoint(self, checkpoint):
        # Older checkpoints (saved before _baseline_sep was de-registered) leaked the rank-0 baseline
        # separator into state_dict -> resume failed with "Unexpected key(s) _baseline_sep.*". Strip them
        # so the (separator-free) model loads cleanly.
        sd = checkpoint.get('state_dict')
        if sd is not None:
            for k in [key for key in sd if key.startswith('_baseline_sep')]:
                del sd[k]

    @torch.no_grad()
    def _baseline_estimate(self, sep, mix, pos_wave, neg_wave, tgt):
        """proposed-monaural forward -> sign-matched estimate. pos_wave/neg_wave: [B,1,NW] (already mono)."""
        cond_emb = self.pn_encoder.encode(pos_wave, neg_wave)
        m = mix.unsqueeze(1) if mix.dim() == 2 else mix
        out, _ = sep(m, cond_emb, sep.init_buffers(m.shape[0], m.device))
        out = torch.cat([out, -out], dim=1)
        idx = torch.min((out - tgt[:, None, :]).pow(2).sum(dim=-1), dim=1)[1]
        idx = idx[:, None, None].repeat((1, 1, out.shape[-1]))
        return torch.gather(out, 1, idx).squeeze(1)

    # optimizer(최적화기)와 scheduler(학습률 스케줄) 설정.
    def configure_optimizers(self):
        # 학습 대상은 "학습 가능한(requires_grad=True)" 파라미터뿐: UDiT + enroll_cond 어댑터.
        # frozen PN encoder는 정식 서브모듈로 등록돼 있어도 requires_grad=False라 자동으로 제외됩니다.
        optimizer = make_optimizer([p for p in self.parameters() if p.requires_grad], **self.config['optim'])
        
        if self.config['scheduler']['type'] == 'ReduceLROnPlateau':
            scheduler = {
                'scheduler': ReduceLROnPlateau(
                    optimizer=optimizer,
                    factor=self.config['scheduler']['lr_reduce_factor'],
                    patience=self.config['scheduler']['lr_reduce_patience']
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        elif self.config['scheduler']['type'] == 'CosineAnnealingLR':
            def warmup_lambda(epoch):
                if epoch < self.config['scheduler']['warmup_epochs']:
                    return epoch / self.config['scheduler']['warmup_epochs']
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['scheduler']['t_max'],
                eta_min=self.config['scheduler']['eta_min']
            )

            scheduler = {
                'scheduler': SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.config['scheduler']['warmup_epochs']]
                ),
                'interval': 'epoch',
                'frequency': 1
            }

            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))

    def setup(self, stage=None):
        self.train_loader, self.val_loader = get_dataloaders(
            self.config,
            is_ddp=False,
            world_size=self.world_size,
            rank=self.rank
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main():
    # 1) config 읽고, 재현성 위해 시드 고정. matmul precision 'medium'으로 속도/메모리 균형.
    args = parse_args()
    config = parse_config(args.config)
    pl.seed_everything(config['seed'])
    torch.set_float32_matmul_precision('medium')

    # 로그/체크포인트 폴더 미리 만들기.
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)

    # --- 모델 만들기 / 체크포인트 불러오기 -------------------------------------
    # load_weights_only=True + resume 경로가 있으면: 가중치만 불러와 처음부터(스텝0) 시작(fine-tune 느낌).
    # 그 외에는: 새 모델을 만들고, 아래 trainer.fit에서 ckpt_path로 "이어서" 학습할 수 있음.
    if config['checkpoint']['load_weights_only'] and config['checkpoint']['resume']:
        model = LightningModule.load_from_checkpoint(
            config['checkpoint']['resume'],
            config=config,
            strict=True
        )
    else:
        model = LightningModule(config)

    data_module = DataModule(config)

    callbacks = []

    # Add metric printer callback
    metric_printer = MetricPrinterCallback()
    callbacks.append(metric_printer)
    callbacks.append(TQDMProgressBar(refresh_rate=1))  # live tqdm (it/s + step/total + ETA); refresh every step

    if config['early_stopping']['enabled']:
        early_stopping_callback = EarlyStopping(
            monitor=config['early_stopping']['monitor'],
            patience=config['early_stopping']['patience'],
            verbose=config['early_stopping']['verbose'],
            mode=config['early_stopping']['mode'],
            min_delta=config['early_stopping']['delta'],
        )
        callbacks.append(early_stopping_callback)

    # 메인 체크포인트 저장: monitor(=val_loss) 기준 best와 last를 저장합니다.
    # Main checkpoint callback for best/last
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint']['dir'],
        filename=config['checkpoint']['ckpt_name'],
        save_top_k=config['checkpoint']['save_best'],
        save_last=config['checkpoint']['save_last'],
        verbose=config['checkpoint']['verbose'],
        monitor=config['checkpoint']['monitor'],
        mode=config['checkpoint']['mode'],
    )
    callbacks.append(checkpoint_callback)

    # Periodic checkpoint callback (saves every N epochs)
    periodic_config = config['checkpoint'].get('periodic', {})
    if periodic_config.get('enabled', False):
        save_every_n_epochs = periodic_config.get('save_every_n_epochs', 100)
        periodic_checkpoint_callback = ModelCheckpoint(
            dirpath=config['checkpoint']['dir'],
            filename='epoch_{epoch:04d}',
            save_top_k=-1,  # Save all periodic checkpoints
            every_n_epochs=save_every_n_epochs,
            verbose=periodic_config.get('verbose', True),
            save_on_train_epoch_end=True,
        )
        callbacks.append(periodic_checkpoint_callback)
        print(f"Periodic checkpointing enabled: saving every {save_every_n_epochs} epochs")

    if config['train']['log_lr']:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

    tb_logger = TensorBoardLogger(
        save_dir=config['train']['log_dir'],
        name='lightning_logs',
        version='0',
    )

    # --- DDP(멀티 GPU 분산학습) 설정 ----------------------------------------
    # single node(한 대 컴퓨터)에서 GPU 1~4장을 쓰는 상황을 가정합니다.
    ddp_config = config.get('ddp', {})
    use_ddp = ddp_config.get('use_ddp', False)
    num_nodes = ddp_config.get('num_nodes', 1)
    num_gpus = ddp_config.get('num_gpus', torch.cuda.device_count())
    # find_unused_parameters=True: MeanFlow's loss switches its autograd graph step-to-step
    # (rectified-flow / alpha-flow-alpha1 / alpha-flow = 1 vs 2 model forwards via the alpha schedule
    # + flow_ratio). DDP's default static reducer (False) assumes a fixed graph and deadlocks when it
    # changes; True re-traverses the graph each iteration. (env DDP_FIND_UNUSED=0 forces the old behavior.)
    import os as _os
    _find_unused = _os.environ.get('DDP_FIND_UNUSED', '1') != '0'
    # broadcast_buffers=False: this model's buffers are all either deterministic (RoPE inv_freq,
    # positional encoding) or frozen-from-checkpoint (PN encoder) -> identical across ranks by
    # construction, never updated in training, so DDP's start-of-forward buffer broadcast is
    # unnecessary. That broadcast otherwise DEADLOCKS at >2 GPUs: a frozen PN-encoder buffer has a
    # data-dependent shape that differs across ranks (TORCH_DISTRIBUTED_DEBUG=DETAIL caught it:
    # BROADCAST TensorShape=[34080] on rank0 vs [544] on rank1/3 -> mismatched coalesced broadcast).
    strategy = DDPStrategy(find_unused_parameters=_find_unused, broadcast_buffers=False)

    # GPU 2장 이상 + use_ddp면 DDP(분산) Trainer, 아니면 단일 GPU(또는 CPU) Trainer.
    # 두 분기의 학습 로직은 같고, 멀티 GPU 분기만 strategy/devices/num_nodes가 추가됩니다.
    if use_ddp and num_gpus > 1:
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            accelerator='gpu',
            devices=num_gpus,
            num_nodes=num_nodes,
            strategy=strategy,
            accumulate_grad_batches=config['train']['accumulation_steps'],
            callbacks=callbacks,
            default_root_dir=config['train']['log_dir'],
            logger=tb_logger,
            log_every_n_steps=config['train']['log_interval'],
            precision=config['train']['precision'],
            detect_anomaly=config['train']['detect_anomaly'],
            gradient_clip_val=config['train']['gradient_clip_val'],
            limit_train_batches=config['train']['limit_train_batches'],
            limit_val_batches=config['train']['limit_val_batches'],
            enable_progress_bar=True,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            accumulate_grad_batches=config['train']['accumulation_steps'],
            callbacks=callbacks,
            default_root_dir=config['train']['log_dir'],
            logger=tb_logger,
            log_every_n_steps=config['train']['log_interval'],
            precision=config['train']['precision'],
            gradient_clip_val=config['train']['gradient_clip_val'],
            detect_anomaly=config['train']['detect_anomaly'],
            limit_train_batches=config['train']['limit_train_batches'],
            limit_val_batches=config['train']['limit_val_batches'],
            enable_progress_bar=True
        )

    # 학습 시작! load_weights_only면 가중치만 들고 처음부터, 아니면 ckpt_path로 "이어서" 학습합니다.
    # (resume 경로가 비어 있으면 ckpt_path=None -> 완전히 새로 시작)
    ckpt_path = config['checkpoint']['resume'] if config['checkpoint']['resume'] else None

    if config['checkpoint']['load_weights_only']:
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
