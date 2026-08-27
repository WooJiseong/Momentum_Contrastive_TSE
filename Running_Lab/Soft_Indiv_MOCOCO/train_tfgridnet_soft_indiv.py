from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from pn_soft_indiv_mococo.data import get_stage1_dataloaders
from pn_soft_indiv_mococo.losses import (
    build_target_orthogonal_leakage_kwargs,
    target_orthogonal_leakage_loss,
)
from pn_soft_indiv_mococo.resume import ResumableLightningMixin, ResumeStateCallback

project_dir = Path(__file__).resolve().parent
pn_mococo = project_dir.parent / "PN_MOCOCO"
sys.path.insert(0, str(pn_mococo))
sys.path.insert(0, str(project_dir))

base_spec = importlib.util.spec_from_file_location(
    "soft_indiv_base_train_tfgridnet", pn_mococo / "train_tfgridnet.py"
)
if base_spec is None or base_spec.loader is None:
    raise ImportError(f"Cannot load PN_MOCOCO trainer from {pn_mococo / 'train_tfgridnet.py'}")
base_train = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base_train
base_spec.loader.exec_module(base_train)

from pn_mococo.tfgridnet_model import neg_si_sdr_loss, separation_metrics


class SoftIndivStage1LightningModule(ResumableLightningMixin, base_train.LightningModule):
    """SI-SDR plus target-orthogonal leakage from actual individual mixture sources."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.init_reproducibility(int(config.get("seed", 42)))
        loss_cfg = config.get("loss", {})
        self.orthogonal_weight = float(loss_cfg.get("orthogonal_leakage_weight", 0.1))
        self.leakage_loss_kwargs = build_target_orthogonal_leakage_kwargs(loss_cfg)
        self.gate_mode = str(self.leakage_loss_kwargs["gate_mode"]).lower().strip()
        self.verify_nuisance_contract = bool(loss_cfg.get("verify_nuisance_contract", True))
        self.fail_on_nonfinite = bool(loss_cfg.get("fail_on_nonfinite", True))
        self._nuisance_contract_checked = False

    def _nuisance_sources(
        self, batch: dict, target: torch.Tensor, mixture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "nuisance_sources" not in batch:
            raise KeyError("Stage1 leakage loss requires actual sample[1:] nuisance_sources, not aggregate background.")
        nuisance_sources = batch["nuisance_sources"]
        if nuisance_sources.ndim != 3:
            raise ValueError(f"nuisance_sources must be [B, J, T], got {tuple(nuisance_sources.shape)}")
        if nuisance_sources.shape[0] != target.shape[0] or nuisance_sources.shape[-1] != target.shape[-1]:
            raise ValueError(
                "nuisance_sources must match source batch/time dimensions: "
                f"sources={tuple(nuisance_sources.shape)}, source={tuple(target.shape)}"
            )
        reconstructed_background = nuisance_sources.sum(dim=1)
        expected_background = mixture - target
        reconstruction_mae = (reconstructed_background - expected_background).abs().mean()
        if self.verify_nuisance_contract and not self._nuisance_contract_checked:
            max_error = (reconstructed_background - expected_background).abs().amax()
            allowed = 1e-5 * expected_background.abs().amax().clamp_min(1.0)
            if not torch.isfinite(nuisance_sources).all() or max_error > allowed:
                raise RuntimeError(
                    "Invalid nuisance contract: sum(nuisance_sources) must reconstruct mixture - source. "
                    f"max_error={float(max_error):.3e}, allowed={float(allowed):.3e}"
                )
            self._nuisance_contract_checked = True
        return nuisance_sources, reconstruction_mae

    def _step(self, batch: dict, train: bool):
        est, target, mixture = self._forward_batch(batch)
        si_sdr_loss = neg_si_sdr_loss(est, target)
        nuisance_sources, reconstruction_mae = self._nuisance_sources(batch, target, mixture)
        leakage_loss, stats = target_orthogonal_leakage_loss(
            est,
            target,
            nuisance_sources,
            **self.leakage_loss_kwargs,
            return_stats=True,
        )
        loss = si_sdr_loss + self.orthogonal_weight * leakage_loss
        if self.fail_on_nonfinite and not torch.isfinite(loss).all():
            raise FloatingPointError(
                "Soft_Indiv_MOCOCO Stage1 produced a non-finite loss: "
                f"si_sdr={float(si_sdr_loss.detach()):.6g}, leakage={float(leakage_loss.detach()):.6g}"
            )
        metrics = separation_metrics(
            est.detach() if train else est,
            target,
            mixture,
            reference=not train,
        )
        prefix = "train" if train else "val"
        bsz = int(target.shape[0])
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_si_sdr_loss", si_sdr_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_leakage_loss", leakage_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_gate_ratio", stats["gate_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_local_leakage", stats["local_leakage"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_target_activity_ratio", stats["target_activity_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_nuisance_activity_ratio", stats["nuisance_activity_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_ratio", stats["orthogonal_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        if self.gate_mode == "stft_magnitude_overlap":
            self.log(f"{prefix}_stft_magnitude_overlap", stats["stft_magnitude_overlap"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
            self.log(f"{prefix}_stft_magnitude_gate_ratio", stats["stft_magnitude_gate_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_nuisance_count", float(nuisance_sources.shape[1]), on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_nuisance_reconstruction_mae", reconstruction_mae, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        for key, value in metrics.items():
            self.log(f"{prefix}_{key}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, train=True)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, train=False)


class SoftIndivStage1DataModule(base_train.DataModule):
    def setup(self, stage=None):
        self.train_loader, self.val_loader = get_stage1_dataloaders(
            self.config,
            is_ddp=False,
            world_size=self.world_size,
            rank=self.rank,
        )


class ReproducibilityMetricPrinter(base_train.MetricPrinterCallback):
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        checkpoint_dir = Path(pl_module.config["checkpoint"]["dir"])
        ResumeStateCallback(
            checkpoint_dir, "stage1", int(pl_module.config.get("seed", 42))
        )._save(trainer, "last")


base_train.LightningModule = SoftIndivStage1LightningModule
base_train.DataModule = SoftIndivStage1DataModule
base_train.MetricPrinterCallback = ReproducibilityMetricPrinter


def main() -> None:
    base_train.main()


if __name__ == "__main__":
    main()
