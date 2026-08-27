from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

project_dir = Path(__file__).resolve().parent
pn_mococo = project_dir.parent / "PN_MOCOCO"
sys.path.insert(0, str(pn_mococo))
sys.path.insert(0, str(project_dir))

base_spec = importlib.util.spec_from_file_location(
    "pn_mococo_base_train_attn_tfgridnet_leakage",
    pn_mococo / "train_tfgridnet.py",
)
if base_spec is None or base_spec.loader is None:
    raise ImportError(f"Cannot load PN_MOCOCO trainer from {pn_mococo / 'train_tfgridnet.py'}")
base_train = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base_train
base_spec.loader.exec_module(base_train)

from pn_attn_mococo.losses import (
    build_target_orthogonal_leakage_kwargs,
    target_orthogonal_leakage_loss,
)
from pn_attn_mococo.stage1_data import get_attn_leakage_dataloaders
from pn_mococo.tfgridnet_model import neg_si_sdr_loss, separation_metrics


class AttnLeakageLightningModule(base_train.LightningModule):
    def __init__(self, config: dict):
        super().__init__(config)
        loss_cfg = config.get("loss", {})
        self.orthogonal_weight = float(loss_cfg.get("orthogonal_leakage_weight", 0.1))
        self.leakage_loss_kwargs = build_target_orthogonal_leakage_kwargs(loss_cfg)
        self.gate_mode = str(self.leakage_loss_kwargs["gate_mode"]).lower().strip()
        self.verify_nuisance_contract = bool(loss_cfg.get("verify_nuisance_contract", True))
        self.fail_on_nonfinite = bool(loss_cfg.get("fail_on_nonfinite", True))
        self._nuisance_contract_checked = False
        self.step_debug = os.environ.get("ATTN_LEAKAGE_STEP_DEBUG", "0") == "1"

    def _debug_mark(self, label: str, start: float | None = None) -> float:
        now = time.perf_counter()
        if self.step_debug:
            rank = os.environ.get("LOCAL_RANK", "0")
            elapsed = "" if start is None else f" elapsed={now - start:.3f}s"
            print(f"[leakage-debug rank={rank}] {label}{elapsed}", flush=True)
        return now

    def _nuisance_sources(self, batch, target, mixture):
        if "nuisance_sources" not in batch:
            raise KeyError("Batch is missing individual nuisance_sources [B,J,T]")
        nuisance_sources = batch["nuisance_sources"]
        if nuisance_sources.ndim != 3:
            raise ValueError(f"nuisance_sources must be [B,J,T], got {tuple(nuisance_sources.shape)}")
        if nuisance_sources.shape[0] != target.shape[0] or nuisance_sources.shape[-1] != target.shape[-1]:
            raise ValueError("nuisance_sources must match source batch/time dimensions")
        reconstructed_background = nuisance_sources.sum(dim=1)
        expected_background = mixture - target
        reconstruction_mae = (reconstructed_background - expected_background).abs().mean()
        if self.verify_nuisance_contract and not self._nuisance_contract_checked:
            max_error = (reconstructed_background - expected_background).abs().amax()
            allowed_error = 1e-5 * expected_background.abs().amax().clamp_min(1.0)
            if not torch.isfinite(nuisance_sources).all() or max_error > allowed_error:
                raise RuntimeError(
                    "Invalid nuisance contract: sum(nuisance_sources) must reconstruct mixture-source; "
                    f"max_error={float(max_error):.3e}, allowed={float(allowed_error):.3e}"
                )
            self._nuisance_contract_checked = True
        return nuisance_sources, reconstruction_mae

    def _attn_step(self, batch, train: bool):
        started = self._debug_mark(f"{'train' if train else 'val'}_step_start")
        est, target, mixture = self._forward_batch(batch)
        started = self._debug_mark("forward_done", started)
        base_loss = neg_si_sdr_loss(est, target)
        started = self._debug_mark("si_sdr_done", started)
        interferers, reconstruction_mae = self._nuisance_sources(batch, target, mixture)
        leakage_loss, stats = target_orthogonal_leakage_loss(
            est,
            target,
            interferers,
            **self.leakage_loss_kwargs,
            return_stats=True,
        )
        started = self._debug_mark("leakage_done", started)
        total = base_loss + self.orthogonal_weight * leakage_loss
        if self.fail_on_nonfinite and not torch.isfinite(total).all():
            raise FloatingPointError(
                f"Non-finite Stage1 loss: si_sdr={float(base_loss.detach()):.6g}, "
                f"leakage={float(leakage_loss.detach()):.6g}"
            )
        metrics = separation_metrics(
            est.detach() if train else est,
            target,
            mixture,
            reference=not train,
        )
        started = self._debug_mark("metrics_done", started)
        prefix = "train" if train else "val"
        bsz = int(target.shape[0])
        self.log(f"{prefix}_loss", total, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_si_sdr_loss", base_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_leakage_loss", leakage_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_gate_ratio", stats["gate_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_local_leakage", stats["local_leakage"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_target_activity_ratio", stats["target_activity_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_orthogonal_nuisance_activity_ratio", stats["nuisance_activity_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        if self.gate_mode == "stft_magnitude_overlap":
            self.log(f"{prefix}_stft_magnitude_overlap", stats["stft_magnitude_overlap"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
            self.log(f"{prefix}_stft_magnitude_gate_ratio", stats["stft_magnitude_gate_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(f"{prefix}_nuisance_reconstruction_mae", reconstruction_mae, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        for key, value in metrics.items():
            self.log(f"{prefix}_{key}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self._debug_mark(f"{prefix}_step_done", started)
        return total

    def training_step(self, batch, batch_idx):
        return self._attn_step(batch, train=True)

    def validation_step(self, batch, batch_idx):
        return self._attn_step(batch, train=False)


class AttnLeakageDataModule(base_train.DataModule):
    def setup(self, stage=None):
        self.train_loader, self.val_loader = get_attn_leakage_dataloaders(
            self.config,
            is_ddp=False,
            world_size=self.world_size,
            rank=self.rank,
        )


base_train.LightningModule = AttnLeakageLightningModule
base_train.DataModule = AttnLeakageDataModule
base_train.main()
