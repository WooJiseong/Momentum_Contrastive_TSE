"""Train Attn_MOCOCO + fixed improved-anchor Soft teacher Stage0."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import torch


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
PN_MOCOCO_DIR = LAB_DIR.parent / "PN_MOCOCO"
ATTN_DIR = LAB_DIR.parent / "Attn_MOCOCO"
BASE_DIR = PROJECT_ROOT / "Base" / "TSE-through-Positive-Negative-Enroll"

sys.path[:0] = [
    str(LAB_DIR),
    str(ATTN_DIR),
    str(PN_MOCOCO_DIR),
    str(BASE_DIR),
    str(PROJECT_ROOT),
]

# Patch only the loader function used by the shared MoCo learner.  The
# contrastive objective, EMA queue, speaker masking, and Attn projection remain
# the established Attn_MOCOCO implementation.
from improved_pn import load_trainable_pn_encoder
from pn_mococo import moco_encoder

moco_encoder.load_trainable_pn_encoder = load_trainable_pn_encoder

base_spec = importlib.util.spec_from_file_location(
    "improved_attn_mococo_base_trainer",
    PN_MOCOCO_DIR / "train_moco_encoder.py",
)
if base_spec is None or base_spec.loader is None:
    raise ImportError("Unable to load the shared PN_MOCOCO trainer")
base_train = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base_train
base_spec.loader.exec_module(base_train)
BASE_LIGHTNING_MODULE = base_train.LightningModule

from pn_attn_mococo.attn_moco import AttnSoftMomentumContrastivePNLearner


def normalize_training_paths(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("paths", {})
    cfg.setdefault("checkpoint", {})

    def resolve(section: str, key: str) -> None:
        value = cfg.get(section, {}).get(key)
        if value:
            path = Path(value).expanduser()
            cfg[section][key] = str(path if path.is_absolute() else (LAB_DIR / path).resolve())

    for section, key in (
        ("train", "log_dir"),
        ("checkpoint", "dir"),
        ("checkpoint", "resume"),
        ("paths", "initial_pn_ckpt"),
        ("contrastive", "initial_pn_ckpt"),
    ):
        resolve(section, key)
    return cfg


class ImprovedAttnLightningModule(base_train.pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.learner = AttnSoftMomentumContrastivePNLearner(config)
        self._last_train_bsz = 1
        ccfg = config.get("contrastive", {}) or {}
        self.teacher_weight_start = float(
            ccfg.get("teacher_loss_weight_start", ccfg.get("teacher_loss_weight", 0.1))
        )
        self.teacher_weight_end = float(
            ccfg.get("teacher_loss_weight_end", self.teacher_weight_start)
        )
        self.teacher_weight_schedule = str(
            ccfg.get("teacher_loss_weight_schedule", "constant")
        ).lower().strip()
        if self.teacher_weight_start < 0 or self.teacher_weight_end < 0:
            raise ValueError("Teacher loss weights must be non-negative.")
        self._epoch_metric_sums = {"train": {}, "val": {}}
        self._epoch_metric_counts = {"train": 0, "val": 0}

    def _augment_wave(self, x, strong: bool):
        """Reuse the established PN_MOCOCO waveform augmentation path."""
        return BASE_LIGHTNING_MODULE._augment_wave(self, x, strong)

    def _pairs(self, batch: dict, train: bool):
        """Reuse the established positive/negative pair construction path."""
        return BASE_LIGHTNING_MODULE._pairs(self, batch, train)

    def _current_teacher_weight(self) -> float:
        """Return the teacher coefficient at the current optimizer progress."""
        if self.teacher_weight_schedule in {"constant", "none", ""}:
            return self.teacher_weight_start
        if self.teacher_weight_schedule != "cosine":
            raise ValueError(
                "contrastive.teacher_loss_weight_schedule must be 'constant' "
                f"or 'cosine', got {self.teacher_weight_schedule!r}"
            )

        trainer = getattr(self, "trainer", None)
        total_steps = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        if total_steps > 1:
            progress = float(self.global_step) / float(total_steps - 1)
        else:
            max_epochs = int(
                getattr(trainer, "max_epochs", self.config["train"]["num_epochs"])
            )
            progress = float(self.current_epoch) / float(max(1, max_epochs - 1))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.teacher_weight_end + (
            self.teacher_weight_start - self.teacher_weight_end
        ) * cosine

    def _shared_step(self, batch: dict, train: bool):
        (
            q_pos,
            q_neg,
            pos_key,
            neg_key,
            negative_keys,
            target_spk_id,
            negative_speaker_ids,
        ) = self._pairs(batch, train=train)
        query = self.learner.student_embedding(q_pos, q_neg)
        with base_train.torch.no_grad():
            positive = self.learner.momentum_embedding(pos_key, neg_key)
        moco_loss, logs = self.learner.contrastive_loss(
            query=query,
            positive_key=positive,
            negative_keys=negative_keys,
            use_queue=train and bool(self.config.get("contrastive", {}).get("use_queue", True)),
            update_queue=train,
            query_speaker_ids=target_spk_id,
            negative_speaker_ids=negative_speaker_ids,
        )
        teacher_loss = self.learner.teacher_loss(q_pos, q_neg)
        teacher_weight = self._current_teacher_weight()
        total = moco_loss + teacher_weight * teacher_loss
        logs["teacher_loss"] = teacher_loss.detach()
        logs["teacher_weight"] = moco_loss.new_tensor(teacher_weight)
        return total, logs

    def _reset_epoch_metrics(self, prefix: str) -> None:
        self._epoch_metric_sums[prefix] = {}
        self._epoch_metric_counts[prefix] = 0

    def _accumulate_epoch_metrics(self, prefix: str, metrics: dict) -> None:
        sums = self._epoch_metric_sums[prefix]
        for name, value in metrics.items():
            value = value.detach().float()
            sums[name] = sums.get(name, torch.zeros_like(value)) + value
        self._epoch_metric_counts[prefix] += 1

    def _log_epoch_metrics(self, prefix: str) -> None:
        sums = self._epoch_metric_sums[prefix]
        count = self._epoch_metric_counts[prefix]
        if not sums or count <= 0:
            return

        world_size = 1
        if base_train.torch.distributed.is_available() and base_train.torch.distributed.is_initialized():
            world_size = base_train.torch.distributed.get_world_size()

        for name, total in sums.items():
            total = total.clone()
            denominator = total.new_tensor(float(count))
            if world_size > 1:
                base_train.torch.distributed.all_reduce(
                    total, op=base_train.torch.distributed.ReduceOp.SUM
                )
                denominator = total.new_tensor(float(count))
                denominator *= world_size
            # The value is already globally reduced. Disable Lightning's own
            # distributed reduction, which otherwise creates a CPU state tensor.
            self.log(
                f"{prefix}_{name}",
                total / denominator,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                batch_size=1,
            )
        self._reset_epoch_metrics(prefix)

    def on_train_epoch_start(self):
        self._reset_epoch_metrics("train")

    def on_validation_epoch_start(self):
        self._reset_epoch_metrics("val")

    def training_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch, True)
        self._accumulate_epoch_metrics(
            "train",
            {
                "loss": loss,
                "moco_loss": logs["loss"],
                "teacher_loss": logs["teacher_loss"],
                "teacher_weight": logs["teacher_weight"],
                "pos_sim": logs["pos_sim"],
                "queue_masked_ratio": logs["queue_masked_ratio"],
            },
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch, False)
        self._accumulate_epoch_metrics(
            "val",
            {
                "loss": loss,
                "moco_loss": logs["loss"],
                "teacher_loss": logs["teacher_loss"],
                "teacher_weight": logs["teacher_weight"],
                "pos_sim": logs["pos_sim"],
            },
        )
        return loss

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_before_zero_grad(self, optimizer):
        """Update EMA without Lightning's CPU metric accumulation path."""
        total = int(getattr(self.trainer, "estimated_stepping_batches", 1) or 1)
        momentum = self.learner.ema_momentum(int(self.global_step), total)
        self.learner.update_momentum_encoder(momentum)

    def configure_optimizers(self):
        """Build the optimizer for the improved student encoder and head.

        This wrapper subclasses LightningModule directly, so it cannot inherit
        the optimizer hook from PN_MOCOCO's trainer. Keep the same parameter
        groups and warmup/cosine schedule as the shared Stage0 implementation.
        """
        ocfg = self.config["optim"]
        base_lr = float(ocfg.get("lr", 1e-4))
        enc_scale = float(
            self.config.get("contrastive", {}).get("encoder_lr_scale", 0.03)
        )
        enc_params = [
            p for p in self.learner.student_encoder.parameters() if p.requires_grad
        ]
        head_params = [
            p for p in self.learner.student_head.parameters() if p.requires_grad
        ]
        param_groups = []
        if enc_params:
            param_groups.append({"params": enc_params, "lr": base_lr * enc_scale})
        if head_params:
            param_groups.append({"params": head_params, "lr": base_lr})
        if not param_groups:
            raise ValueError("No trainable improved Stage0 parameters were found.")

        optimizer = base_train.build_optimizer(
            param_groups,
            {**ocfg, "lr": base_lr},
        )
        scheduler_cfg = self.config.get("scheduler", {})
        if scheduler_cfg.get("type") != "CosineAnnealingLR":
            return optimizer

        warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))

        def warmup_lambda(epoch):
            if warmup_epochs <= 0:
                return 1.0
            return epoch / warmup_epochs if epoch < warmup_epochs else 1.0

        warmup = base_train.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine = base_train.CosineAnnealingLR(
            optimizer,
            T_max=int(
                scheduler_cfg.get("t_max", self.config["train"]["num_epochs"])
            ),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": base_train.SequentialLR(
                    optimizer,
                    [warmup, cosine],
                    milestones=[warmup_epochs],
                ),
                "interval": "epoch",
                "frequency": 1,
            },
        }


class PeriodicImprovedExportCallback(base_train.ExportPNEncoderCallback):
    """Export best/last weights and a directly reusable 50-epoch sweep grid."""

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        completed_epoch = int(trainer.current_epoch) + 1
        if completed_epoch % 50 == 0:
            self._export(pl_module, self.export_dir / f"pn_encoder_{completed_epoch}ep.pt")


base_train.LightningModule = ImprovedAttnLightningModule
base_train.ExportPNEncoderCallback = PeriodicImprovedExportCallback
base_train.normalize_paths = normalize_training_paths


if __name__ == "__main__":
    base_train.main()
