"""Train the standard Soft_MOCOCO encoder with a decaying teacher anchor."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


LAB_DIR = Path(__file__).resolve().parent
PN_MOCOCO_DIR = LAB_DIR.parent / "PN_MOCOCO"
sys.path[:0] = [str(LAB_DIR), str(PN_MOCOCO_DIR)]

base_spec = importlib.util.spec_from_file_location(
    "soft_teacher_decay_base_train_moco_encoder",
    PN_MOCOCO_DIR / "train_moco_encoder.py",
)
if base_spec is None or base_spec.loader is None:
    raise ImportError(f"Cannot load PN_MOCOCO trainer from {PN_MOCOCO_DIR / 'train_moco_encoder.py'}")
base_train = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base_train
base_spec.loader.exec_module(base_train)

from pn_soft_mococo.paths import normalize_training_paths
from pn_soft_mococo.soft_moco import SoftMomentumContrastivePNLearner


_BASE_LIGHTNING_MODULE = base_train.LightningModule


class SoftTeacherDecayLightningModule(_BASE_LIGHTNING_MODULE):
    """Standard Soft_MOCOCO objective with a cosine-decayed fixed teacher term."""

    def __init__(self, config: dict):
        base_train.pl.LightningModule.__init__(self)
        self.config = config
        self.save_hyperparameters(config)
        self.learner = SoftMomentumContrastivePNLearner(config)
        self._last_train_bsz = 1
        self._epoch_metric_sums = {"train": {}, "val": {}}
        self._epoch_metric_counts = {"train": 0, "val": 0}

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

    def _current_teacher_weight(self) -> float:
        if self.teacher_weight_schedule in {"constant", "none", ""}:
            return self.teacher_weight_start
        if self.teacher_weight_schedule != "cosine":
            raise ValueError(
                "contrastive.teacher_loss_weight_schedule must be 'constant' or 'cosine', "
                f"got {self.teacher_weight_schedule!r}"
            )

        # Use Lightning's internal reference so pre-Trainer smoke tests also
        # fall back cleanly to the configured epoch count.
        trainer = getattr(self, "_trainer", None)
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
        with torch.no_grad():
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
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

        for name, total in sums.items():
            total = total.clone()
            if world_size > 1:
                torch.distributed.all_reduce(
                    total, op=torch.distributed.ReduceOp.SUM
                )
            denominator = total.new_tensor(float(count * world_size))
            # The value is globally reduced above. Avoid Lightning's CPU
            # cumulated_batch_size reduction with the NCCL backend.
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
        loss, logs = self._shared_step(batch, train=True)
        bsz = int(batch["pos_wave"].shape[0])
        self._last_train_bsz = bsz
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
        loss, logs = self._shared_step(batch, train=False)
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
        total = int(getattr(self.trainer, "estimated_stepping_batches", 1) or 1)
        momentum = self.learner.ema_momentum(int(self.global_step), total)
        self.learner.update_momentum_encoder(momentum)

    def configure_optimizers(self):
        """Reuse the standard Soft/PN_MOCOCO optimizer and LR schedule."""
        return _BASE_LIGHTNING_MODULE.configure_optimizers(self)


class PeriodicSoftExportCallback(base_train.ExportPNEncoderCallback):
    """Keep the normal best/last exports and one student export per 50 epochs."""

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        completed_epoch = int(trainer.current_epoch) + 1
        if completed_epoch % 50 == 0:
            self._export(
                pl_module,
                self.export_dir / f"pn_encoder_{completed_epoch}ep.pt",
            )


base_train.LightningModule = SoftTeacherDecayLightningModule
base_train.ExportPNEncoderCallback = PeriodicSoftExportCallback
base_train.normalize_paths = normalize_training_paths

if __name__ == "__main__":
    base_train.main()
