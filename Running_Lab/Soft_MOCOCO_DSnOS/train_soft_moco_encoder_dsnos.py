"""Soft_MOCOCO Stage0 with dataset-balanced root sampling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
PN_MOCOCO_DIR = PROJECT_ROOT / "Running_Lab" / "PN_MOCOCO"
sys.path.insert(0, str(PROJECT_ROOT / "Running_Lab" / "Soft_MOCOCO"))
sys.path.insert(0, str(PN_MOCOCO_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(LAB_DIR / "src"))

base_spec = importlib.util.spec_from_file_location(
    "dsnos_base_train_moco_encoder",
    PN_MOCOCO_DIR / "train_moco_encoder.py",
)
if base_spec is None or base_spec.loader is None:
    raise ImportError(f"Cannot load {PN_MOCOCO_DIR / 'train_moco_encoder.py'}")
base_train = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base_train
base_spec.loader.exec_module(base_train)

from pn_soft_mococo.paths import normalize_training_paths
from pn_soft_mococo.soft_moco import SoftMomentumContrastivePNLearner
from dsnos_data import get_dataloaders


class SoftLightningModule(base_train.LightningModule):
    """Keep Soft_MOCOCO loss and swap only the train-root sampler."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.learner = SoftMomentumContrastivePNLearner(config)

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
            use_queue=train
            and bool(self.config.get("contrastive", {}).get("use_queue", True)),
            update_queue=train,
            query_speaker_ids=target_spk_id,
            negative_speaker_ids=negative_speaker_ids,
        )
        teacher_loss = self.learner.teacher_loss(q_pos, q_neg)
        total = moco_loss + self.learner.teacher_loss_weight * teacher_loss
        logs["teacher_loss"] = teacher_loss.detach()
        return total, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch, True)
        bsz = int(batch["pos_wave"].shape[0])
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_moco_loss", logs["loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_teacher_loss", logs["teacher_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_pos_sim", logs["pos_sim"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_queue_masked_ratio", logs["queue_masked_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch, False)
        bsz = int(batch["pos_wave"].shape[0])
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("val_moco_loss", logs["loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("val_teacher_loss", logs["teacher_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("val_pos_sim", logs["pos_sim"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        return loss


base_train.LightningModule = SoftLightningModule
base_train.get_dataloaders = get_dataloaders
base_train.normalize_paths = normalize_training_paths
base_train.main()
