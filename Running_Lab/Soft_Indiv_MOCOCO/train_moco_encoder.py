from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from pn_soft_indiv_mococo.data import get_stage0_dataloaders
from pn_soft_indiv_mococo.paths import add_repo_paths, project_path
from pn_soft_indiv_mococo.resume import ResumableLightningMixin, ResumeStateCallback
from pn_soft_indiv_mococo.soft_indiv_moco import SoftIndividualNegativeMoCo

add_repo_paths()


class ExportPNEncoderCallback(Callback):
    def __init__(self, export_dir: Path, monitor: str, mode: str, every_n_epochs: int):
        super().__init__()
        self.export_dir = Path(export_dir)
        self.monitor = monitor
        self.mode = mode
        self.every_n_epochs = int(every_n_epochs)
        self.best: float | None = None

    def state_dict(self) -> dict:
        return {"best": self.best}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best = state_dict.get("best")

    def _is_better(self, value: float) -> bool:
        return self.best is None or (value < self.best if self.mode == "min" else value > self.best)

    @staticmethod
    def _export(pl_module, path: Path, completed_epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu() for key, value in pl_module.model.student.state_dict().items()}
        torch.save(
            {
                "state_dict": state,
                "meta": {
                    "format": "PNEncodePath encoder.* + encoder_head.*",
                    "source": "Soft_Indiv_MOCOCO frozen-Teacher individual-negative student encoder",
                    "global_step": int(pl_module.global_step),
                    "epoch": completed_epoch,
                    "teacher_loss_weight": pl_module.model.teacher_loss_weight,
                    "negative_aggregation": pl_module.model.negative_aggregation,
                },
            },
            path,
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        completed = int(trainer.current_epoch) + 1
        self._export(pl_module, self.export_dir / "pn_encoder_last.pt", completed)
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is not None and self._is_better(float(metric.detach().cpu())):
            self.best = float(metric.detach().cpu())
            self._export(pl_module, self.export_dir / "pn_encoder_best.pt", completed)
        if self.every_n_epochs > 0 and completed % self.every_n_epochs == 0:
            trainer.save_checkpoint(str(self.export_dir / f"stage0_epoch_{completed:03d}.ckpt"))
            self._export(pl_module, self.export_dir / f"pn_encoder_epoch_{completed:03d}.pt", completed)

    def on_train_end(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self._export(pl_module, self.export_dir / "pn_encoder_last.pt", int(trainer.current_epoch) + 1)


class MetricPrinterCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        value = metrics.get("val_loss")
        if value is not None:
            teacher = metrics.get("val_teacher_loss")
            message = f"Epoch {trainer.current_epoch + 1} | val_loss={float(value):.4f}"
            if teacher is not None:
                message += f" | val_teacher_loss={float(teacher):.4f}"
            print(message, flush=True)


class SoftIndivLightningModule(ResumableLightningMixin, pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.init_reproducibility(int(config["seed"]))
        self.model = SoftIndividualNegativeMoCo(config)

    def _step(self, batch: dict, train: bool):
        pos = batch["pos_wave"].float()
        neg = batch["neg_wave"].float()
        neg_items = batch["neg_waves"].float()
        valid = batch["neg_valid"].bool()
        target_spk_id = batch.get("target_spk_id")
        negative_spk_ids = batch.get("negative_spk_ids")
        if train and self.config.get("augment", {}).get("noise_std", 0.0):
            std = float(self.config["augment"]["noise_std"])
            rms = pos.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            pos = pos + torch.randn_like(pos) * std * rms
        query = self.model.student_embedding(pos, neg)
        with torch.no_grad():
            positive = self.model.momentum_embedding(pos, neg)
            negative = self.model.momentum_negative_embeddings(pos, neg_items, valid)
        moco_loss, logs = self.model.loss(
            query,
            positive,
            negative,
            valid,
            query_speaker_ids=target_spk_id,
            negative_speaker_ids=negative_spk_ids,
        )
        teacher_loss = self.model.teacher_loss(pos, neg)
        loss = moco_loss + self.model.teacher_loss_weight * teacher_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Soft_Indiv_MOCOCO Stage0 produced a non-finite loss: "
                f"moco={float(moco_loss.detach()):.6g}, teacher={float(teacher_loss.detach()):.6g}"
            )
        prefix = "train_" if train else "val_"
        bsz = int(pos.shape[0])
        self.log(prefix + "loss", loss, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(prefix + "moco_loss", moco_loss, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log(prefix + "teacher_loss", teacher_loss, on_epoch=True, sync_dist=True, batch_size=bsz)
        for key, value in logs.items():
            self.log(prefix + key, value, on_epoch=True, sync_dist=True, batch_size=bsz)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, train=True)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, train=False)

    def on_before_zero_grad(self, optimizer):
        self.model.update_momentum()

    def configure_optimizers(self):
        ocfg = self.config["optim"]
        scale = float(self.config["contrastive"].get("encoder_lr_scale", 0.03))
        encoder = [parameter for parameter in self.model.student.parameters() if parameter.requires_grad]
        parameters = [
            {"params": encoder, "lr": float(ocfg["lr"]) * scale},
            {"params": self.model.projection.parameters(), "lr": float(ocfg["lr"])},
        ]
        return torch.optim.AdamW(parameters, weight_decay=float(ocfg.get("weight_decay", 0.04)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Soft_Indiv_MOCOCO Stage0.")
    parser.add_argument("--config", default="configs/config_soft_indiv_moco.yaml")
    parser.add_argument("--resume", default=None, help="Lightning .ckpt to resume, including optimizer and RNG state.")
    return parser.parse_args()


def load_config(path: str, resume: str | None) -> dict:
    with open(path) as file:
        config = yaml.safe_load(file)
    config["seed"] = int(config.get("seed", 42))
    config["contrastive"]["initial_pn_ckpt"] = project_path(config["contrastive"]["initial_pn_ckpt"])
    config["train"]["log_dir"] = project_path(config["train"]["log_dir"])
    config["checkpoint"]["dir"] = project_path(config["checkpoint"]["dir"])
    configured_resume = resume or config["checkpoint"].get("resume")
    config["checkpoint"]["resume"] = project_path(configured_resume) if configured_resume else None
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.resume)
    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium")
    checkpoint_dir = Path(config["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_stage0_dataloaders(config)
    module = SoftIndivLightningModule(config)
    ddp = config.get("ddp", {})
    num_gpus = int(ddp.get("num_gpus", 1))
    use_ddp = torch.cuda.is_available() and num_gpus > 1
    strategy = DDPStrategy(find_unused_parameters=bool(ddp.get("find_unused_parameters", True))) if use_ddp else "auto"
    sweep_interval = int(config["checkpoint"].get("sweep_interval_epochs", 50))
    callbacks = [
        ExportPNEncoderCallback(
            checkpoint_dir,
            monitor=config["checkpoint"].get("monitor", "val_loss"),
            mode=config["checkpoint"].get("mode", "min"),
            every_n_epochs=sweep_interval,
        ),
        ResumeStateCallback(checkpoint_dir, "stage0", config["seed"], every_n_epochs=sweep_interval),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=config["checkpoint"].get("ckpt_name", "soft_indiv_moco_best"),
            monitor=config["checkpoint"].get("monitor", "val_loss"),
            mode=config["checkpoint"].get("mode", "min"),
            save_top_k=int(config["checkpoint"].get("save_best", 1)),
            save_last=bool(config["checkpoint"].get("save_last", True)),
        ),
        MetricPrinterCallback(),
        TQDMProgressBar(refresh_rate=1),
        LearningRateMonitor("epoch"),
    ]
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus if torch.cuda.is_available() else 1,
        strategy=strategy,
        max_epochs=int(config["train"]["num_epochs"]),
        accumulate_grad_batches=int(config["train"].get("accumulation_steps", 1)),
        precision=config["train"].get("precision", "bf16-mixed"),
        gradient_clip_val=float(config["train"].get("gradient_clip_val", 0.5)),
        log_every_n_steps=int(config["train"].get("log_interval", 20)),
        logger=TensorBoardLogger(config["train"]["log_dir"], name="lightning_logs", version="0"),
        callbacks=callbacks,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=config["checkpoint"]["resume"])


if __name__ == "__main__":
    main()
