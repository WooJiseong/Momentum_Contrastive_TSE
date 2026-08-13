from __future__ import annotations

import argparse
import os

os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/pn_mococo_numba_cache_{os.environ.get('USER', 'user')}")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from pn_mococo.paths import add_repo_paths, project_path

add_repo_paths()

from data.datasets import get_dataloaders
from pn_mococo.moco_encoder import ensure_channel
from pn_mococo.tfgridnet_model import (
    build_model_from_config,
    causal_forward,
    neg_si_sdr_loss,
    separation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PN_MOCOCO TFGridNet extractor.")
    parser.add_argument("--config", default="configs/config_tfgridnet_supervised.yaml")
    return parser.parse_args()


def parse_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return normalize_paths(cfg)


def normalize_paths(cfg: dict) -> dict:
    cfg = dict(cfg)
    if cfg.get("train", {}).get("log_dir"):
        cfg["train"]["log_dir"] = project_path(cfg["train"]["log_dir"])
    if cfg.get("checkpoint", {}).get("dir"):
        cfg["checkpoint"]["dir"] = project_path(cfg["checkpoint"]["dir"])
    if cfg.get("checkpoint", {}).get("resume"):
        cfg["checkpoint"]["resume"] = project_path(cfg["checkpoint"]["resume"])
    paths = cfg.setdefault("paths", {})
    for key in ("initial_model_ckpt", "encoder_override_ckpt"):
        if paths.get(key):
            paths[key] = project_path(paths[key])
    return cfg


def build_optimizer(params, cfg: dict):
    name = str(cfg.get("optimizer", "AdamW")).lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    eps = float(cfg.get("eps", 1e-8))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    raise ValueError(f"Unsupported optimizer '{cfg.get('optimizer')}'. Use AdamW or Adam.")


class MetricPrinterCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        pieces = [f"Epoch {trainer.current_epoch}"]
        for key in ("train_loss", "train_si_sdr", "train_si_sdri", "lr"):
            val = metrics.get(key)
            if val is not None:
                pieces.append(f"{key}={float(val):.4f}")
        print(" | ".join(pieces), flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        pieces = [f"Epoch {trainer.current_epoch}"]
        for key in ("val_loss", "val_si_sdr", "val_si_sdri", "val_snr", "val_snri"):
            val = metrics.get(key)
            if val is not None:
                pieces.append(f"{key}={float(val):.4f}")
        print(" | ".join(pieces), flush=True)


class LightningModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.model = build_model_from_config(config)
        dcfg = config.get("dataset", {})
        self.chunk_samples = int(config.get("inference", {}).get("chunk_samples", dcfg.get("sample_rate", 16000)))

    def on_train_epoch_start(self):
        self.model.to_train()

    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for key in ("mixture", "source", "pos_wave", "neg_wave"):
            if key not in batch:
                raise KeyError(
                    f"Batch is missing '{key}'. Set enroll.provider=v2b_pn_encoder so data.datasets emits pos_wave/neg_wave."
                )
        mixture = ensure_channel(batch["mixture"])
        target = batch["source"].float()
        pos = ensure_channel(batch["pos_wave"])
        neg = ensure_channel(batch["neg_wave"])
        est = causal_forward(self.model, mixture, pos, neg, self.chunk_samples)
        return est, target, mixture.squeeze(1)

    def training_step(self, batch, batch_idx):
        est, target, mixture = self._forward_batch(batch)
        loss = neg_si_sdr_loss(est, target)
        metrics = separation_metrics(est.detach(), target.detach(), mixture.detach())
        bsz = int(target.shape[0])
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_si_sdr", metrics["si_sdr"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_si_sdri", metrics["si_sdri"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        return loss

    def validation_step(self, batch, batch_idx):
        est, target, mixture = self._forward_batch(batch)
        loss = neg_si_sdr_loss(est, target)
        metrics = separation_metrics(est, target, mixture)
        bsz = int(target.shape[0])
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        for key, value in metrics.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        return loss

    def configure_optimizers(self):
        ocfg = self.config["optim"]
        groups = []
        if bool(self.config.get("trainable", {}).get("encoder", False)):
            groups.append({
                "params": [p for p in self.model.get_encoder_params() if p.requires_grad],
                "lr": float(ocfg.get("encoder_lr", ocfg.get("lr", 1e-4))),
            })
        if bool(self.config.get("trainable", {}).get("encoder_head", False)):
            groups.append({
                "params": [p for p in self.model.get_encoder_head_params() if p.requires_grad],
                "lr": float(ocfg.get("encoder_head_lr", ocfg.get("lr", 1e-4))),
            })
        if bool(self.config.get("trainable", {}).get("separator", True)):
            groups.append({
                "params": [p for p in self.model.get_main_params() if p.requires_grad],
                "lr": float(ocfg.get("separator_lr", ocfg.get("lr", 1e-4))),
            })
        groups = [g for g in groups if g["params"]]
        if not groups:
            raise ValueError("No trainable parameter groups were configured.")

        optimizer = build_optimizer(groups, ocfg)
        sch = self.config.get("scheduler", {})
        if sch.get("type") == "CosineAnnealingLR":
            warmup_epochs = int(sch.get("warmup_epochs", 0))

            def warmup_lambda(epoch):
                if warmup_epochs <= 0:
                    return 1.0
                return epoch / warmup_epochs if epoch < warmup_epochs else 1.0

            warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=int(sch.get("t_max", self.config["train"]["num_epochs"])),
                eta_min=float(sch.get("eta_min", 1e-6)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs]),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))

    def setup(self, stage=None):
        self.train_loader, self.val_loader = get_dataloaders(
            self.config,
            is_ddp=False,
            world_size=self.world_size,
            rank=self.rank,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main() -> None:
    args = parse_args()
    config = parse_config(args.config)
    pl.seed_everything(int(config.get("seed", 42)))
    torch.set_float32_matmul_precision("medium")

    os.makedirs(config["train"]["log_dir"], exist_ok=True)
    os.makedirs(config["checkpoint"]["dir"], exist_ok=True)

    model = LightningModule(config)
    data_module = DataModule(config)

    callbacks: list[Callback] = [
        MetricPrinterCallback(),
        TQDMProgressBar(refresh_rate=1),
        ModelCheckpoint(
            dirpath=config["checkpoint"]["dir"],
            filename=config["checkpoint"].get("ckpt_name", "tfgridnet_best"),
            save_top_k=int(config["checkpoint"].get("save_best", 1)),
            save_last=bool(config["checkpoint"].get("save_last", True)),
            verbose=bool(config["checkpoint"].get("verbose", True)),
            monitor=config["checkpoint"].get("monitor", "val_loss"),
            mode=config["checkpoint"].get("mode", "min"),
        ),
    ]

    if config["train"].get("log_lr", True):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    if config.get("early_stopping", {}).get("enabled", False):
        ecfg = config["early_stopping"]
        callbacks.append(EarlyStopping(
            monitor=ecfg.get("monitor", "val_loss"),
            patience=int(ecfg.get("patience", 20)),
            verbose=bool(ecfg.get("verbose", True)),
            mode=ecfg.get("mode", "min"),
            min_delta=float(ecfg.get("delta", 0.0)),
        ))

    tb_logger = TensorBoardLogger(
        save_dir=config["train"]["log_dir"],
        name="lightning_logs",
        version="0",
    )

    ddp = config.get("ddp", {})
    use_ddp = bool(ddp.get("use_ddp", False))
    num_gpus = int(ddp.get("num_gpus", torch.cuda.device_count()))
    num_nodes = int(ddp.get("num_nodes", 1))
    resume = config.get("checkpoint", {}).get("resume") or None

    common = dict(
        max_epochs=int(config["train"]["num_epochs"]),
        accumulate_grad_batches=int(config["train"]["accumulation_steps"]),
        callbacks=callbacks,
        default_root_dir=config["train"]["log_dir"],
        logger=tb_logger,
        log_every_n_steps=int(config["train"]["log_interval"]),
        precision=config["train"].get("precision", "bf16-mixed"),
        gradient_clip_val=float(config["train"].get("gradient_clip_val", 0.5)),
        detect_anomaly=bool(config["train"].get("detect_anomaly", False)),
        limit_train_batches=config["train"].get("limit_train_batches"),
        limit_val_batches=config["train"].get("limit_val_batches"),
        overfit_batches=float(config["train"].get("overfit_batches", 0.0)),
        enable_progress_bar=True,
    )
    if torch.cuda.is_available() and use_ddp and num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=bool(ddp.get("find_unused_parameters", False)),
            broadcast_buffers=bool(ddp.get("broadcast_buffers", False)),
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=num_gpus,
            num_nodes=num_nodes,
            strategy=strategy,
            **common,
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            **common,
        )

    trainer.fit(model, datamodule=data_module, ckpt_path=resume)


if __name__ == "__main__":
    main()
