from __future__ import annotations

import argparse
import os
from pathlib import Path

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

from pn_mococo.speaker_data import get_speaker_aware_dataloaders as get_dataloaders
from pn_mococo.moco_encoder import MomentumContrastivePNLearner, ensure_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PN_MOCOCO momentum contrastive PN encoder.")
    parser.add_argument("--config", default="configs/config_moco_encoder.yaml")
    return parser.parse_args()


def parse_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return normalize_paths(cfg)


def normalize_paths(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg.setdefault("paths", {})
    if cfg.get("train", {}).get("log_dir"):
        cfg["train"]["log_dir"] = project_path(cfg["train"]["log_dir"])
    if cfg.get("checkpoint", {}).get("dir"):
        cfg["checkpoint"]["dir"] = project_path(cfg["checkpoint"]["dir"])
    if cfg.get("checkpoint", {}).get("resume"):
        cfg["checkpoint"]["resume"] = project_path(cfg["checkpoint"]["resume"])
    if cfg.get("paths", {}).get("initial_pn_ckpt"):
        cfg["paths"]["initial_pn_ckpt"] = project_path(cfg["paths"]["initial_pn_ckpt"])
    if cfg.get("contrastive", {}).get("initial_pn_ckpt"):
        cfg["contrastive"]["initial_pn_ckpt"] = project_path(cfg["contrastive"]["initial_pn_ckpt"])
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


class ExportPNEncoderCallback(Callback):
    """Export student PN weights in the original proposed-monaural-compatible format."""

    def __init__(self, export_dir: str, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.export_dir = Path(export_dir)
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    @staticmethod
    def _export(pl_module: "LightningModule", path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            k: v.detach().cpu()
            for k, v in pl_module.learner.student_encoder.state_dict().items()
        }
        torch.save(
            {
                "state_dict": state,
                "meta": {
                    "format": "PNEncodePath encoder.* + encoder_head.*",
                    "source": "PN_MOCOCO momentum contrastive student encoder",
                    "global_step": int(pl_module.global_step),
                    "epoch": int(pl_module.current_epoch),
                },
            },
            path,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._export(pl_module, self.export_dir / "pn_encoder_last.pt")
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric.detach().cpu())
        if self._is_better(value):
            self.best = value
            self._export(pl_module, self.export_dir / "pn_encoder_best.pt")

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._export(pl_module, self.export_dir / "pn_encoder_last.pt")


class MetricPrinterCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        pieces = [f"Epoch {trainer.current_epoch}"]
        for key in (
            "train_loss",
            "train_acc",
            "train_pos_sim",
            "train_neg_sim_max",
            "train_queue_len",
            "train_queue_masked_ratio",
            "train_ema_m",
        ):
            val = metrics.get(key)
            if val is not None:
                pieces.append(f"{key}={float(val):.4f}")
        print(" | ".join(pieces), flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val = metrics.get("val_loss")
        if val is not None:
            pieces = [f"Epoch {trainer.current_epoch}", f"Val Loss={float(val):.4f}"]
            acc = metrics.get("val_acc")
            if acc is not None:
                pieces.append(f"Val Acc={float(acc):.4f}")
            print(" | ".join(pieces), flush=True)


class LightningModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.learner = MomentumContrastivePNLearner(config)
        self._last_train_bsz = 1

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Make checkpoints created before speaker-aware queue metadata resumable."""
        state_dict = checkpoint.setdefault("state_dict", {})
        key = "learner.queue_speaker_ids"
        expected = self.learner.queue_speaker_ids.detach().cpu()
        saved = state_dict.get(key)
        if saved is None or tuple(saved.shape) != tuple(expected.shape):
            state_dict[key] = expected.clone()

    def _augment_wave(self, x: torch.Tensor, strong: bool) -> torch.Tensor:
        acfg = self.config.get("augment", {}) or {}
        y = ensure_channel(x).float().clone()
        bsz, _, n = y.shape
        device = y.device

        gain_min, gain_max = acfg.get("gain_range", [0.8, 1.25])
        gains = torch.empty(bsz, 1, 1, device=device).uniform_(float(gain_min), float(gain_max))
        y = y * gains

        noise_std = float(acfg.get("noise_std_strong" if strong else "noise_std_weak", 0.0))
        if noise_std > 0:
            rms = y.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            y = y + torch.randn_like(y) * rms * noise_std

        mask_prob = float(acfg.get("mask_prob_strong" if strong else "mask_prob_weak", 0.0))
        max_ratio = float(acfg.get("mask_max_ratio", 0.25))
        if mask_prob > 0 and max_ratio > 0:
            max_len = max(1, int(n * max_ratio))
            for i in range(bsz):
                if torch.rand((), device=device) < mask_prob:
                    length = int(torch.randint(1, max_len + 1, (), device=device).item())
                    start = int(torch.randint(0, max(1, n - length + 1), (), device=device).item())
                    y[i, :, start:start + length] = 0.0

        shift_ratio = float(acfg.get("shift_max_ratio", 0.0))
        if shift_ratio > 0:
            max_shift = int(n * shift_ratio)
            if max_shift > 0:
                shifted = []
                for i in range(bsz):
                    shift = int(torch.randint(-max_shift, max_shift + 1, (), device=device).item())
                    shifted.append(torch.roll(y[i], shifts=shift, dims=-1))
                y = torch.stack(shifted, dim=0)

        return y

    def _pairs(self, batch: dict, train: bool):
        ccfg = self.config.get("contrastive", {})
        positive_key = ccfg.get("positive_key", "source")
        if positive_key == "source" and "source" in batch:
            pos_key = ensure_channel(batch["source"])
        elif positive_key == "pos":
            pos_key = ensure_channel(batch["pos_wave"])
        else:
            raise ValueError("contrastive.positive_key must be 'source' or 'pos'")

        q_pos = ensure_channel(batch["pos_wave"])
        q_neg = ensure_channel(batch["neg_wave"])
        neg_key = q_neg

        if train:
            q_pos = self._augment_wave(q_pos, strong=bool(ccfg.get("query_strong_aug", True)))
            q_neg = self._augment_wave(q_neg, strong=bool(ccfg.get("query_strong_aug", True)))
            pos_key = self._augment_wave(pos_key, strong=bool(ccfg.get("key_strong_aug", False)))
            neg_key = self._augment_wave(neg_key, strong=bool(ccfg.get("key_strong_aug", False)))

        negative_pos = [neg_key]
        negative_neg = [pos_key]
        if bool(ccfg.get("use_background_negative", False)) and "background" in batch:
            bg = ensure_channel(batch["background"])
            if train:
                bg = self._augment_wave(bg, strong=bool(ccfg.get("key_strong_aug", False)))
            negative_pos.append(bg)
            negative_neg.append(pos_key)

        negative_pairs = [
            self.learner.momentum_embedding(p, n)
            for p, n in zip(negative_pos, negative_neg)
        ]
        target_spk_id = None
        negative_speaker_ids = None
        if bool(ccfg.get("speaker_aware_queue", False)):
            if "target_spk_id" not in batch:
                raise KeyError(
                    "speaker_aware_queue is enabled but the batch has no target_spk_id."
                )
            target_spk_id = batch["target_spk_id"].long().reshape(-1)
            negative_speaker_ids = [target_spk_id for _ in negative_pairs]
        return (
            q_pos,
            q_neg,
            pos_key,
            neg_key,
            negative_pairs,
            target_spk_id,
            negative_speaker_ids,
        )

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
        return self.learner.contrastive_loss(
            query=query,
            positive_key=positive,
            negative_keys=negative_keys,
            use_queue=train and bool(self.config.get("contrastive", {}).get("use_queue", True)),
            update_queue=train,
            query_speaker_ids=target_spk_id,
            negative_speaker_ids=negative_speaker_ids,
        )

    def training_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch, train=True)
        bsz = int(ensure_channel(batch["pos_wave"]).shape[0])
        self._last_train_bsz = bsz
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_acc", logs["acc"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_pos_sim", logs["pos_sim"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_neg_sim", logs["neg_sim"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_neg_sim_max", logs["neg_sim_max"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_queue_len", logs["queue_len"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("train_queue_masked_ratio", logs["queue_masked_ratio"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch, train=False)
        bsz = int(ensure_channel(batch["pos_wave"]).shape[0])
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("val_acc", logs["acc"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("val_pos_sim", logs["pos_sim"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        self.log("val_neg_sim_max", logs["neg_sim_max"], on_step=False, on_epoch=True, sync_dist=True, batch_size=bsz)
        return loss

    def on_before_zero_grad(self, optimizer):
        total = int(getattr(self.trainer, "estimated_stepping_batches", 1) or 1)
        momentum = self.learner.ema_momentum(int(self.global_step), total)
        self.learner.update_momentum_encoder(momentum)
        self.log(
            "train_ema_m",
            torch.tensor(momentum, device=self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self._last_train_bsz,
        )

    def configure_optimizers(self):
        ocfg = self.config["optim"]
        base_lr = float(ocfg.get("lr", 1e-4))
        enc_scale = float(self.config.get("contrastive", {}).get("encoder_lr_scale", 0.03))
        enc_params = [p for p in self.learner.student_encoder.parameters() if p.requires_grad]
        head_params = list(self.learner.student_head.parameters())
        param_groups = []
        if enc_params:
            param_groups.append({"params": enc_params, "lr": base_lr * enc_scale})
        param_groups.append({"params": head_params, "lr": base_lr})
        optimizer = build_optimizer(param_groups, {**ocfg, "lr": base_lr})

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
        ExportPNEncoderCallback(
            export_dir=config["checkpoint"]["dir"],
            monitor=config["checkpoint"].get("monitor", "val_loss"),
            mode=config["checkpoint"].get("mode", "min"),
        ),
        ModelCheckpoint(
            dirpath=config["checkpoint"]["dir"],
            filename=config["checkpoint"].get("ckpt_name", "moco_best"),
            save_top_k=int(config["checkpoint"].get("save_best", 1)),
            save_last=bool(config["checkpoint"].get("save_last", True)),
            verbose=bool(config["checkpoint"].get("verbose", True)),
            monitor=config["checkpoint"].get("monitor", "val_loss"),
            mode=config["checkpoint"].get("mode", "min"),
        ),
    ]

    if config.get("early_stopping", {}).get("enabled", False):
        ecfg = config["early_stopping"]
        callbacks.append(EarlyStopping(
            monitor=ecfg.get("monitor", "val_loss"),
            patience=int(ecfg.get("patience", 20)),
            verbose=bool(ecfg.get("verbose", True)),
            mode=ecfg.get("mode", "min"),
            min_delta=float(ecfg.get("delta", 0.0)),
        ))
    if config["train"].get("log_lr", True):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

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
