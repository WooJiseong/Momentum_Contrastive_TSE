from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/pn_indiv_mococo_numba_cache_{os.environ.get('USER', 'user')}")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from pn_indiv_mococo.paths import add_repo_paths, project_path

add_repo_paths()

from pn_indiv_mococo.data import get_dataloaders
from pn_indiv_mococo.moco_encoder import IndividualNegativeMoCo


class ExportPNEncoderCallback(Callback):
    def __init__(self, export_dir: Path, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.export_dir = Path(export_dir)
        self.monitor = monitor
        self.mode = mode
        self.best: float | None = None

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    @staticmethod
    def _export(pl_module: pl.LightningModule, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            k: v.detach().cpu()
            for k, v in pl_module.model.student.state_dict().items()
        }
        torch.save(
            {
                "state_dict": state,
                "meta": {
                    "negative_weighting": "1/n",
                    "individual_negative_rows": True,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_indiv_moco.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["contrastive"]["initial_pn_ckpt"] = project_path(cfg["contrastive"]["initial_pn_ckpt"])
    cfg["train"]["log_dir"] = project_path(cfg["train"]["log_dir"])
    cfg["checkpoint"]["dir"] = project_path(cfg["checkpoint"]["dir"])
    if cfg.get("checkpoint", {}).get("resume"):
        cfg["checkpoint"]["resume"] = project_path(cfg["checkpoint"]["resume"])
    pl.seed_everything(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision("medium")

    class Lit(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = IndividualNegativeMoCo(cfg)
            self.save_hyperparameters(cfg)

        def _step(self, batch, train):
            pos = batch["pos_wave"].float()
            neg = batch["neg_wave"].float()
            neg_items = batch["neg_waves"].float()
            valid = batch["neg_valid"].bool()
            target_spk_id = batch.get("target_spk_id")
            negative_spk_ids = batch.get("negative_spk_ids")
            if train and cfg.get("augment", {}).get("noise_std", 0.0):
                std = float(cfg["augment"]["noise_std"])
                pos = pos + torch.randn_like(pos) * std * pos.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            query = self.model.student_embedding(pos, neg)
            with torch.no_grad():
                positive = self.model.momentum_embedding(pos, neg)
                negative = self.model.momentum_negative_embeddings(pos, neg_items, valid)
            loss, logs = self.model.loss(
                query,
                positive,
                negative,
                valid,
                query_speaker_ids=target_spk_id,
                negative_speaker_ids=negative_spk_ids,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("PN_Indiv_MOCOCO Stage0 produced a non-finite loss.")
            prefix = "train_" if train else "val_"
            self.log(prefix + "loss", loss, on_epoch=True, sync_dist=True, batch_size=pos.shape[0])
            for key, value in logs.items():
                self.log(prefix + key, value, on_epoch=True, sync_dist=True, batch_size=pos.shape[0])
            return loss

        def training_step(self, batch, batch_idx):
            return self._step(batch, True)

        def validation_step(self, batch, batch_idx):
            return self._step(batch, False)

        def on_before_zero_grad(self, optimizer):
            self.model.update_momentum()

        def configure_optimizers(self):
            ocfg = cfg["optim"]
            scale = float(cfg["contrastive"].get("encoder_lr_scale", 0.03))
            enc = [p for p in self.model.student.parameters() if p.requires_grad]
            params = [{"params": enc, "lr": float(ocfg["lr"]) * scale}, {"params": self.model.projection.parameters(), "lr": float(ocfg["lr"])}]
            return torch.optim.AdamW(params, weight_decay=float(ocfg.get("weight_decay", 0.04)))

    train_loader, val_loader = get_dataloaders(cfg)
    module = Lit()
    out = Path(cfg["checkpoint"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    ddp = cfg.get("ddp", {})
    num_gpus = int(ddp.get("num_gpus", 1))
    use_ddp = torch.cuda.is_available() and num_gpus > 1
    strategy = (
        DDPStrategy(find_unused_parameters=bool(ddp.get("find_unused_parameters", True)))
        if use_ddp
        else "auto"
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus if torch.cuda.is_available() else 1,
        strategy=strategy,
        max_epochs=int(cfg["train"]["num_epochs"]),
        accumulate_grad_batches=int(cfg["train"].get("accumulation_steps", 1)),
        precision=cfg["train"].get("precision", "bf16-mixed"),
        gradient_clip_val=float(cfg["train"].get("gradient_clip_val", 0.5)),
        log_every_n_steps=int(cfg["train"].get("log_interval", 20)),
        logger=TensorBoardLogger(cfg["train"]["log_dir"], name="lightning_logs"),
        callbacks=[
            ExportPNEncoderCallback(out, monitor=cfg["checkpoint"].get("monitor", "val_loss"), mode=cfg["checkpoint"].get("mode", "min")),
            ModelCheckpoint(
                dirpath=out,
                filename=cfg["checkpoint"].get("ckpt_name", "indiv_moco_best"),
                monitor=cfg["checkpoint"].get("monitor", "val_loss"),
                mode=cfg["checkpoint"].get("mode", "min"),
                save_top_k=int(cfg["checkpoint"].get("save_best", 1)),
                save_last=bool(cfg["checkpoint"].get("save_last", True)),
            ),
            TQDMProgressBar(refresh_rate=1),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=cfg.get("checkpoint", {}).get("resume") or None)


if __name__ == "__main__":
    main()
