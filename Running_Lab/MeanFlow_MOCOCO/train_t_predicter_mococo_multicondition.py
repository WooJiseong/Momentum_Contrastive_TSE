#!/usr/bin/env python3
"""Train a MOCOCO t-predictor on the same conditions used by evaluation.

The read-only PNFlowTSE trainer is reused for the model, optimizer, DDP and
checkpointing.  This file only supplies a balanced four-condition dataloader
and a more robust regression objective.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
PNFLOW_ROOT = PROJECT_ROOT.parent


def _load_trainer():
    from train_t_predicter_mococo import _load_trainer

    return _load_trainer()


class BalancedConditionDataset(Dataset):
    """Keep the four mixture/enrollment conditions balanced by index."""

    def __init__(self, datasets: list[Dataset], samples_per_condition: int):
        self.datasets = datasets
        self.samples_per_condition = int(samples_per_condition)
        self.length = self.samples_per_condition * len(self.datasets)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        condition = int(index) // self.samples_per_condition
        local_index = int(index) % self.samples_per_condition
        return self.datasets[condition][local_index]


def _make_inner(NoisyFlowTSEDataset, dcfg: dict, root: str, noise_dir: str,
                mixture_speakers: int, enroll_speakers: int, reproducable: bool):
    """Build the same online mixer as four-condition evaluation."""
    sr = int(dcfg["sample_rate"])
    data_root = root if os.path.isabs(root) else str(PNFLOW_ROOT / "data" / root)
    return NoisyFlowTSEDataset(
        root_dir=data_root,
        noise_dir=noise_dir,
        sample_rate=sr,
        wave_length=sr * int(dcfg["segment"]),
        pos_example_length=sr * int(dcfg["segment_aux"]),
        neg_example_length=sr * int(dcfg["segment_aux"]),
        source_num=int(mixture_speakers),
        enroll_num=int(enroll_speakers),
        active_num=tuple(dcfg.get("active_num", (-1, 1, -1))),
        snr_db_range=list(dcfg.get("snr_db_range", []) or []),
        special_spk=tuple(dcfg.get("special_spk", ()) or ()),
        partial_range=tuple(dcfg.get("partial_range", (0.33, 0.66))),
        neg_partial_range=tuple(dcfg.get("neg_partial_range", (0.33, 1.0))),
        clean_enroll=bool(dcfg.get("clean_enroll", False)),
        reproducable=reproducable,
        filling_pattern=dcfg.get("filling_pattern", "repeat"),
        enroll_exclude_mixture_utt=bool(dcfg.get("enroll_exclude_mixture_utt", False)),
        min_utts_per_speaker=int(dcfg.get("min_utts_per_speaker", 2)),
    )


def _build_condition_loaders(config: dict, is_ddp=False, world_size=1, rank=0):
    """Return train/validation loaders balanced over 2/2, 2/3, 3/2 and 3/3."""
    import random
    import numpy as np
    from data.datasets import PNLibriMixInformed, _ddp_rank
    from data.datasets import _noise_dir
    from pndata.online_mixer import NoisyFlowTSEDataset

    dcfg = config["dataset"]
    tcfg = config["train"]
    conditions = [tuple(map(int, item)) for item in dcfg["conditions"]]
    has_noise = bool(dcfg.get("snr_db_range", []))
    train_noise = _noise_dir(dcfg.get("train_noise", "wham_noise/tr"), has_noise)
    val_noise = _noise_dir(dcfg.get("val_noise", "wham_noise/cv"), has_noise)
    train_roots = list(dcfg["train_roots"])
    val_root = dcfg["val_root"]
    emit_posneg = True

    train_sets = []
    val_sets = []
    train_samples_per_condition = max(
        1, int(dcfg["samples_per_epoch"]) // len(conditions)
    )
    for mixture_speakers, enroll_speakers in conditions:
        train_cfg = copy.deepcopy(dcfg)
        train_cfg["source_num"] = mixture_speakers
        train_cfg["enroll_num"] = enroll_speakers
        train_inners = [
            _make_inner(NoisyFlowTSEDataset, train_cfg, root, train_noise,
                        mixture_speakers, enroll_speakers, reproducable=False)
            for root in train_roots
        ]
        train_sets.append(PNLibriMixInformed(
            train_inners, train_cfg, train_samples_per_condition,
            train=True, emit_posneg=emit_posneg,
        ))

        val_cfg = copy.deepcopy(dcfg)
        val_cfg["source_num"] = mixture_speakers
        val_cfg["enroll_num"] = enroll_speakers
        val_inner = _make_inner(
            NoisyFlowTSEDataset, val_cfg, val_root, val_noise,
            mixture_speakers, enroll_speakers, reproducable=True,
        )
        val_sets.append(PNLibriMixInformed(
            [val_inner], val_cfg, int(dcfg["val_size"]),
            train=False, emit_posneg=emit_posneg,
        ))

    if int(tcfg["num_workers"]) == 0:
        base = (torch.initial_seed() + 100003 * _ddp_rank()) % (2 ** 31 - 1)
        random.seed(base)
        np.random.seed(base)

    train_ds = BalancedConditionDataset(train_sets, train_samples_per_condition)
    val_ds = BalancedConditionDataset(val_sets, int(dcfg["val_size"]))
    workers = int(tcfg["num_workers"])
    worker_init = getattr(__import__("data.datasets", fromlist=["_worker_init"]), "_worker_init")
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=True, worker_init_fn=worker_init,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False,
        num_workers=min(2, workers), pin_memory=True, worker_init_fn=worker_init,
        persistent_workers=False,
    )
    return train_loader, val_loader


def _regression_terms(pred: torch.Tensor, target: torch.Tensor, cfg: dict):
    error = pred - target
    mse = error.square().mean()
    smooth_l1 = F.smooth_l1_loss(pred, target, beta=float(cfg.get("huber_beta", 0.05)))
    bias = error.mean().square()
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    covariance = (pred_centered * target_centered).mean()
    denom = torch.sqrt(pred_centered.square().mean() * target_centered.square().mean()).clamp_min(1e-6)
    correlation = (covariance / denom).clamp(-1.0, 1.0)
    corr_loss = 1.0 - correlation
    loss = (
        float(cfg.get("mse_weight", 1.0)) * mse
        + float(cfg.get("smooth_l1_weight", 0.25)) * smooth_l1
        + float(cfg.get("bias_weight", 0.10)) * bias
        + float(cfg.get("correlation_weight", 0.05)) * corr_loss
    )
    return loss, mse, smooth_l1, bias, correlation


def install_improved_module(trainer):
    BaseLightningModule = trainer.LightningModule
    CondOTProbPath = trainer.CondOTProbPath

    class ImprovedLightningModule(BaseLightningModule):
        def _step(self, batch, train: bool):
            source = batch["source_rescaled"]
            background = batch["background_rescaled"]
            batch_size = source.size(0)
            true_alpha = batch["mixing_ratio"].reshape(batch_size).to(source.device)

            # Mostly use the real mixture seen at inference.  Keep a small
            # synthetic branch so the predictor also sees the full [0, 1] path.
            if train:
                path = CondOTProbPath()
                synthetic_alpha = torch.rand((batch_size,), device=source.device)
                synthetic_mixture = path.sample(
                    t=synthetic_alpha, x_0=background, x_1=source
                ).x_t
                use_synthetic = torch.rand((batch_size,), device=source.device) < float(
                    self.config.get("data_mix", {}).get("synthetic_probability", 0.25)
                )
                mixture = torch.where(use_synthetic[:, None], synthetic_mixture, batch["mixture"])
                alpha = torch.where(use_synthetic, synthetic_alpha, true_alpha)
                aug = self.config["train"].get("spectral_aug", False)
            else:
                mixture = batch["mixture"]
                alpha = true_alpha
                aug = False

            enroll_emb = self._enroll_emb(batch)
            alpha_hat = self.model(mixture, enroll_emb, aug=aug)
            terms = _regression_terms(alpha_hat, alpha, self.config.get("loss", {}))
            loss, mse, smooth_l1, bias, correlation = terms
            prefix = "train" if train else "val"
            self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{prefix}_mse", mse, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)
            self.log(f"{prefix}_mae", (alpha_hat - alpha).abs().mean(), on_step=False,
                     on_epoch=True, prog_bar=not train, sync_dist=True, batch_size=batch_size)
            self.log(f"{prefix}_rmse", mse.sqrt(), on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)
            self.log(f"{prefix}_bias", (alpha_hat - alpha).mean(), on_step=False,
                     on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{prefix}_corr", correlation, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)
            self.log(f"{prefix}_smooth_l1", smooth_l1, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)
            self.log(f"{prefix}_bias_loss", bias, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)
            return loss

        def training_step(self, batch, batch_idx):
            return self._step(batch, train=True)

        def validation_step(self, batch, batch_idx):
            return self._step(batch, train=False)

    trainer.LightningModule = ImprovedLightningModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # Lightning's DDP launcher re-executes this file with the config argument
    # but does not preserve custom CLI arguments.  Keep the checkpoint in the
    # environment as a second, DDP-safe transport path.
    parser.add_argument("--stage0-ckpt", default=os.environ.get("STAGE0_CKPT"))
    args = parser.parse_args()
    if not args.stage0_ckpt:
        parser.error("the following arguments are required: --stage0-ckpt (or STAGE0_CKPT)")
    config_path = Path(args.config).expanduser().resolve()
    stage0_path = Path(args.stage0_ckpt).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not stage0_path.is_file():
        raise FileNotFoundError(stage0_path)
    os.environ["STAGE0_CKPT"] = str(stage0_path)

    trainer = _load_trainer()
    install_improved_module(trainer)
    original_parse_config = trainer.parse_config

    def parse_config_with_stage0(path):
        config = original_parse_config(path)
        config.setdefault("paths", {})["pn_ckpt"] = str(stage0_path)
        if os.environ.get("TPRED_SMOKE") == "1":
            config["train"]["num_epochs"] = 1
            config["train"]["limit_train_batches"] = 2
            config["train"]["limit_val_batches"] = 2
            config["train"]["num_workers"] = 0
        return config

    trainer.parse_config = parse_config_with_stage0
    trainer.get_dataloaders = _build_condition_loaders
    sys.argv = [sys.argv[0], "--config", str(config_path)]
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/soft_best_tpred_multicondition_numba_cache")
    trainer.main()


if __name__ == "__main__":
    main()
