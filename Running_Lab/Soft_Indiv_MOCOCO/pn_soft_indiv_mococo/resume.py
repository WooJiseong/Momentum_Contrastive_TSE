from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class ResumableLightningMixin:
    """Store seed and RNG in every Lightning checkpoint and restore it on fit start."""

    def init_reproducibility(self, seed: int) -> None:
        self._experiment_seed = int(seed)
        self._pending_rng_state: dict | None = None

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        parent = getattr(super(), "on_save_checkpoint", None)
        if parent is not None:
            parent(checkpoint)
        checkpoint["soft_indiv_reproducibility"] = {
            "seed": self._experiment_seed,
            "rng_state": capture_rng_state(),
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        parent = getattr(super(), "on_load_checkpoint", None)
        if parent is not None:
            parent(checkpoint)
        state = checkpoint.get("soft_indiv_reproducibility")
        if state:
            self._pending_rng_state = state.get("rng_state")

    def on_fit_start(self) -> None:
        parent = getattr(super(), "on_fit_start", None)
        if parent is not None:
            parent()
        restore_rng_state(self._pending_rng_state)
        self._pending_rng_state = None


class ResumeStateCallback(Callback):
    """Keep a human-inspectable optimizer/RNG snapshot beside Lightning checkpoints."""

    def __init__(self, checkpoint_dir: str | Path, phase: str, seed: int, every_n_epochs: int = 0):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.phase = phase
        self.seed = int(seed)
        self.every_n_epochs = int(every_n_epochs)

    def _save(self, trainer, label: str) -> None:
        if not trainer.is_global_zero:
            return
        state_dir = self.checkpoint_dir.parent / "resume_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": self.phase,
            "label": label,
            "seed": self.seed,
            "epoch_completed": int(trainer.current_epoch) + 1,
            "global_step": int(trainer.global_step),
            "rng_state": capture_rng_state(),
            "optimizer_states": [optimizer.state_dict() for optimizer in trainer.optimizers],
        }
        path = state_dir / f"{self.phase}_{label}_state.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._save(trainer, "last")
        completed = int(trainer.current_epoch) + 1
        if self.every_n_epochs > 0 and completed % self.every_n_epochs == 0:
            self._save(trainer, f"epoch_{completed:03d}")
