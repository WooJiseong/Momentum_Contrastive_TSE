#!/usr/bin/env python3
"""Run MeanFlow Stage1 with DSnOS Stage0 and an optional dual-DSO loss."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parents[1]
SNIPPET_DIR = ROOT / "Base" / "Code_Snippet"
MEANFLOW_ENTRY = ROOT / "Running_Lab" / "MeanFlow_MOCOCO" / "train_meanflow_mococo.py"
DUALDSO_PATH = SNIPPET_DIR / "dualdso_loss.py"

# Install this before SIA/ESPnet imports, as required by the speedup module.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    f"/tmp/dsnos_moco_stage1_meanflow_{os.environ.get('USER', 'user')}",
)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SNIPPET_DIR))
import speedups  # noqa: F401,E402


def _install_dualdso() -> None:
    spec = importlib.util.spec_from_file_location("meanflow", DUALDSO_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {DUALDSO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["meanflow"] = module
    spec.loader.exec_module(module)


def _patch_sia_for_dualdso(meanflow_module) -> None:
    """Remove unsynchronized rank-0 media I/O while keeping scalar metrics."""
    original_loader = meanflow_module._load_sia_trainer

    def load_sia_trainer():
        trainer_module = original_loader()
        lightning_module = trainer_module.LightningModule
        if not getattr(lightning_module, "_dsnos_dualdso_scalar_only", False):
            def scalar_only_validation_epoch_end(self):
                self._val_samples = None

            lightning_module.on_validation_epoch_end = scalar_only_validation_epoch_end
            lightning_module._dsnos_dualdso_scalar_only = True
            print(
                "[DSnOS DualDSO] disabled rank-0 validation media hook; "
                "scalar validation metrics remain enabled",
                flush=True,
            )
        metric_printer = getattr(trainer_module, "MetricPrinterCallback", None)
        if metric_printer is not None and not getattr(
            metric_printer,
            "_dsnos_dualdso_no_epoch_reduce",
            False,
        ):
            # The callback only prints callback_metrics.  Accessing that
            # property forces Lightning to reduce every epoch metric in
            # callback order, which is fragile when DualDSO adds diagnostics.
            # The self.log(..., sync_dist=True) calls remain active for TB.
            metric_printer.on_train_epoch_end = lambda self, trainer, pl_module: None
            metric_printer.on_validation_epoch_end = lambda self, trainer, pl_module: None
            metric_printer._dsnos_dualdso_no_epoch_reduce = True
            print(
                "[DSnOS DualDSO] disabled epoch metric printer; "
                "distributed scalar logging remains enabled",
                flush=True,
            )
            class NoOpMetricPrinter(trainer_module.Callback):
                def __init__(self, *args, **kwargs):
                    super().__init__()

            trainer_module.MetricPrinterCallback = NoOpMetricPrinter
            trainer_module.main.__globals__["MetricPrinterCallback"] = (
                NoOpMetricPrinter
            )
        progress_bar = getattr(trainer_module, "TQDMProgressBar", None)
        if progress_bar is not None and not getattr(
            progress_bar,
            "_dsnos_dualdso_no_epoch_reduce",
            False,
        ):
            class NoOpProgressBar(trainer_module.Callback):
                def __init__(self, *args, **kwargs):
                    super().__init__()

            trainer_module.TQDMProgressBar = NoOpProgressBar
            trainer_module.TQDMProgressBar._dsnos_dualdso_no_epoch_reduce = True
            trainer_module.main.__globals__["TQDMProgressBar"] = (
                NoOpProgressBar
            )
            print(
                "[DSnOS DualDSO] disabled progress-bar metric reduction; "
                "distributed scalar logging remains enabled",
                flush=True,
            )
            trainer_module.TQDMProgressBar = NoOpProgressBar
        if os.environ.get("DSNOS_DISABLE_CHECKPOINT") == "1":
            class NoOpModelCheckpoint(trainer_module.Callback):
                def __init__(self, *args, **kwargs):
                    super().__init__()

            trainer_module.ModelCheckpoint = NoOpModelCheckpoint
            trainer_module.main.__globals__["ModelCheckpoint"] = (
                NoOpModelCheckpoint
            )
            print(
                "[DSnOS DualDSO] checkpoint callback disabled for smoke test",
                flush=True,
            )
        if not getattr(trainer_module, "_dsnos_dualdso_trainer_patched", False):
            original_trainer = trainer_module.pl.Trainer

            class DSnOSTrainer(original_trainer):
                def __init__(self, *args, **kwargs):
                    # All callbacks in the reference trainer inspect
                    # callback/progress metrics at epoch end.  For DualDSO,
                    # retain scalar self.log records but remove those
                    # callback-side reductions from the DDP graph.
                    kwargs["callbacks"] = []
                    # Avoid Lightning's progress-bar metric snapshot, which
                    # performs an implicit DDP reduction at epoch end.
                    kwargs["enable_progress_bar"] = False
                    super().__init__(*args, **kwargs)

            trainer_module.pl.Trainer = DSnOSTrainer
            trainer_module._dsnos_dualdso_trainer_patched = True
            print(
                "[DSnOS DualDSO] removed epoch-end print/progress callbacks; "
                "distributed scalar logging remains enabled",
                flush=True,
            )
        import pytorch_lightning.trainer.call as lightning_call

        if not getattr(
            lightning_call,
            "_dsnos_dualdso_callback_hooks_patched",
            False,
        ):
            original_callback_hooks = lightning_call._call_callback_hooks

            def skip_epoch_metric_callbacks(
                trainer,
                hook_name,
                *args,
                **kwargs,
            ):
                if hook_name in {
                    "on_train_epoch_end",
                    "on_validation_epoch_end",
                } and kwargs.get("monitoring_callbacks") is False:
                    return None
                return original_callback_hooks(
                    trainer,
                    hook_name,
                    *args,
                    **kwargs,
                )

            lightning_call._call_callback_hooks = skip_epoch_metric_callbacks
            lightning_call._dsnos_dualdso_callback_hooks_patched = True
            print(
                "[DSnOS DualDSO] skipped Lightning epoch-end callback hooks; "
                "distributed scalar logging remains enabled",
                flush=True,
            )
        if not getattr(
            trainer_module.pl.Trainer,
            "_dsnos_dualdso_fit_callbacks_patched",
            False,
        ):
            original_fit = trainer_module.pl.Trainer.fit

            def fit_without_epoch_callbacks(self, *args, **kwargs):
                if os.environ.get("DSNOS_DISABLE_CHECKPOINT") == "1":
                    self.callbacks = []
                self.enable_progress_bar = False
                return original_fit(self, *args, **kwargs)

            trainer_module.pl.Trainer.fit = fit_without_epoch_callbacks
            trainer_module.pl.Trainer._dsnos_dualdso_fit_callbacks_patched = True
            print(
                "[DSnOS DualDSO] cleared Trainer callbacks before fit; "
                "distributed scalar logging remains enabled",
                flush=True,
            )
        return trainer_module

    meanflow_module._load_sia_trainer = load_sia_trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage0-ckpt", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--loss-impl",
        choices=("reference", "dualdso"),
        default="reference",
    )
    args = parser.parse_args()

    # Lightning's subprocess launcher re-executes this script after the
    # MeanFlow wrapper has reduced sys.argv to only --config.  Preserve the
    # loss implementation explicitly so every DDP rank imports the same
    # objective and applies the same runtime patches.
    propagated_loss_impl = os.environ.get("DSNOS_LOSS_IMPL")
    if propagated_loss_impl in {"reference", "dualdso"}:
        args.loss_impl = propagated_loss_impl
    os.environ["DSNOS_LOSS_IMPL"] = args.loss_impl

    if args.loss_impl == "dualdso":
        _install_dualdso()

    spec = importlib.util.spec_from_file_location(
        "dsnos_meanflow_mococo_entry", MEANFLOW_ENTRY
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {MEANFLOW_ENTRY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if args.loss_impl == "dualdso":
        _patch_sia_for_dualdso(module)

    forwarded = [sys.argv[0], "--config", str(Path(args.config).resolve())]
    if args.stage0_ckpt:
        forwarded.extend(["--stage0-ckpt", str(Path(args.stage0_ckpt).resolve())])
    if args.resume:
        forwarded.extend(["--resume", str(Path(args.resume).resolve())])
    sys.argv = forwarded
    module.main()


if __name__ == "__main__":
    main()
