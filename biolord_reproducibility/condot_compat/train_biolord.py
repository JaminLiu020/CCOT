from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import biolord
import numpy as np
import torch


def _patch_biolord_training_step_if_needed() -> None:
    """Patch biolord 0.0.3 training_step for compatibility with newer Lightning.

    In some environments, biolordTrainingPlan.training_step performs manual
    optimization but returns None. Lightning then checks torch.isnan(loss),
    causing TypeError when loss is None.
    """
    try:
        from biolord._train import biolordTrainingPlan
    except Exception:
        return

    if getattr(biolordTrainingPlan, "_condot_compat_patched", False):
        return

    original_training_step = biolordTrainingPlan.training_step

    def _patched_training_step(self, batch, *args, **kwargs):
        out = original_training_step(self, batch)
        if out is None:
            try:
                device = next(self.module.parameters()).device
            except Exception:
                device = "cpu"
            return torch.tensor(0.0, device=device)
        return out

    biolordTrainingPlan.training_step = _patched_training_step
    biolordTrainingPlan._condot_compat_patched = True


def _patch_scvi_savecheckpoint_if_needed() -> None:
    """Patch scvi SaveCheckpoint callback to tolerate None outputs.

    With manual optimization training plans, some Lightning/scvi combinations
    pass None as batch outputs to on_train_batch_end.
    """
    try:
        from scvi.train._callbacks import SaveCheckpoint
    except Exception:
        return

    if getattr(SaveCheckpoint, "_condot_compat_patched", False):
        return

    original_on_train_batch_end = SaveCheckpoint.on_train_batch_end

    def _patched_on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        return original_on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx)

    SaveCheckpoint.on_train_batch_end = _patched_on_train_batch_end
    SaveCheckpoint._condot_compat_patched = True


def set_gpu_device(gpu_id: Optional[int] = None) -> str:
    """Select GPU for the current process.

    Note: CUDA_VISIBLE_DEVICES MUST be set before torch is imported to have
    any effect. Callers are expected to do that at the top of their entry
    script/notebook (see notebook 5 cell 1). This function only pins the
    active CUDA device inside the already-initialized torch runtime.
    """
    if not torch.cuda.is_available():
        return "cpu"

    if gpu_id is not None:
        try:
            # After CUDA_VISIBLE_DEVICES masking, the requested device is visible
            # as index 0; but if the user did not mask env, fall back to the
            # literal index they asked for.
            target = 0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") == str(gpu_id) else int(gpu_id)
            torch.cuda.set_device(target)
        except Exception:
            pass
    return "cuda"


def _latest_model_dir(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None

    candidates = [path for path in checkpoint_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _training_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.parent / "training_state.json"


def _load_training_state(checkpoint_dir: Path) -> dict:
    state_path = _training_state_path(checkpoint_dir)
    if not state_path.exists():
        return {}

    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_training_state(checkpoint_dir: Path, state: dict) -> None:
    state_path = _training_state_path(checkpoint_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_checkpoint_monitor(requested_monitor: str) -> tuple[str, str]:
    # biolord/scvi do not log `validation_loss`; map it to an available val loss key.
    aliases = {
        "validation_loss": "val_reconstruction_loss",
    }
    monitor = aliases.get(requested_monitor, requested_monitor)

    max_metrics = {
        "val_generative_mean_accuracy",
        "val_generative_var_accuracy",
        "val_biolord_metric",
    }
    mode = "max" if monitor in max_metrics else "min"
    return monitor, mode


def train_biolord_model(
    adata,
    split_key: str,
    model_name: str = "biolord_condot_aligned",
    n_latent: int = 256,
    max_epochs: int = 200,
    batch_size: int = 512,
    checkpoint_freq: int = 10,
    num_workers: int = 1,
    early_stopping_patience: int = 20,
    seed: int = 42,
    gpu_id: Optional[int] = None,
    module_params: Optional[dict] = None,
    trainer_params: Optional[dict] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_monitor: str = "validation_loss",
    resume: bool = True,
    skip_if_complete: bool = True,
    load_best_on_end: bool = True,
):
    _patch_biolord_training_step_if_needed()
    _patch_scvi_savecheckpoint_if_needed()

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_gpu_device(gpu_id)

    if module_params is None or trainer_params is None:
        from utils.parameters_sciplex3 import module_params as _module_params
        from utils.parameters_sciplex3 import trainer_params as _trainer_params

        module_params = dict(_module_params) if module_params is None else module_params
        trainer_params = dict(_trainer_params) if trainer_params is None else trainer_params

    resolved_monitor, monitor_mode = _resolve_checkpoint_monitor(checkpoint_monitor)
    if resolved_monitor != checkpoint_monitor:
        print(
            "[biolord] checkpoint_monitor='"
            f"{checkpoint_monitor}' is not logged by this model, using '{resolved_monitor}' instead."
        )

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        state = _load_training_state(checkpoint_path)
        if skip_if_complete and state.get("status") == "complete":
            best_model_path = state.get("best_model_path")
            if best_model_path:
                best_model_path = Path(best_model_path)
                if best_model_path.exists():
                    print(f"[biolord] Found completed run, loading: {best_model_path}")
                    return biolord.Biolord.load(str(best_model_path), adata=adata)

    resume_path: Path | None = None
    if checkpoint_path is not None and resume:
        state = _load_training_state(checkpoint_path)
        best_model_path = state.get("best_model_path")
        if best_model_path:
            candidate = Path(best_model_path)
            if candidate.exists():
                resume_path = candidate
        if resume_path is None:
            resume_path = _latest_model_dir(checkpoint_path)

    if resume_path is not None:
        print(f"[biolord] Resuming from: {resume_path}")
        model = biolord.Biolord.load(str(resume_path), adata=adata)
    else:
        biolord.Biolord.setup_anndata(
            adata,
            ordered_attributes_keys=["rdkit2d_dose"],
            categorical_attributes_keys=["cell_type"],
            retrieval_attribute_key=None,
        )

        model = biolord.Biolord(
            adata=adata,
            n_latent=n_latent,
            model_name=model_name,
            module_params=module_params,
            train_classifiers=False,
            split_key=split_key,
        )

    callbacks = []
    if checkpoint_path is not None:
        from scvi.train._callbacks import SaveCheckpoint

        callbacks.append(
            SaveCheckpoint(
                dirpath=str(checkpoint_path),
                monitor=resolved_monitor,
                mode=monitor_mode,
                load_best_on_end=load_best_on_end,
            )
        )

    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=early_stopping_patience,
        check_val_every_n_epoch=checkpoint_freq,
        num_workers=num_workers,
        enable_checkpointing=checkpoint_path is not None,
        checkpointing_monitor=resolved_monitor,
        callbacks=callbacks,
    )

    if checkpoint_path is not None:
        checkpoint_callback = getattr(getattr(model, "trainer", None), "checkpoint_callback", None)
        best_model_path = getattr(checkpoint_callback, "best_model_path", None) if checkpoint_callback else None
        if not best_model_path:
            latest_dir = _latest_model_dir(checkpoint_path)
            best_model_path = str(latest_dir or checkpoint_path)

        _write_training_state(
            checkpoint_path,
            {
                "status": "complete",
                "best_model_path": str(Path(best_model_path).resolve()),
                "checkpoint_dir": str(checkpoint_path.resolve()),
                "checkpoint_monitor": resolved_monitor,
                "checkpoint_mode": monitor_mode,
                "seed": seed,
            },
        )

    return model
