"""LoRA adapter checkpointing during training.

A checkpoint includes:
- The LoRA adapter weights only (PEFT's ``save_pretrained``)
- Optimizer state
- LR scheduler state
- Current optimizer step, best val loss seen so far
- Bookkeeping JSON for resume-from-disk logic

Everything lives under ``<drive_root>/checkpoints/<experiment_name>/``.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Optional

import torch
from peft import PeftModel


class CheckpointManager:
    """Save / load LoRA adapter checkpoints to Google Drive."""

    def __init__(self, config: Dict[str, Any], experiment_name: str) -> None:
        self._root: str = os.path.join(
            config["drive_root"], "checkpoints", experiment_name
        )
        os.makedirs(self._root, exist_ok=True)
        self._best_path: str = os.path.join(self._root, "best")
        self._latest_path: str = os.path.join(self._root, "latest")
        self._state_json_path: str = os.path.join(self._root, "state.json")

    def save(
        self,
        model: PeftModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        step: int,
        epoch: int,
        val_loss: float,
        best_val_loss: float,
        best_step: int,
        evals_without_improvement: int,
    ) -> None:
        """Write a ``latest`` checkpoint and, if improved, also a ``best`` copy.

        ``epoch`` is the index of the epoch we were inside at the time of
        the save (0-indexed). On resume, training restarts at this epoch
        — the partial work of an interrupted epoch is redone, but the
        optimizer and scheduler state are preserved so the LR schedule
        continues from where it left off.
        """
        os.makedirs(self._latest_path, exist_ok=True)
        model.save_pretrained(self._latest_path)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "step": step,
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "evals_without_improvement": evals_without_improvement,
            },
            os.path.join(self._latest_path, "training_state.pt"),
        )
        self._write_state(step=step, best_val_loss=best_val_loss)

        if val_loss <= best_val_loss:
            self._copy_dir(self._latest_path, self._best_path)

    def load_latest_adapter(self, model: PeftModel) -> None:
        """Reload LoRA adapter weights from ``latest/`` onto ``model``.

        Call this on a freshly-loaded PEFT model before ``load_latest``
        so the policy weights match the optimizer state being restored.
        """
        if not os.path.isdir(self._latest_path):
            raise FileNotFoundError(f"No latest checkpoint at {self._latest_path}")
        model.load_adapter(
            self._latest_path, adapter_name="default", is_trainable=True
        )

    def has_resume_point(self) -> bool:
        return os.path.isdir(self._latest_path) and os.path.isfile(
            os.path.join(self._latest_path, "training_state.pt")
        )

    def load_latest(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    ) -> Dict[str, Any]:
        """Restore optimizer + scheduler state from ``latest``.

        Does NOT touch model weights — the LoRA adapter at
        ``self._latest_path`` should be reloaded separately by the caller
        via ``model.load_adapter(...)`` because the exact reload path
        depends on how the model was constructed (``PeftModel.from_pretrained``
        vs ``load_adapter`` on a fresh wrap).
        """
        if not self.has_resume_point():
            raise FileNotFoundError(f"No checkpoint at {self._latest_path}")
        state = torch.load(
            os.path.join(self._latest_path, "training_state.pt"),
            map_location="cpu",
        )
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        return state

    def load_best(self, model: PeftModel) -> None:
        """Reload best adapter weights onto an already-initialized PeftModel.

        Drops any existing ``"default"`` adapter first so PEFT versions that
        don't auto-overwrite on ``load_adapter`` (older releases) won't
        choke. No-op when PEFT does auto-overwrite cleanly.
        """
        if not os.path.isdir(self._best_path):
            print("[checkpoint] No 'best' checkpoint to load; leaving model as is.")
            return
        if "default" in getattr(model, "peft_config", {}):
            try:
                model.delete_adapter("default")
            except Exception as e:
                print(f"[checkpoint] delete_adapter('default') failed: {e}; continuing")
        model.load_adapter(self._best_path, adapter_name="default", is_trainable=True)

    def _write_state(self, step: int, best_val_loss: float) -> None:
        with open(self._state_json_path, "w") as f:
            json.dump({"step": step, "best_val_loss": best_val_loss}, f, indent=2)

    @staticmethod
    def _copy_dir(src: str, dst: str) -> None:
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
