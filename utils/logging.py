"""Training-time logging to TensorBoard and JSONL.

Each metrics dict from the loss function (plus learning rate, grad norm,
GPU memory) gets written to both:
- TensorBoard ``SummaryWriter`` at ``<drive_root>/metrics/<experiment_name>/tb/``
- Append-only JSONL at ``<drive_root>/metrics/<experiment_name>/log.jsonl``

The JSONL is what survives a kernel restart; TensorBoard is for the live
view during a session.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


class TrainingLogger:
    """Stream metrics to TensorBoard and append to a JSONL log file."""

    def __init__(self, config: Dict[str, Any], experiment_name: str) -> None:
        root = os.path.join(
            config["drive_root"], "metrics", experiment_name
        )
        os.makedirs(root, exist_ok=True)
        self._jsonl_path: str = os.path.join(root, "log.jsonl")
        self._tb_dir: str = os.path.join(root, "tb")
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
            self._writer = SummaryWriter(self._tb_dir)
        except Exception as e:
            print(f"[logger] TensorBoard unavailable ({e}); JSONL only.")
        self._start_time = time.time()

    def log_step(
        self, metrics: Dict[str, float], step: int, phase: str = "train"
    ) -> None:
        """Write one metrics dict for the current optimizer step."""
        self._write_jsonl({"step": step, "phase": phase, **metrics})
        if self._writer is not None:
            for k, v in metrics.items():
                try:
                    self._writer.add_scalar(f"{phase}/{k}", float(v), step)
                except Exception:
                    pass

    def log_eval(self, metrics: Dict[str, float], step: int) -> None:
        """Convenience wrapper for validation metrics."""
        self.log_step(metrics, step, phase="val")

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        record["elapsed_sec"] = round(time.time() - self._start_time, 2)
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
