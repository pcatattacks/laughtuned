"""Regenerate training_curves.png as a 3-panel figure: train loss, val loss, KTO log-ratio.

Folds the KTO log-ratio collapse panel into this figure so we save a top-of-page slot.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_REPO_ROOT = os.path.dirname(REPO_ROOT)
METRICS_DIR = os.path.join(PARENT_REPO_ROOT, "metrics")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


VARIANTS = [
    ("dpo_rubric_short", "DPO (short)", "#4C78A8"),
    ("dpo_rubric_long",  "DPO (long)",  "#F58518"),
    ("kto_binary_short", "KTO (short)", "#54A24B"),
    ("kto_binary_long",  "KTO (long)",  "#E45756"),
]

KTO_VARIANTS = [
    ("kto_binary_short", "KTO (short)", "#54A24B"),
    ("kto_binary_long",  "KTO (long)",  "#E45756"),
]


def _load(path: str, phase: str, key: str = "loss") -> Tuple[List[int], List[float]]:
    steps, values = [], []
    seen = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") != phase:
                continue
            s = int(r["step"])
            if phase == "val" and s in seen:
                continue
            seen.add(s)
            steps.append(s)
            values.append(float(r[key]))
    return steps, values


def render() -> None:
    fig, (ax_t, ax_v, ax_lr) = plt.subplots(1, 3, figsize=(15, 3.0))

    for variant, label, color in VARIANTS:
        log = os.path.join(METRICS_DIR, variant, "log.jsonl")
        s_t, l_t = _load(log, "train", "loss")
        s_v, l_v = _load(log, "val",   "loss")
        ax_t.plot(s_t, l_t, color=color, alpha=0.55, linewidth=0.7, label=label)
        ax_v.plot(s_v, l_v, color=color, marker="o", markersize=3, linewidth=1.2, label=label)

    for variant, label, color in KTO_VARIANTS:
        log = os.path.join(METRICS_DIR, variant, "log.jsonl")
        s_v, lr_v = _load(log, "val", "log_ratio_mean")
        ax_lr.plot(s_v, lr_v, color=color, marker="o", markersize=3, linewidth=1.2, label=label)

    ax_t.set_xlabel("Optimizer step")
    ax_t.set_ylabel("Loss")
    ax_t.set_title("(a) Training loss")
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.grid(alpha=0.25)

    ax_v.set_xlabel("Optimizer step")
    ax_v.set_ylabel("Loss")
    ax_v.set_title("(b) Validation loss")
    ax_v.legend(loc="upper right", fontsize=8)
    ax_v.grid(alpha=0.25)

    ax_lr.axhline(0, color="black", linewidth=0.6)
    ax_lr.set_xlabel("Optimizer step")
    ax_lr.set_ylabel(r"Mean $\log\pi_\theta/\pi_{\mathrm{ref}}$")
    ax_lr.set_title("(c) KTO policy drift")
    ax_lr.legend(loc="upper right", fontsize=8)
    ax_lr.grid(alpha=0.25)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"training_curves.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"saved: {path}")


if __name__ == "__main__":
    render()
