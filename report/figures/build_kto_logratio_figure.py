"""Build the KTO log-ratio collapse plot (F6).

Shows ``log_ratio_mean`` over training for the two KTO variants, alongside
their val-loss curves, demonstrating that loss continues to fall while
the policy drifts far from the reference.
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


def _load_val(path: str) -> Tuple[List[int], List[float], List[float]]:
    steps: List[int] = []
    losses: List[float] = []
    log_ratios: List[float] = []
    seen = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") != "val":
                continue
            s = int(r["step"])
            if s in seen:
                continue
            seen.add(s)
            steps.append(s)
            losses.append(float(r["loss"]))
            log_ratios.append(float(r["log_ratio_mean"]))
    return steps, losses, log_ratios


def render() -> None:
    s_short, loss_s, lr_s = _load_val(
        os.path.join(METRICS_DIR, "kto_binary_short", "log.jsonl")
    )
    s_long, loss_l, lr_l = _load_val(
        os.path.join(METRICS_DIR, "kto_binary_long", "log.jsonl")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 2.2), sharex=True)

    ax1.plot(s_short, loss_s, marker="o", markersize=3,
             color="#4C78A8", label="KTO short")
    ax1.plot(s_long, loss_l, marker="s", markersize=3,
             color="#E45756", label="KTO long")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Validation loss")
    ax1.set_title("(a) Val loss falls monotonically")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.plot(s_short, lr_s, marker="o", markersize=3,
             color="#4C78A8", label="KTO short")
    ax2.plot(s_long, lr_l, marker="s", markersize=3,
             color="#E45756", label="KTO long")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel(r"Mean $\log\pi_\theta / \pi_{\mathrm{ref}}$")
    ax2.set_title("(b) Policy drifts far from reference")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"kto_log_ratio_collapse.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"saved: {path}")


if __name__ == "__main__":
    render()
