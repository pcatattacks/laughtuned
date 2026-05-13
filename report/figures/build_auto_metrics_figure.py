"""Build the perplexity + BERTScore grouped bar figure (F5)."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_REPO_ROOT = os.path.dirname(REPO_ROOT)
METRICS_PATH = os.path.join(PARENT_REPO_ROOT, "metrics", "automated_metrics.json")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


VARIANT_ORDER = [
    ("base_short", "Base\nshort"),
    ("dpo_rubric_short_short", "DPO\nshort"),
    ("kto_binary_short_short", "KTO\nshort"),
    ("base_long", "Base\nlong"),
    ("dpo_rubric_long_long", "DPO\nlong"),
    ("kto_binary_long_long", "KTO\nlong"),
]


def render() -> None:
    with open(METRICS_PATH) as f:
        m = json.load(f)

    labels = [lab for _, lab in VARIANT_ORDER]
    ppl = [m[k]["perplexity_mean"] for k, _ in VARIANT_ORDER]
    bert = [m[k]["bertscore_mean"] for k, _ in VARIANT_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))

    colors = ["#888888", "#4C78A8", "#E45756"] * 2
    x = np.arange(len(labels))

    bars1 = ax1.bar(x, ppl, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Perplexity (base-model)")
    ax1.set_title("(a) Perplexity")
    ax1.axvline(2.5, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    for bar, v in zip(bars1, ppl):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                 f"{v:.2f}", ha="center", fontsize=7.5)
    ax1.set_ylim(0, max(ppl) * 1.18)

    bars2 = ax2.bar(x, bert, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("BERTScore F1 vs context")
    ax2.set_title("(b) BERTScore")
    ax2.axvline(2.5, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    for bar, v in zip(bars2, bert):
        offset = -0.006 if v < 0 else 0.003
        va = "top" if v < 0 else "bottom"
        ax2.text(bar.get_x() + bar.get_width() / 2, v + offset,
                 f"{v:.3f}", ha="center", va=va, fontsize=7.5)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"auto_metrics.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"saved: {path}")


if __name__ == "__main__":
    render()
