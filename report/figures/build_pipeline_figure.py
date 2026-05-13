"""Render the LaughTuned method pipeline as a single-row block diagram for the 2-col CVPR layout."""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _box(ax, x, y, w, h, text, fc, ec="#222222", fontsize=13):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, wrap=True,
    )


def _arrow(ax, x0, y0, x1, y1, color="#222222", lw=1.6):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
    )


def render() -> None:
    fig, ax = plt.subplots(figsize=(16, 3.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    blue = "#D4E4F7"
    yellow = "#FFF4C2"
    green = "#D7EFD3"
    pink = "#F8D7DA"

    box_w, box_h = 2.3, 1.25
    row_y = 1.35
    n = 6
    total_box = n * box_w
    gap = (16 - total_box) / (n + 1)
    xs = [gap + i * (box_w + gap) for i in range(n)]

    stages: List[Tuple[str, str]] = [
        ("Guardian\narticles (~500)", blue),
        ("Comedy prompts\n(3 styles \xd7 2 contexts)", blue),
        ("Base Mistral\npaired sampling", yellow),
        ("Claude rubric\njudge", yellow),
        ("DPO + KTO\n\xd7 short/long\n(4 variants)", green),
        ("Cross-judge eval\n(LLM) +\nhuman eval", pink),
    ]

    for x, (label, color) in zip(xs, stages):
        _box(ax, x, row_y, box_w, box_h, label, color, fontsize=13)

    for i in range(n - 1):
        x0 = xs[i] + box_w
        x1 = xs[i + 1]
        _arrow(ax, x0, row_y + box_h / 2, x1, row_y + box_h / 2)

    callout_y = row_y + box_h + 0.28
    callouts = [
        (1, "observational / absurdist / one-liner"),
        (3, "rubric: humor, specificity, format"),
        (4, "from-scratch losses (no TRL)"),
        (5, "rank-3 forced choice"),
    ]
    for idx, txt in callouts:
        ax.text(
            xs[idx] + box_w / 2, callout_y, txt,
            ha="center", va="center", fontsize=11, color="#555555", style="italic",
        )

    legend_y = 0.15
    legend_h = 0.40
    swatches = [
        ("data", blue),
        ("gen / judge", yellow),
        ("training", green),
        ("evaluation", pink),
    ]
    sw_w = 2.2
    total_legend = len(swatches) * sw_w
    sw_x = (16 - total_legend) / 2
    for label, color in swatches:
        rect = mpatches.Rectangle((sw_x, legend_y), 0.4, legend_h, facecolor=color, edgecolor="#222222")
        ax.add_patch(rect)
        ax.text(sw_x + 0.55, legend_y + legend_h / 2, label, va="center", ha="left", fontsize=12)
        sw_x += sw_w

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"pipeline.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"saved: {path}")


if __name__ == "__main__":
    render()
