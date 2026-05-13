"""Regenerate winrates_pairwise.png with a flatter aspect ratio for the 2-col CVPR layout."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_REPO_ROOT = os.path.dirname(REPO_ROOT)
SNAP = os.path.join(PARENT_REPO_ROOT, "experiment_snapshot.json")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def render() -> None:
    snap = json.loads(open(SNAP).read())
    short = snap["winrates_short"]
    long_ = snap["winrates_long"]

    pairs = [("DPO vs base", "#4C78A8"), ("KTO vs base", "#F58518"), ("DPO vs KTO", "#54A24B")]
    rates_short = [
        short["base__vs__dpo_rubric_short"]["dpo_rubric_short_wins"]
        / (short["base__vs__dpo_rubric_short"]["dpo_rubric_short_wins"] + short["base__vs__dpo_rubric_short"]["base_wins"]),
        short["base__vs__kto_binary_short"]["kto_binary_short_wins"]
        / (short["base__vs__kto_binary_short"]["kto_binary_short_wins"] + short["base__vs__kto_binary_short"]["base_wins"]),
        short["dpo_rubric_short__vs__kto_binary_short"]["dpo_rubric_short_wins"]
        / (short["dpo_rubric_short__vs__kto_binary_short"]["dpo_rubric_short_wins"] + short["dpo_rubric_short__vs__kto_binary_short"]["kto_binary_short_wins"]),
    ]
    rates_long = [
        long_["base__vs__dpo_rubric_long"]["dpo_rubric_long_wins"]
        / (long_["base__vs__dpo_rubric_long"]["dpo_rubric_long_wins"] + long_["base__vs__dpo_rubric_long"]["base_wins"]),
        long_["base__vs__kto_binary_long"]["kto_binary_long_wins"]
        / (long_["base__vs__kto_binary_long"]["kto_binary_long_wins"] + long_["base__vs__kto_binary_long"]["base_wins"]),
        long_["dpo_rubric_long__vs__kto_binary_long"]["dpo_rubric_long_wins"]
        / (long_["dpo_rubric_long__vs__kto_binary_long"]["dpo_rubric_long_wins"] + long_["dpo_rubric_long__vs__kto_binary_long"]["kto_binary_long_wins"]),
    ]

    fig, ax = plt.subplots(figsize=(14, 2.6))

    n_groups = 2
    n_bars = 3
    width = 0.25
    x_short = np.arange(n_bars) * width
    x_long = np.arange(n_bars) * width + n_bars * width + 0.4

    for i, (label, color) in enumerate(pairs):
        ax.bar(x_short[i], rates_short[i], width, color=color, label=label)
        ax.bar(x_long[i], rates_long[i], width, color=color)

    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="parity")
    for x, v in zip(list(x_short) + list(x_long), rates_short + rates_long):
        ax.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    ax.set_ylim(0, 1)
    ax.set_xticks([x_short.mean(), x_long.mean()])
    ax.set_xticklabels(["Short context", "Long context"])
    ax.set_ylabel("Win rate of first variant in pair")
    ax.set_title("Pairwise win rates from LLM cross-judge (n=30 prompts)")
    ax.legend(loc="upper right", ncol=4, fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"winrates_pairwise.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"saved: {path}")


if __name__ == "__main__":
    render()
