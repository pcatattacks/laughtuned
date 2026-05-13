"""Build the human-vs-LLM agreement figure for the report.

Two panels:
  (a) Side-by-side win rates: LLM judge vs human, across the six headline
      pairwise comparisons (3 algorithm pairs x 2 context lengths).
  (b) Pairwise agreement breakdown overall + per context, with a 33% chance
      reference line.

Reads ``data/eval/llm_judge_results.jsonl`` and ``data/eval/human_judgments.jsonl``
(both in the repo) and saves PNG + PDF to ``report/figures/``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from itertools import combinations
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_REPO_ROOT = os.path.dirname(REPO_ROOT)
LLM_PATH = os.path.join(PARENT_REPO_ROOT, "data", "eval", "llm_judge_results.jsonl")
HUMAN_PATH = os.path.join(PARENT_REPO_ROOT, "data", "eval", "human_judgments.jsonl")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _winrate(records: List[dict], ctx: str, a_sub: str, b_sub: str, key: str) -> float:
    """Win rate of variant matching a_sub vs variant matching b_sub."""
    wins = losses = 0
    for r in records:
        if r["context_length"] != ctx:
            continue
        ranks = r[key]
        a = next((v for v in ranks if a_sub in v), None)
        b = next((v for v in ranks if b_sub in v), None)
        if a is None or b is None:
            continue
        if ranks.index(a) < ranks.index(b):
            wins += 1
        else:
            losses += 1
    return wins / (wins + losses) if (wins + losses) else 0.0


def _pair_agreement(human_records: List[dict]) -> Tuple[float, float, float]:
    def _dom(rank, a, b):
        return rank.index(a) < rank.index(b)
    overall = [0, 0]
    per_ctx = {"short": [0, 0], "long": [0, 0]}
    for p in human_records:
        for a, b in combinations(set(p["human_variants"]), 2):
            h_dom = _dom(p["human_variants"], a, b)
            l_dom = _dom(p["llm_variants"], a, b)
            overall[1] += 1
            per_ctx[p["context_length"]][1] += 1
            if h_dom == l_dom:
                overall[0] += 1
                per_ctx[p["context_length"]][0] += 1
    return (
        overall[0] / overall[1] if overall[1] else 0.0,
        per_ctx["short"][0] / per_ctx["short"][1] if per_ctx["short"][1] else 0.0,
        per_ctx["long"][0] / per_ctx["long"][1] if per_ctx["long"][1] else 0.0,
    )


def _build_human_pseudo_records(human_records: List[dict]) -> List[dict]:
    """Adapt human_records into the same shape as llm_records for _winrate."""
    return [
        {
            "context_length": p["context_length"],
            "variant_rankings": p["human_variants"],
        }
        for p in human_records
    ]


def render() -> None:
    llm_records = _load_jsonl(LLM_PATH)
    human_records = _load_jsonl(HUMAN_PATH)

    fig, (ax_w, ax_a) = plt.subplots(
        1, 2, figsize=(14, 2.6),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )

    # --- Panel A: side-by-side win rates per comparison ---
    comparisons = [
        ("DPO/base", "dpo", "base"),
        ("KTO/base", "kto", "base"),
        ("DPO/KTO",  "dpo", "kto"),
    ]
    contexts = ["short", "long"]

    # x positions: 6 groups across (3 comparisons x 2 contexts), 2 bars each
    group_labels = [f"{cmp_label}\n{ctx}" for ctx in contexts for cmp_label, _, _ in comparisons]
    n_groups = len(group_labels)
    x = np.arange(n_groups)
    width = 0.36

    human_pseudo = _build_human_pseudo_records(human_records)
    llm_rates: List[float] = []
    human_rates: List[float] = []
    for ctx in contexts:
        for _, a, b in comparisons:
            llm_rates.append(_winrate(llm_records, ctx, a, b, "variant_rankings"))
            human_rates.append(_winrate(human_pseudo, ctx, a, b, "variant_rankings"))

    ax_w.bar(x - width / 2, llm_rates, width, label="LLM judge",
             color="#4C78A8")
    ax_w.bar(x + width / 2, human_rates, width, label="Human (author)",
             color="#E45756")
    ax_w.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels(group_labels, fontsize=8)
    ax_w.set_ylim(0, 1)
    ax_w.set_ylabel("Win rate of first variant in pair")
    ax_w.set_title("(a) Win rates: LLM vs human")
    ax_w.legend(loc="upper right", fontsize=9)
    for i, (lr, hr) in enumerate(zip(llm_rates, human_rates)):
        ax_w.text(x[i] - width / 2, lr + 0.02, f"{lr:.0%}", ha="center", fontsize=7.5)
        ax_w.text(x[i] + width / 2, hr + 0.02, f"{hr:.0%}", ha="center", fontsize=7.5)

    # --- Panel B: pairwise agreement breakdown ---
    overall, short_ag, long_ag = _pair_agreement(human_records)
    cats = ["overall", "short", "long"]
    vals = [overall, short_ag, long_ag]
    bars = ax_a.bar(cats, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    ax_a.axhline(1.0 / 3, color="black", linestyle=":", linewidth=1, label="chance (33%)")
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("Pairwise agreement")
    ax_a.set_title("(b) Human-vs-LLM agreement")
    for bar, val in zip(bars, vals):
        ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                  f"{val:.0%}", ha="center", fontsize=9)
    ax_a.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"agreement_matrix.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"saved: {path}")


if __name__ == "__main__":
    render()
