"""Comedy prompt construction from ingested articles.

Builds one prompt per article with a randomly assigned comedy style. Each
prompt has both a short-context and a long-context version using the same
template — the only variable is what goes inside ``<context>``. Articles
without a synthesized backstory degrade to ``long_context == short_context``
for that prompt (no contrast contributed by that article).

Persisted artifacts:
- ``<drive_root>/data/prompts/prompts.jsonl`` — one PromptRecord per line
- ``<drive_root>/data/splits/split_indices.json`` — train/val/eval prompt IDs
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple, TypedDict

from data.fetch_articles import ArticleRecord


COMEDY_STYLES: List[str] = ["observational", "absurdist", "one-liner"]


_PROMPT_TEMPLATES: Dict[str, str] = {
    "observational": (
        "<role>You are a sharp observational comedian.</role>\n"
        "\n"
        "<context>\n"
        "{context}\n"
        "</context>\n"
        "\n"
        "<task>Write a short, specific comedic take (2-3 sentences). "
        "Reference real details from the context. Don't be generic.</task>\n"
        "\n"
        "<output_format>Respond with only the joke. "
        "No preamble, no explanation, no quotation marks.</output_format>"
    ),
    "absurdist": (
        "<role>You are an absurdist comedian who takes real situations to "
        "their logical extreme.</role>\n"
        "\n"
        "<context>\n"
        "{context}\n"
        "</context>\n"
        "\n"
        "<task>Write a short comedic take (2-3 sentences) that starts "
        "grounded in the real details, then spirals into absurdity.</task>\n"
        "\n"
        "<output_format>Respond with only the joke. "
        "No preamble, no explanation, no quotation marks.</output_format>"
    ),
    "one-liner": (
        "<role>You are a comedian known for razor-sharp one-liners.</role>\n"
        "\n"
        "<context>\n"
        "{context}\n"
        "</context>\n"
        "\n"
        "<task>Write a single punchy joke about this. "
        "Reference specific details.</task>\n"
        "\n"
        "<output_format>Respond with only the joke. "
        "No preamble, no explanation, no quotation marks.</output_format>"
    ),
}


class PromptRecord(TypedDict):
    """One comedy prompt paired with an article, with short and long variants."""

    prompt_id: str
    article_id: str
    style: str
    prompt_text_short: str
    prompt_text_long: str
    short_context: str
    long_context: str
    split: str  # "train", "val", or "eval"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_long_context(article: ArticleRecord) -> Tuple[str, str]:
    """Return ``(backstory, short_context)`` from an article record.

    ``backstory`` is the empty string when the article has no synthesized
    historical context (e.g., ``older_article_ids`` was empty during
    ingestion).
    """
    long_ctx = article["long_context"]
    short_ctx = article["short_context"]
    if long_ctx == short_ctx:
        return "", short_ctx
    sep = "\n\n"
    if long_ctx.endswith(sep + short_ctx):
        return long_ctx[: -len(sep + short_ctx)], short_ctx
    return "", short_ctx


def _build_long_context_xml(backstory: str, short_ctx: str) -> str:
    """Wrap a backstory + short context with ``<background>`` and ``<latest>``."""
    if backstory:
        return f"<background>{backstory}</background>\n<latest>{short_ctx}</latest>"
    return short_ctx


def _build_prompt_text(style: str, context_inner: str) -> str:
    return _PROMPT_TEMPLATES[style].format(context=context_inner)


def _prompts_jsonl_path(config: Dict[str, Any]) -> str:
    return os.path.join(config["drive_root"], "data", "prompts", "prompts.jsonl")


def _splits_json_path(config: Dict[str, Any]) -> str:
    return os.path.join(
        config["drive_root"], "data", "splits", "split_indices.json"
    )


def load_existing_prompts(config: Dict[str, Any]) -> List[PromptRecord]:
    """Return prompts previously persisted to disk (possibly empty)."""
    path = _prompts_jsonl_path(config)
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_prompts(config: Dict[str, Any], prompts: List[PromptRecord]) -> None:
    path = _prompts_jsonl_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")


def _save_splits(config: Dict[str, Any], splits: Dict[str, List[str]]) -> None:
    path = _splits_json_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def build_prompts(
    config: Dict[str, Any],
    articles: List[ArticleRecord],
) -> List[PromptRecord]:
    """Generate one prompt per article and assign train/val/eval splits.

    Idempotent: if a non-empty ``prompts.jsonl`` already exists on disk,
    the existing list is returned unchanged. Delete the file to rebuild.

    Args:
        config: Project config. Reads ``seed``, ``num_eval_prompts``,
            ``val_ratio``, and ``drive_root``.
        articles: Ingested article records (output of ``ingest_articles``).

    Returns:
        Every PromptRecord written to disk, in stable order by ``prompt_id``.
    """
    existing = load_existing_prompts(config)
    if existing:
        print(
            f"[build_prompts] Found {len(existing)} existing prompts on disk; "
            "skipping rebuild."
        )
        return existing

    rng = random.Random(config.get("seed", 42))
    n_eval: int = int(config.get("num_eval_prompts", 30))
    val_ratio: float = float(config.get("val_ratio", 0.1))

    sorted_articles = sorted(articles, key=lambda a: a["article_id"])

    prompts: List[PromptRecord] = []
    for i, art in enumerate(sorted_articles):
        backstory, short_ctx = _split_long_context(art)
        style = rng.choice(COMEDY_STYLES)
        prompt_short = _build_prompt_text(style, short_ctx)
        prompt_long = _build_prompt_text(
            style, _build_long_context_xml(backstory, short_ctx)
        )
        prompts.append(
            PromptRecord(
                prompt_id=f"p_{i:04d}",
                article_id=art["article_id"],
                style=style,
                prompt_text_short=prompt_short,
                prompt_text_long=prompt_long,
                short_context=art["short_context"],
                long_context=art["long_context"],
                split="",
            )
        )

    indices = list(range(len(prompts)))
    rng.shuffle(indices)
    eval_cut = min(n_eval, len(prompts))
    eval_indices = set(indices[:eval_cut])
    rest = indices[eval_cut:]
    val_count = max(1, int(len(rest) * val_ratio)) if rest else 0
    val_indices = set(rest[:val_count])

    for i, p in enumerate(prompts):
        if i in eval_indices:
            p["split"] = "eval"
        elif i in val_indices:
            p["split"] = "val"
        else:
            p["split"] = "train"

    _save_prompts(config, prompts)

    splits: Dict[str, List[str]] = {
        "train": [p["prompt_id"] for p in prompts if p["split"] == "train"],
        "val": [p["prompt_id"] for p in prompts if p["split"] == "val"],
        "eval": [p["prompt_id"] for p in prompts if p["split"] == "eval"],
    }
    _save_splits(config, splits)

    style_counts: Dict[str, int] = {}
    no_backstory_count = 0
    for p in prompts:
        style_counts[p["style"]] = style_counts.get(p["style"], 0) + 1
        if p["prompt_text_long"] == p["prompt_text_short"]:
            no_backstory_count += 1

    print(
        f"[build_prompts] Built {len(prompts)} prompts | "
        f"train={len(splits['train'])}, val={len(splits['val'])}, "
        f"eval={len(splits['eval'])}"
    )
    print(f"[build_prompts] Styles: {style_counts}")
    print(
        f"[build_prompts] {no_backstory_count} prompts have no backstory "
        f"(long_context == short_context)."
    )

    return prompts
