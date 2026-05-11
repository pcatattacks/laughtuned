"""Guardian article ingestion with Claude-synthesized historical backstories.

Pulls articles from a diverse set of Guardian sections, then for each article
issues a relevance search for older articles on the same topic and asks Claude
to summarize the relevant prior context into a 150-word backstory. Each article
is saved to JSONL as soon as it's processed, so a crashed run resumes from
where it left off.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TypedDict

import anthropic
import requests
from anthropic.types import TextBlock
from transformers import PreTrainedTokenizerBase


# Guardian sections to sample from, per the project spec.
GUARDIAN_SECTIONS: List[str] = [
    "politics",
    "business",
    "technology",
    "sport",
    "culture",
    "science",
    "environment",
    "world",
]

GUARDIAN_BASE_URL = "https://content.guardianapis.com"
# Guardian developer tier caps at 1 req/sec and 500 req/day.
GUARDIAN_MIN_REQUEST_INTERVAL_S = 1.05
_last_guardian_request_ts: float = 0.0

SHORT_CONTEXT_BODY_TOKEN_BUDGET = 400
NUM_HISTORICAL_ARTICLES = 3
BACKSTORY_MAX_TOKENS = 400  # ~150 words plus headroom

# Historical lookback window for backstory context. The buffer skips the
# same-news-cycle period immediately before the article (otherwise Guardian
# returns articles that are part of the same ongoing story, which would
# defeat the purpose of "historical" context).
BACKSTORY_LOOKBACK_DAYS = 60
BACKSTORY_BUFFER_DAYS = 7

BACKSTORY_PROMPT_TEMPLATE = """<task>Write a 150-word backstory summarizing only the events from the older articles that are directly relevant to today's story. Ignore anything unrelated. Write in a factual, concise style.</task>

<current_article>
<headline>{headline}</headline>
<body>{body_snippet}</body>
</current_article>

<older_articles>
{older_articles_block}
</older_articles>

<output_format>Write the backstory as a single paragraph. No preamble.</output_format>"""

_OLDER_ARTICLE_BLOCK_TEMPLATE = """<article>
<headline>{headline}</headline>
<body>{body}</body>
</article>"""


class ArticleRecord(TypedDict):
    """One ingested Guardian article with both short and long context variants."""

    article_id: str
    section: str
    headline: str
    published_at: str
    short_context: str
    long_context: str
    topic_query: str
    older_article_ids: List[str]


# ---------------------------------------------------------------------------
# Low-level Guardian API
# ---------------------------------------------------------------------------


def _throttle_guardian() -> None:
    """Sleep just long enough to stay under the Guardian rate cap."""
    global _last_guardian_request_ts
    now = time.monotonic()
    elapsed = now - _last_guardian_request_ts
    if elapsed < GUARDIAN_MIN_REQUEST_INTERVAL_S:
        time.sleep(GUARDIAN_MIN_REQUEST_INTERVAL_S - elapsed)
    _last_guardian_request_ts = time.monotonic()


def _guardian_get(
    endpoint: str,
    params: Dict[str, Any],
    api_keys: List[str],
    max_retries: int = 4,
) -> Dict[str, Any]:
    """GET a Guardian endpoint with throttling, backoff, and key rotation.

    Tries each key in ``api_keys`` in order. If a key exhausts its retries
    (e.g., daily-limit 429s that don't recover under backoff), rotates to
    the next key. Raises only if all keys fail.
    """
    url = f"{GUARDIAN_BASE_URL}{endpoint}"
    last_error: Optional[Exception] = None

    for key_idx, api_key in enumerate(api_keys):
        params_with_auth = {**params, "api-key": api_key, "format": "json"}
        for attempt in range(max_retries):
            _throttle_guardian()
            try:
                response = requests.get(url, params=params_with_auth, timeout=20)
                if response.status_code == 200:
                    payload: Dict[str, Any] = response.json()
                    return payload["response"]
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
            except (requests.RequestException, ValueError) as e:
                last_error = e
                if attempt == max_retries - 1:
                    break
                time.sleep(2**attempt)
        if key_idx < len(api_keys) - 1:
            print(
                f"  [guardian] key {key_idx + 1}/{len(api_keys)} exhausted; "
                "rotating to next"
            )
    raise RuntimeError(
        f"Guardian GET {url} exhausted all {len(api_keys)} key(s): {last_error}"
    )


def search_section(
    api_keys: List[str],
    section: str,
    from_date: str,
    to_date: str,
    page_size: int,
    page: int = 1,
    production_office: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return articles from a section between two ISO dates (inclusive).

    ``api_keys`` is a list of Guardian keys to rotate through on
    rate-limit exhaustion. ``production_office`` filters to one of
    Guardian's regional offices (``"us"``, ``"uk"``, ``"aus"``); ``None``
    returns global content.
    """
    params: Dict[str, Any] = {
        "section": section,
        "from-date": from_date,
        "to-date": to_date,
        "page-size": page_size,
        "page": page,
        "show-fields": "bodyText,headline,trailText",
        "order-by": "newest",
    }
    if production_office is not None:
        params["production-office"] = production_office
    resp = _guardian_get("/search", params, api_keys)
    return list(resp.get("results", []))


def _select_time_diverse(
    pool: List[Dict[str, Any]],
    n: int,
    from_date: str,
    to_date: str,
) -> List[Dict[str, Any]]:
    """Pick ``n`` articles maximally spread across the lookback window.

    Partitions ``from_date`` → ``to_date`` into ``n`` equal time slices and
    selects the most-relevant article in each non-empty slice. ``pool`` is
    expected to come from Guardian in relevance order (first = most
    relevant), so within each time bucket we keep that ranking.
    """
    if len(pool) <= n:
        return pool

    try:
        from_dt = datetime.fromisoformat(from_date)
        to_dt = datetime.fromisoformat(to_date)
    except ValueError:
        return pool[:n]

    total_seconds = (to_dt - from_dt).total_seconds()
    if total_seconds <= 0:
        return pool[:n]
    slice_seconds = total_seconds / n

    relevance_rank: Dict[int, int] = {id(a): idx for idx, a in enumerate(pool)}
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    for art in pool:
        pub_raw = art.get("webPublicationDate", "")
        try:
            pub_dt = datetime.fromisoformat(
                pub_raw.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except ValueError:
            buckets[-1].append(art)
            continue
        offset = (pub_dt - from_dt).total_seconds()
        idx = int(offset / slice_seconds)
        idx = min(max(idx, 0), n - 1)
        buckets[idx].append(art)

    selected: List[Dict[str, Any]] = []
    for bucket in buckets:
        if bucket:
            best = min(bucket, key=lambda a: relevance_rank[id(a)])
            selected.append(best)
    return selected


def search_related(
    api_keys: List[str],
    topic_query: str,
    to_date: str,
    from_date: Optional[str] = None,
    top_n: int = NUM_HISTORICAL_ARTICLES,
    production_office: Optional[str] = None,
    search_pool_size: int = 10,
) -> List[Dict[str, Any]]:
    """Return up to ``top_n`` older articles relevant to ``topic_query``.

    Fetches ``search_pool_size`` candidates by relevance, then partitions
    them across the ``from_date``→``to_date`` window so the chosen articles
    span the lookback range rather than clustering in one time period.

    Both ``from_date`` and ``to_date`` are inclusive ISO dates. The caller
    is responsible for choosing a ``to_date`` that excludes the current
    article's news cycle (typically ``article_date - BACKSTORY_BUFFER_DAYS``)
    so that "older" results are genuinely historical rather than same-cycle
    coverage of the same event.

    Pass the same ``production_office`` used for the section search to keep
    the historical arc consistent with the current article's edition.
    """
    params: Dict[str, Any] = {
        "q": topic_query,
        "to-date": to_date,
        "page-size": search_pool_size,
        "show-fields": "bodyText,headline,trailText",
        "order-by": "relevance",
    }
    if from_date is not None:
        params["from-date"] = from_date
    if production_office is not None:
        params["production-office"] = production_office
    resp = _guardian_get("/search", params, api_keys)
    pool = list(resp.get("results", []))
    if from_date is not None and len(pool) > top_n:
        return _select_time_diverse(pool, top_n, from_date, to_date)
    return pool[:top_n]


# ---------------------------------------------------------------------------
# Article extraction helpers
# ---------------------------------------------------------------------------


_SLUG_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _article_id_from_guardian(raw: Dict[str, Any]) -> str:
    """Stable filename-safe ID derived from Guardian's path."""
    return _SLUG_SANITIZE_RE.sub("_", raw["id"])


def _body_text(raw: Dict[str, Any]) -> str:
    fields = raw.get("fields") or {}
    return fields.get("bodyText", "") or ""


def extract_short_context(
    raw: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    body_token_budget: int = SHORT_CONTEXT_BODY_TOKEN_BUDGET,
) -> str:
    """Headline plus the first ``body_token_budget`` tokens of bodyText."""
    headline: str = (raw.get("fields") or {}).get("headline") or raw.get(
        "webTitle", ""
    )
    body = _body_text(raw)
    if not body:
        return headline

    token_ids = tokenizer.encode(body, add_special_tokens=False)[:body_token_budget]
    body_snippet = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return f"{headline}\n\n{body_snippet}"


def extract_topic_query(raw: Dict[str, Any]) -> str:
    """Use the headline as the natural-language relevance query.

    Guardian's relevance search handles full-sentence queries well, so we
    avoid hand-rolled keyword extraction (which would lose context).
    """
    fields = raw.get("fields") or {}
    return fields.get("headline") or raw.get("webTitle", "")


# ---------------------------------------------------------------------------
# Claude backstory synthesis
# ---------------------------------------------------------------------------


def _claude_text(message: Any) -> str:
    """Pull the text content from a Claude Messages API response."""
    for block in message.content:
        if isinstance(block, TextBlock):
            return block.text
    return ""


def synthesize_backstory(
    client: anthropic.Anthropic,
    model: str,
    current_raw: Dict[str, Any],
    older_raws: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    body_token_budget: int = SHORT_CONTEXT_BODY_TOKEN_BUDGET,
) -> str:
    """Ask Claude for a 150-word backstory referencing only the older articles."""
    if not older_raws:
        return ""

    headline = (current_raw.get("fields") or {}).get("headline") or current_raw.get(
        "webTitle", ""
    )
    body_ids = tokenizer.encode(_body_text(current_raw), add_special_tokens=False)
    current_body_snippet = tokenizer.decode(
        body_ids[:body_token_budget], skip_special_tokens=True
    ).strip()

    older_blocks: List[str] = []
    for older in older_raws:
        older_headline = (older.get("fields") or {}).get("headline") or older.get(
            "webTitle", ""
        )
        older_body_ids = tokenizer.encode(_body_text(older), add_special_tokens=False)
        older_body_snippet = tokenizer.decode(
            older_body_ids[:body_token_budget], skip_special_tokens=True
        ).strip()
        older_blocks.append(
            _OLDER_ARTICLE_BLOCK_TEMPLATE.format(
                headline=older_headline, body=older_body_snippet
            )
        )

    prompt = BACKSTORY_PROMPT_TEMPLATE.format(
        headline=headline,
        body_snippet=current_body_snippet,
        older_articles_block="\n".join(older_blocks),
    )

    message = client.messages.create(
        model=model,
        max_tokens=BACKSTORY_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return _claude_text(message).strip()


# ---------------------------------------------------------------------------
# Disk I/O (idempotency)
# ---------------------------------------------------------------------------


def _articles_dir(config: Dict[str, Any]) -> str:
    return os.path.join(config["drive_root"], "data", "articles")


def _articles_jsonl_path(config: Dict[str, Any]) -> str:
    return os.path.join(_articles_dir(config), "articles.jsonl")


def load_existing_articles(config: Dict[str, Any]) -> List[ArticleRecord]:
    """Return every article previously saved to disk (possibly empty)."""
    path = _articles_jsonl_path(config)
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_article(config: Dict[str, Any], record: ArticleRecord) -> None:
    os.makedirs(_articles_dir(config), exist_ok=True)
    with open(_articles_jsonl_path(config), "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def ingest_articles(
    config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    articles_per_section: int = 63,
    days_back: int = 30,
    sections: Optional[List[str]] = None,
) -> List[ArticleRecord]:
    """Fetch + process articles, persisting each to JSONL as it completes.

    Resumes automatically: any article ID already present in the JSONL is
    skipped, so re-running after a crash or rate-limit pause continues
    where it left off.

    Args:
        config: Project config. Reads ``guardian_api_key``,
            ``anthropic_api_key``, ``judge_model``, ``drive_root``.
        tokenizer: Tokenizer used to truncate article bodies to a fixed
            token budget for short_context construction.
        articles_per_section: How many articles to attempt per section.
            ``8 sections * 63 ≈ 504`` covers the 500 prompt budget.
        days_back: Look-back window for the primary article search.
        sections: Optional override of ``GUARDIAN_SECTIONS``.

    Returns:
        The full list of articles on disk after this run completes.
    """
    sections = sections or GUARDIAN_SECTIONS
    guardian_keys: List[str] = [
        k
        for k in (
            config.get("guardian_api_key", ""),
            config.get("guardian_api_key_2", ""),
        )
        if k
    ]
    anthropic_key: str = config["anthropic_api_key"]
    if not guardian_keys or not anthropic_key:
        raise RuntimeError(
            "guardian_api_key (and optionally guardian_api_key_2) plus "
            "anthropic_api_key must be set on CONFIG before calling "
            "ingest_articles."
        )

    claude = anthropic.Anthropic(api_key=anthropic_key)
    claude_model: str = config["judge_model"]
    production_office: Optional[str] = config.get("guardian_production_office")

    existing = load_existing_articles(config)
    seen_ids = {rec["article_id"] for rec in existing}
    print(f"[fetch_articles] Resuming with {len(existing)} articles already saved.")
    print(f"[fetch_articles] Guardian keys available: {len(guardian_keys)}")
    if production_office:
        print(f"[fetch_articles] Filtering to Guardian {production_office!r} edition.")

    today = datetime.now(timezone.utc).date()
    from_date = (today - timedelta(days=days_back)).isoformat()
    to_date = today.isoformat()

    for section in sections:
        print(f"[fetch_articles] Section: {section}")
        try:
            raw_articles = search_section(
                guardian_keys,
                section=section,
                from_date=from_date,
                to_date=to_date,
                page_size=articles_per_section,
                production_office=production_office,
            )
        except Exception as e:
            print(f"  failed to list section {section}: {e}")
            continue

        for raw in raw_articles:
            article_id = _article_id_from_guardian(raw)
            if article_id in seen_ids:
                continue
            if not _body_text(raw):
                continue

            published_at: str = raw.get("webPublicationDate", "")
            if published_at:
                published_date = datetime.fromisoformat(
                    published_at.replace("Z", "+00:00")
                ).date()
            else:
                published_date = today
            related_to_date = (
                published_date - timedelta(days=BACKSTORY_BUFFER_DAYS)
            ).isoformat()
            related_from_date = (
                published_date - timedelta(days=BACKSTORY_LOOKBACK_DAYS)
            ).isoformat()

            short_context = extract_short_context(raw, tokenizer)
            topic_query = extract_topic_query(raw)

            try:
                older_raws = search_related(
                    guardian_keys,
                    topic_query=topic_query,
                    to_date=related_to_date,
                    from_date=related_from_date,
                    production_office=production_office,
                )
                # Defensive: drop the current article if Guardian somehow
                # returns it (e.g., via a slug variant).
                older_raws = [
                    o
                    for o in older_raws
                    if _article_id_from_guardian(o) != article_id
                ]
            except Exception as e:
                # Skip rather than save a partial record so a re-run (e.g.,
                # after swapping API keys past a daily-limit hit) retries it.
                print(f"  related-search failed for {article_id}: {e}; skipping")
                continue

            try:
                backstory = synthesize_backstory(
                    claude, claude_model, raw, older_raws, tokenizer
                )
            except Exception as e:
                print(f"  backstory synth failed for {article_id}: {e}; skipping")
                continue

            long_context = (
                f"{backstory}\n\n{short_context}" if backstory else short_context
            )

            record: ArticleRecord = {
                "article_id": article_id,
                "section": section,
                "headline": (raw.get("fields") or {}).get("headline")
                or raw.get("webTitle", ""),
                "published_at": published_at,
                "short_context": short_context,
                "long_context": long_context,
                "topic_query": topic_query,
                "older_article_ids": [
                    _article_id_from_guardian(o) for o in older_raws
                ],
            }
            _append_article(config, record)
            seen_ids.add(article_id)
            existing.append(record)

    print(f"[fetch_articles] Total articles on disk: {len(existing)}")
    return existing
