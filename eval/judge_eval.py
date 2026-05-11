"""LLM cross-judge for the final evaluation.

For each held-out eval prompt we run two 3-way rankings via Claude:
- short context:  base vs dpo_rubric_short vs kto_binary_short
- long  context:  base vs dpo_rubric_long  vs kto_binary_long

The judge sees the article context (short or long) and three labeled
responses (X, Y, Z, randomized so the judge can't bias by variant
position) and ranks them best→worst. Pairwise win rates are derived
client-side from the rankings.

Uses a DIFFERENT prompt template than the training-time rubric judge (per
spec, to avoid circularity).
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Tuple, TypedDict

import anthropic
from anthropic.types import TextBlock
from tqdm.auto import tqdm


_CROSS_JUDGE_PROMPT_TEMPLATE = """<task>You are judging a comedy competition. Three responses were written about the same topic. Rank them from funniest to least funny. The labels X, Y, Z are in randomized order.</task>

<context>
{context}
</context>

<responses>
<response_x>{response_x}</response_x>
<response_y>{response_y}</response_y>
<response_z>{response_z}</response_z>
</responses>

<output_format>
Respond in exactly this format and nothing else:
RANKING: [1st], [2nd], [3rd] (use X, Y, Z)
REASON: (one sentence about why the top choice works best)
</output_format>"""


class CrossJudgeRecord(TypedDict):
    """One 3-way ranking judgment for one (prompt, context_length) pair."""

    prompt_id: str
    context_length: str  # "short" or "long"
    variant_rankings: List[str]  # variant names, best -> worst
    label_mapping: Dict[str, str]  # "X" -> variant name (for blinding)
    reason: str
    raw_response: str


_RANKING_RE = re.compile(
    r"RANKING\s*:\s*([XYZ])[\s,]+([XYZ])[\s,]+([XYZ])",
    re.IGNORECASE,
)
_REASON_RE = re.compile(
    r"REASON\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# One judging call
# ---------------------------------------------------------------------------


def _claude_text(message: Any) -> str:
    for block in message.content:
        if isinstance(block, TextBlock):
            return block.text
    return ""


def _parse_cross_judge_response(raw: str) -> Tuple[List[str], str]:
    """Pull (ranking_letters, reason) out of Claude's reply.

    Raises ``ValueError`` if the format doesn't match.
    """
    m = _RANKING_RE.search(raw)
    if m is None:
        raise ValueError(f"Could not parse RANKING from:\n{raw}")
    ranking_letters = [m.group(i).upper() for i in (1, 2, 3)]
    if set(ranking_letters) != {"X", "Y", "Z"}:
        raise ValueError(
            f"RANKING must list X, Y, Z exactly once each. Got: {ranking_letters}"
        )
    reason_match = _REASON_RE.search(raw)
    reason = reason_match.group(1).strip() if reason_match else ""
    return ranking_letters, reason


def judge_triplet(
    claude: anthropic.Anthropic,
    model: str,
    context: str,
    variant_to_response: Dict[str, str],
    rng: random.Random,
    max_tokens: int = 300,
) -> Tuple[List[str], Dict[str, str], str, str]:
    """Send one 3-way ranking call to Claude.

    Args:
        claude: Anthropic client.
        model: Claude model ID.
        context: Article context (``short_context`` or ``long_context``).
        variant_to_response: Exactly 3 entries, mapping variant name to
            response text.
        rng: Used to shuffle the X/Y/Z labeling so the judge can't bias
            by variant position.
        max_tokens: Output cap. 300 fits the short structured reply.

    Returns:
        ``(variant_rankings, label_mapping, reason, raw_response)`` —
        ``variant_rankings`` is the variants in best-to-worst order;
        ``label_mapping`` is ``{"X": variant_name, ...}``.
    """
    if len(variant_to_response) != 3:
        raise ValueError("judge_triplet requires exactly 3 variants")

    variants = list(variant_to_response.keys())
    rng.shuffle(variants)
    label_mapping = {"X": variants[0], "Y": variants[1], "Z": variants[2]}

    prompt = _CROSS_JUDGE_PROMPT_TEMPLATE.format(
        context=context,
        response_x=variant_to_response[label_mapping["X"]],
        response_y=variant_to_response[label_mapping["Y"]],
        response_z=variant_to_response[label_mapping["Z"]],
    )
    message = claude.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _claude_text(message)
    ranking_letters, reason = _parse_cross_judge_response(raw)
    variant_rankings = [label_mapping[letter] for letter in ranking_letters]
    return variant_rankings, label_mapping, reason, raw


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def _cross_judge_jsonl_path(config: Dict[str, Any]) -> str:
    return os.path.join(
        config["drive_root"], "data", "eval", "llm_judge_results.jsonl"
    )


def load_existing_cross_judge(config: Dict[str, Any]) -> List[CrossJudgeRecord]:
    path = _cross_judge_jsonl_path(config)
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_cross_judge(config: Dict[str, Any], record: CrossJudgeRecord) -> None:
    path = _cross_judge_jsonl_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def cross_judge_all(
    config: Dict[str, Any],
    prompts: List[Dict[str, Any]],  # all PromptRecords (only "eval" split used)
    generations_by_prompt: Dict[str, Dict[Tuple[str, str], str]],
    show_progress: bool = True,
) -> List[CrossJudgeRecord]:
    """Run two 3-way rankings per eval prompt (short ctx + long ctx).

    Args:
        config: Project config; reads ``anthropic_api_key``, ``judge_model``,
            ``seed``, ``drive_root``.
        prompts: All prompt records (we filter to ``split == "eval"``).
        generations_by_prompt: Output of
            ``generate_eval.index_generations``: nested mapping
            ``{prompt_id: {(variant, context_length): response}}``.
        show_progress: tqdm bar.

    Returns:
        Newly judged records (existing on-disk records are skipped).
        Idempotent — resume-safe.
    """
    anthropic_key: str = config["anthropic_api_key"]
    if not anthropic_key:
        raise RuntimeError("anthropic_api_key must be set on CONFIG.")
    claude = anthropic.Anthropic(api_key=anthropic_key)
    judge_model: str = config["judge_model"]
    rng = random.Random(int(config.get("seed", 42)))

    existing = load_existing_cross_judge(config)
    done = {(r["prompt_id"], r["context_length"]) for r in existing}

    tasks: List[Tuple[str, str, str, Dict[str, str]]] = []
    for prompt in prompts:
        if prompt["split"] != "eval":
            continue
        pid: str = prompt["prompt_id"]
        per_prompt_gens = generations_by_prompt.get(pid, {})
        for ctx in ("short", "long"):
            if (pid, ctx) in done:
                continue
            base_resp = per_prompt_gens.get(("base", ctx))
            dpo_resp = per_prompt_gens.get((f"dpo_rubric_{ctx}", ctx))
            kto_resp = per_prompt_gens.get((f"kto_binary_{ctx}", ctx))
            if not (base_resp and dpo_resp and kto_resp):
                missing = [
                    name for name, r in (
                        ("base", base_resp),
                        (f"dpo_rubric_{ctx}", dpo_resp),
                        (f"kto_binary_{ctx}", kto_resp),
                    ) if not r
                ]
                print(f"  missing responses for {pid}/{ctx}: {missing}; skipping")
                continue
            variant_to_response = {
                "base": base_resp,
                f"dpo_rubric_{ctx}": dpo_resp,
                f"kto_binary_{ctx}": kto_resp,
            }
            context_text: str = prompt[f"{ctx}_context"]  # type: ignore[literal-required]
            tasks.append((pid, ctx, context_text, variant_to_response))

    if not tasks:
        print("[cross_judge] Nothing new to judge.")
        return []

    print(
        f"[cross_judge] {len(tasks)} 3-way rankings to run "
        f"(~${len(tasks) * 0.01:.2f} in Claude calls)"
    )

    new_records: List[CrossJudgeRecord] = []
    iterator: Any = tasks
    if show_progress:
        iterator = tqdm(tasks, desc="cross-judging")

    for pid, ctx, context_text, variant_to_response in iterator:
        try:
            variant_rankings, label_mapping, reason, raw = judge_triplet(
                claude, judge_model, context_text, variant_to_response, rng
            )
        except Exception as e:
            print(f"  cross-judge failed for {pid}/{ctx}: {e}")
            continue
        record: CrossJudgeRecord = {
            "prompt_id": pid,
            "context_length": ctx,
            "variant_rankings": variant_rankings,
            "label_mapping": label_mapping,
            "reason": reason,
            "raw_response": raw,
        }
        _append_cross_judge(config, record)
        new_records.append(record)

    return new_records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_pairwise_winrates(
    cross_judge_records: List[CrossJudgeRecord],
    context_length: str | None = None,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Derive pairwise win rates from 3-way rankings.

    For each unordered pair (A, B) appearing in any ranking, count how
    often A ranked above B and vice versa. Optionally restrict to one
    ``context_length``.

    Returns ``{(a, b): {"a_wins": int, "b_wins": int, "win_rate_a": float}}``
    where (a, b) is sorted alphabetically.
    """
    counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for r in cross_judge_records:
        if context_length is not None and r["context_length"] != context_length:
            continue
        ranks = r["variant_rankings"]
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                a, b = ranks[i], ranks[j]  # a is ranked better than b
                key = tuple(sorted([a, b]))  # canonical ordering
                if key not in counts:
                    counts[key] = {key[0]: 0, key[1]: 0}
                counts[key][a] += 1

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (a, b), c in counts.items():
        total = c[a] + c[b]
        out[(a, b)] = {
            f"{a}_wins": c[a],
            f"{b}_wins": c[b],
            f"win_rate_{a}": c[a] / total if total else 0.0,
        }
    return out
