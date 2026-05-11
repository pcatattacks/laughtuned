"""LLM judge for preference labeling using Claude's structured rubric.

For each generated (response_a, response_b) pair we ask Claude Sonnet to
score both responses on three dimensions — context_engagement, comedic
technique, and surprise — and pick the funnier overall. The same rubric
scores feed two downstream consumers:

- **DPO** uses the winner (higher composite score) to build (chosen, rejected) pairs.
- **KTO** uses the composite scores to label each response binarily:
  desirable (composite > config["kto_desirable_threshold"]),
  undesirable (composite < config["kto_undesirable_threshold"]),
  or dropped (in between).

This keeps the evaluation criterion identical across algorithms — the
only difference is how each one consumes the scores. Records are written
to JSONL as soon as each pair is judged (resume-friendly).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

import anthropic
from anthropic.types import TextBlock
from tqdm.auto import tqdm

from data.generate_pairs import GenerationRecord


RUBRIC_DIMENSIONS: List[str] = ["context", "technique", "surprise"]


_RUBRIC_PROMPT_TEMPLATE = """<task>You are judging a comedy competition. Evaluate two responses on three dimensions using the rubric below, then pick the overall funnier response.</task>

<rubric>
<dimension name="context_engagement" description="Does the joke use the material it was given?">
<score value="1">Completely ignores the context, could be about anything</score>
<score value="2">Mentions the topic in passing but the joke doesn't depend on it</score>
<score value="3">References a specific detail from the context</score>
<score value="4">The joke's structure depends on specific context details</score>
<score value="5">The joke would be meaningless without the context — deeply integrated</score>
</dimension>

<dimension name="comedic_technique" description="Does the joke employ recognizable craft?">
<techniques>irony, sarcasm, misdirection/subversion, exaggeration/hyperbole, bathos (undercutting something serious with something trivial), absurdist escalation, callback, rule of three, analogy/comparison, understatement, double meaning/wordplay</techniques>
<score value="1">Just a statement or observation with no technique</score>
<score value="2">Mild wordplay or a weak attempt at a joke structure</score>
<score value="3">Clear use of one technique with a recognizable setup and payoff</score>
<score value="4">Effective use of technique — the setup creates a genuine expectation that the punchline subverts</score>
<score value="5">Sophisticated technique — works on multiple levels or combines techniques effectively</score>
</dimension>

<dimension name="surprise" description="Did the punchline go somewhere unexpected?">
<score value="1">Completely predictable from the setup</score>
<score value="2">Predictable category but mildly unexpected wording</score>
<score value="3">Somewhat unexpected direction</score>
<score value="4">Genuinely surprising while still making logical sense</score>
<score value="5">Completely unexpected but retroactively obvious — the best kind of punchline</score>
</dimension>
</rubric>

<context>
{context}
</context>

<responses>
<response_a>{response_a}</response_a>
<response_b>{response_b}</response_b>
</responses>

<output_format>
Respond in exactly this format and nothing else:
SCORES_A: context=X technique=X surprise=X
SCORES_B: context=X technique=X surprise=X
WINNER: A or B
REASON: (one sentence)
</output_format>"""


class RubricResult(TypedDict):
    """Parsed Claude judge output for one pair."""

    scores_a: Dict[str, int]
    scores_b: Dict[str, int]
    composite_a: float
    composite_b: float
    winner: str  # "a" or "b"
    reason: str
    raw: str


class RubricRecord(TypedDict):
    """One pair's full judging output ready to drive DPO and KTO training."""

    prompt_id: str
    context_length: str
    response_a: str
    response_b: str
    scores_a: Dict[str, int]
    scores_b: Dict[str, int]
    composite_a: float
    composite_b: float
    winner: str
    kto_label_a: Optional[str]
    kto_label_b: Optional[str]
    judge_raw_response: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_SCORES_RE = re.compile(
    r"SCORES_([AB])\s*:\s*"
    r"context\s*=\s*(\d)\s+"
    r"technique\s*=\s*(\d)\s+"
    r"surprise\s*=\s*(\d)",
    re.IGNORECASE,
)
_WINNER_RE = re.compile(r"WINNER\s*:\s*([AB])", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def _parse_rubric_response(raw: str) -> RubricResult:
    """Pull scores, winner, and reason out of Claude's structured reply.

    Raises ``ValueError`` if any required field is missing or malformed.
    """
    scores_by_side: Dict[str, Dict[str, int]] = {}
    for match in _SCORES_RE.finditer(raw):
        side = match.group(1).lower()
        scores_by_side[side] = {
            "context": int(match.group(2)),
            "technique": int(match.group(3)),
            "surprise": int(match.group(4)),
        }

    if "a" not in scores_by_side or "b" not in scores_by_side:
        raise ValueError(f"Could not parse SCORES_A/SCORES_B from:\n{raw}")

    winner_match = _WINNER_RE.search(raw)
    if not winner_match:
        raise ValueError(f"Could not parse WINNER from:\n{raw}")
    winner = winner_match.group(1).lower()

    reason_match = _REASON_RE.search(raw)
    reason = reason_match.group(1).strip() if reason_match else ""

    composite_a = sum(scores_by_side["a"].values()) / len(RUBRIC_DIMENSIONS)
    composite_b = sum(scores_by_side["b"].values()) / len(RUBRIC_DIMENSIONS)

    return RubricResult(
        scores_a=scores_by_side["a"],
        scores_b=scores_by_side["b"],
        composite_a=composite_a,
        composite_b=composite_b,
        winner=winner,
        reason=reason,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# KTO label derivation
# ---------------------------------------------------------------------------


def compute_kto_label(
    composite_score: float,
    desirable_threshold: float,
    undesirable_threshold: float,
) -> Optional[str]:
    """Bucket a composite rubric score into desirable / undesirable / dropped."""
    if composite_score > desirable_threshold:
        return "desirable"
    if composite_score < undesirable_threshold:
        return "undesirable"
    return None


# ---------------------------------------------------------------------------
# Single-pair judging
# ---------------------------------------------------------------------------


def _claude_text(message: Any) -> str:
    for block in message.content:
        if isinstance(block, TextBlock):
            return block.text
    return ""


def judge_pair_rubric(
    claude: anthropic.Anthropic,
    model: str,
    context: str,
    response_a: str,
    response_b: str,
    max_tokens: int = 400,
) -> RubricResult:
    """Send one rubric-judging call to Claude and return parsed scores.

    Args:
        claude: An ``anthropic.Anthropic`` client.
        model: Claude model ID.
        context: The article context shown to Claude for the judging call
            (typically the prompt's ``short_context`` or ``long_context``).
        response_a: Mistral response A.
        response_b: Mistral response B.
        max_tokens: Output token cap. 400 is plenty for the structured
            short reply.

    Raises:
        ValueError: If Claude's reply doesn't parse to the expected format.
    """
    prompt = _RUBRIC_PROMPT_TEMPLATE.format(
        context=context, response_a=response_a, response_b=response_b
    )
    message = claude.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_rubric_response(_claude_text(message))


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def _rubric_jsonl_path(config: Dict[str, Any]) -> str:
    return os.path.join(
        config["drive_root"], "data", "preferences", "pairwise_rubric.jsonl"
    )


def load_existing_rubric_records(
    config: Dict[str, Any],
) -> List[RubricRecord]:
    path = _rubric_jsonl_path(config)
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_rubric_record(
    config: Dict[str, Any], record: RubricRecord
) -> None:
    path = _rubric_jsonl_path(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def judge_all_pairs(
    config: Dict[str, Any],
    generations: List[GenerationRecord],
    contexts_by_prompt: Dict[str, Dict[str, str]],
    show_progress: bool = True,
) -> List[RubricRecord]:
    """Run rubric judging on every generation pair, persist incrementally.

    Args:
        config: Project config. Reads ``anthropic_api_key``, ``judge_model``,
            ``kto_desirable_threshold``, ``kto_undesirable_threshold``,
            ``drive_root``.
        generations: All (prompt, context_length) pairs from Step 4.
        contexts_by_prompt: Lookup ``{prompt_id: {"short": str, "long": str}}``
            so we can pass the matching context to Claude per pair. Build it
            from the PromptRecord list with ``_build_contexts_map``.
        show_progress: tqdm progress bar.

    Returns:
        Newly judged RubricRecords (already-on-disk records are skipped
        and not re-listed). Idempotent: resume from the last saved pair.
    """
    anthropic_key: str = config["anthropic_api_key"]
    if not anthropic_key:
        raise RuntimeError("anthropic_api_key must be set on CONFIG.")
    claude = anthropic.Anthropic(api_key=anthropic_key)
    judge_model: str = config["judge_model"]
    desirable_thresh: float = float(config["kto_desirable_threshold"])
    undesirable_thresh: float = float(config["kto_undesirable_threshold"])

    existing = load_existing_rubric_records(config)
    done = {(r["prompt_id"], r["context_length"]) for r in existing}
    print(f"[judge] Resuming with {len(existing)} pairs already judged.")

    to_do = [
        g for g in generations if (g["prompt_id"], g["context_length"]) not in done
    ]
    if not to_do:
        print("[judge] Nothing to do.")
        return []

    new_records: List[RubricRecord] = []
    iterator: Any = to_do
    if show_progress:
        iterator = tqdm(to_do, desc="judging")

    for gen in iterator:
        ctx_map = contexts_by_prompt.get(gen["prompt_id"])
        if ctx_map is None:
            print(f"  no context lookup for {gen['prompt_id']}; skipping")
            continue
        context_text = ctx_map.get(gen["context_length"], "")

        try:
            result = judge_pair_rubric(
                claude,
                judge_model,
                context=context_text,
                response_a=gen["response_a"],
                response_b=gen["response_b"],
            )
        except Exception as e:
            print(f"  judge failed for {gen['prompt_id']}/{gen['context_length']}: {e}")
            continue

        record: RubricRecord = {
            "prompt_id": gen["prompt_id"],
            "context_length": gen["context_length"],
            "response_a": gen["response_a"],
            "response_b": gen["response_b"],
            "scores_a": result["scores_a"],
            "scores_b": result["scores_b"],
            "composite_a": result["composite_a"],
            "composite_b": result["composite_b"],
            "winner": result["winner"],
            "kto_label_a": compute_kto_label(
                result["composite_a"], desirable_thresh, undesirable_thresh
            ),
            "kto_label_b": compute_kto_label(
                result["composite_b"], desirable_thresh, undesirable_thresh
            ),
            "judge_raw_response": result["raw"],
        }
        _append_rubric_record(config, record)
        new_records.append(record)

    _print_label_distribution(new_records, existing)
    return new_records


def build_contexts_map(
    prompts: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Index prompt records by id with short/long context strings."""
    return {
        p["prompt_id"]: {
            "short": p["short_context"],
            "long": p["long_context"],
        }
        for p in prompts
    }


def _print_label_distribution(
    new_records: List[RubricRecord],
    existing_records: List[RubricRecord],
) -> None:
    all_records = list(existing_records) + list(new_records)
    if not all_records:
        return

    label_counts: Dict[str, int] = {"desirable": 0, "undesirable": 0, "dropped": 0}
    composite_values: List[float] = []
    a_wins = b_wins = 0
    for r in all_records:
        for lbl in (r["kto_label_a"], r["kto_label_b"]):
            if lbl == "desirable":
                label_counts["desirable"] += 1
            elif lbl == "undesirable":
                label_counts["undesirable"] += 1
            else:
                label_counts["dropped"] += 1
        composite_values.extend([r["composite_a"], r["composite_b"]])
        if r["winner"] == "a":
            a_wins += 1
        else:
            b_wins += 1

    total_responses = label_counts["desirable"] + label_counts["undesirable"] + label_counts["dropped"]
    print(
        f"[judge] Pairwise wins: A={a_wins}, B={b_wins} "
        f"(A win rate {a_wins / max(1, a_wins + b_wins):.0%})"
    )
    print(
        f"[judge] KTO label distribution over {total_responses} responses: "
        f"desirable={label_counts['desirable']}, "
        f"undesirable={label_counts['undesirable']}, "
        f"dropped={label_counts['dropped']}"
    )
    if composite_values:
        mean_c = sum(composite_values) / len(composite_values)
        print(f"[judge] Composite score mean: {mean_c:.2f}")
