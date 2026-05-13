"""Automated evaluation metrics for the trained variants.

Two metrics, pragmatically scoped:

- **Perplexity** under the base model. Measures how much each variant's
  outputs have drifted from the base distribution. Some increase is
  expected (the policy is being shifted); a huge spike means fluency is
  degraded.
- **BERTScore** of each generated response against the input article
  context. Measures whether jokes stay on-topic.

Self-BLEU (mode-collapse detector) is intentionally omitted: it requires
sampling multiple responses per prompt per variant, doubling generation
cost. Cross-judge head-to-head comparisons already cover the
distinguishability axis we care about most.

All metrics consume the same ``FinalGenerationRecord`` stream from
``eval/generate_eval.py`` and write per-variant summaries to disk.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from eval.generate_eval import FinalGenerationRecord
from models.log_probs import compute_log_probs


# ---------------------------------------------------------------------------
# Perplexity under the base model
# ---------------------------------------------------------------------------


@torch.inference_mode()
def perplexity_under_base(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    response_text: str,
    max_length: int,
) -> float:
    """Per-token perplexity of ``response_text`` given ``prompt_text``.

    Uses the existing ``compute_log_probs`` helper. The base model should
    have its LoRA adapter disabled (or be a fresh PEFT-wrap whose B=0).

    Returns:
        ``exp(-mean log-prob over response tokens)``. Float ``inf`` if
        no response tokens were tokenized (edge case for empty responses).
    """
    device = next(base_model.parameters()).device
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    overflow = (len(prompt_ids) + len(response_ids)) - max_length
    if overflow > 0:
        if overflow < len(prompt_ids):
            prompt_ids = prompt_ids[overflow:]
        else:
            keep_prompt = max(0, max_length // 2)
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt else []
            remaining = max_length - len(prompt_ids)
            response_ids = response_ids[:remaining]

    combined = prompt_ids + response_ids
    if not response_ids:
        return float("inf")

    n_prompt = len(prompt_ids)
    n_total = len(combined)
    n_response = n_total - n_prompt

    input_ids = torch.tensor([combined], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    label_mask = torch.zeros_like(input_ids)
    label_mask[0, n_prompt:n_total] = 1

    log_prob_sum = compute_log_probs(
        base_model, input_ids, attention_mask, label_mask
    ).item()
    avg_log_prob = log_prob_sum / max(n_response, 1)
    return math.exp(-avg_log_prob)


# ---------------------------------------------------------------------------
# BERTScore against article context
# ---------------------------------------------------------------------------


def bertscore_against_contexts(
    responses: List[str],
    contexts: List[str],
    lang: str = "en",
    rescale_with_baseline: bool = True,
    batch_size: int = 32,
) -> List[float]:
    """Per-pair BERTScore F1 between each response and its article context.

    ``responses[i]`` is scored against ``contexts[i]``. Returns a list of
    F1 scores in [0, 1] (or roughly [-0.2, 1] with baseline rescaling,
    since baseline rescaling normalizes against the dataset's mean).
    """
    if len(responses) != len(contexts):
        raise ValueError("responses and contexts must have the same length")
    if not responses:
        return []

    # Heavy import deferred so the module is cheap to load even when
    # bert_score isn't installed; only this function needs it.
    from bert_score import score as bertscore_fn

    _, _, f1 = bertscore_fn(
        responses,
        contexts,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=batch_size,
        verbose=False,
    )
    return f1.tolist()


# ---------------------------------------------------------------------------
# Aggregation across the final-eval generation set
# ---------------------------------------------------------------------------


def _metrics_path(config: Dict[str, Any]) -> str:
    return os.path.join(config["drive_root"], "metrics", "automated_metrics.json")


def compute_all_auto_metrics(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Dict[str, Any],
    final_generations: List[FinalGenerationRecord],
    prompt_records: List[Dict[str, Any]],
    save: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute perplexity + BERTScore per variant, save a summary.

    Args:
        base_model: A PEFT-wrapped model with the LoRA adapter disabled
            (or a fresh wrap whose B=0). The base distribution for
            perplexity scoring.
        tokenizer: Matching tokenizer.
        config: Project config; reads ``max_seq_length`` and ``drive_root``.
        final_generations: Output of ``generate_eval.load_final_generations``.
        prompt_records: All prompt records (used to look up article context).
        save: If True, write a JSON summary to
            ``<drive_root>/metrics/automated_metrics.json``.

    Returns:
        ``{variant_name: {"perplexity_mean": float, "bertscore_mean": float, "n": int}}``.
    """
    max_length = int(config["max_seq_length"])
    contexts_by_prompt = {
        p["prompt_id"]: {"short": p["short_context"], "long": p["long_context"]}
        for p in prompt_records
    }

    # Group records by (variant, context_length) for variant-level aggregation
    grouped: Dict[str, List[FinalGenerationRecord]] = {}
    for r in final_generations:
        key = f"{r['variant']}_{r['context_length']}"
        grouped.setdefault(key, []).append(r)

    # Perplexity is one model call per (prompt, variant) so we batch by variant.
    print("[metrics] Computing perplexity under base model ...")
    perplexity_means: Dict[str, float] = {}
    for variant_key, records in tqdm(grouped.items(), desc="perplexity"):
        ppls: List[float] = []
        for r in records:
            ppl = perplexity_under_base(
                base_model, tokenizer, r["prompt_text"], r["response"], max_length
            )
            if math.isfinite(ppl):
                ppls.append(ppl)
        perplexity_means[variant_key] = sum(ppls) / len(ppls) if ppls else float("inf")

    # BERTScore is batched internally
    print("[metrics] Computing BERTScore vs article context ...")
    bertscore_means: Dict[str, float] = {}
    for variant_key, records in grouped.items():
        responses = [r["response"] for r in records]
        contexts = [
            contexts_by_prompt[r["prompt_id"]][r["context_length"]]
            for r in records
        ]
        f1s = bertscore_against_contexts(responses, contexts)
        bertscore_means[variant_key] = sum(f1s) / len(f1s) if f1s else 0.0

    summary: Dict[str, Dict[str, float]] = {}
    for variant_key in grouped:
        summary[variant_key] = {
            "perplexity_mean": perplexity_means[variant_key],
            "bertscore_mean": bertscore_means[variant_key],
            "n": len(grouped[variant_key]),
        }

    if save:
        path = _metrics_path(config)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[metrics] Saved automated metrics to {path}")

    return summary
