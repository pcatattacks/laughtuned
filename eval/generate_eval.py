"""Final-eval generation: produce comparable responses from base + each variant.

For each held-out eval prompt, sample one response from:
- the base Mistral (no LoRA), and
- each trained LoRA adapter (DPO short, DPO long, KTO short, KTO long).

Same generation hyperparameters as Step 4 so the only thing that varies is
the policy (or context length). Results land in ``data/eval/final_generations.jsonl``
with one record per (prompt_id, variant) — flat, easy to pivot in Step 11.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, TypedDict

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from data.build_prompts import PromptRecord
from data.generate_pairs import _decode_responses, _prepare_tokenizer_for_generation


# Names of the four trained variants and the "base" baseline. Keep in sync
# with the experiment_name strings used by `train()` in the demo notebook.
VARIANT_NAMES: List[str] = [
    "base",
    "dpo_rubric_short",
    "dpo_rubric_long",
    "kto_binary_short",
    "kto_binary_long",
]


class FinalGenerationRecord(TypedDict):
    """One sampled response from one variant on one eval prompt."""

    prompt_id: str
    variant: str  # one of VARIANT_NAMES
    context_length: str  # "short" or "long" (the prompt's context, not the variant)
    response: str
    prompt_text: str


def _final_generations_path(config: Dict[str, Any]) -> str:
    return os.path.join(
        config["drive_root"], "data", "eval", "final_generations.jsonl"
    )


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _adapter_path(config: Dict[str, Any], variant: str) -> str:
    """Return the on-disk path to a variant's BEST checkpoint."""
    return os.path.join(
        config["drive_root"], "checkpoints", variant, "best"
    )


def _generate_one_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts_with_text: List[tuple],  # list of (PromptRecord, prompt_text)
    config: Dict[str, Any],
    variant_name: str,
    context_length: str,
    batch_size: int,
    output_path: str,
    show_progress: bool,
) -> List[FinalGenerationRecord]:
    """Generate one response per prompt for the given variant."""
    _prepare_tokenizer_for_generation(tokenizer)
    # `prepare_model_for_kbit_training` (called at model load) turns gradient
    # checkpointing ON and KV-cache OFF for training. Both must flip for
    # `model.generate()` to work — otherwise SDPA attention sees a mask
    # shaped for the full sequence while the query is only the new token,
    # raising a shape-mismatch RuntimeError mid-generation.
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.config.use_cache = True
    device = next(model.parameters()).device
    max_new = int(config["gen_max_new_tokens"])
    max_prompt_tokens = int(config["max_seq_length"]) - max_new

    new_records: List[FinalGenerationRecord] = []
    iterator: Any = range(0, len(prompts_with_text), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"eval-gen [{variant_name}/{context_length}]")

    for start in iterator:
        batch = prompts_with_text[start : start + batch_size]
        texts = [t for _, t in batch]
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=float(config["gen_temperature"]),
                top_p=float(config["gen_top_p"]),
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = _decode_responses(output_ids, tokenized.input_ids, tokenizer)
        batch_records: List[FinalGenerationRecord] = [
            FinalGenerationRecord(
                prompt_id=p["prompt_id"],
                variant=variant_name,
                context_length=context_length,
                response=decoded[i].strip(),
                prompt_text=text,
            )
            for i, (p, text) in enumerate(batch)
        ]
        _append_jsonl(output_path, [dict(r) for r in batch_records])
        new_records.extend(batch_records)

    return new_records


def generate_for_variant(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Dict[str, Any],
    prompts: List[PromptRecord],
    variant_name: str,
    context_length: str,
    batch_size: int = 4,
    show_progress: bool = True,
) -> List[FinalGenerationRecord]:
    """Generate one response per eval prompt for one variant.

    Assumes the supplied ``model`` already has the right adapter loaded
    (or no adapter, for the base baseline). Idempotent — skips
    (prompt_id, variant) combinations already on disk.

    Args:
        model: PEFT-wrapped model with the variant's adapter active (or
            base model with adapter disabled / not yet loaded).
        tokenizer: Matching tokenizer.
        config: Project config.
        prompts: All prompt records; only those with ``split == "eval"`` are used.
        variant_name: One of ``VARIANT_NAMES``.
        context_length: ``"short"`` or ``"long"``. Determines which prompt
            field is fed to the model.
        batch_size: Number of prompts per forward pass.
        show_progress: tqdm.
    """
    if variant_name not in VARIANT_NAMES:
        raise ValueError(f"variant_name must be one of {VARIANT_NAMES}")
    if context_length not in ("short", "long"):
        raise ValueError("context_length must be 'short' or 'long'")

    path = _final_generations_path(config)
    existing = _load_jsonl(path)
    done = {
        (r["prompt_id"], r["variant"], r["context_length"])
        for r in existing
    }
    prompt_field = f"prompt_text_{context_length}"
    to_do = [
        (p, p[prompt_field])  # type: ignore[literal-required]
        for p in prompts
        if p["split"] == "eval"
        and (p["prompt_id"], variant_name, context_length) not in done
    ]
    if not to_do:
        print(
            f"[generate_eval] All {variant_name}/{context_length} eval "
            "responses already on disk."
        )
        return []

    return _generate_one_pass(
        model, tokenizer, to_do, config,
        variant_name=variant_name,
        context_length=context_length,
        batch_size=batch_size,
        output_path=path,
        show_progress=show_progress,
    )


def load_final_generations(
    config: Dict[str, Any],
) -> List[FinalGenerationRecord]:
    """Reload everything previously generated for the final eval."""
    return _load_jsonl(_final_generations_path(config))  # type: ignore[return-value]


def index_generations(
    records: List[FinalGenerationRecord],
) -> Dict[str, Dict[tuple, str]]:
    """Pivot records to ``{prompt_id: {(variant, context_length): response}}``.

    Used by Step 11's cross-judge to look up all variants' responses
    side-by-side per prompt. Tuple keys disambiguate ``("base", "short")``
    from ``("base", "long")`` cleanly.
    """
    out: Dict[str, Dict[tuple, str]] = {}
    for r in records:
        key = (r["variant"], r["context_length"])
        out.setdefault(r["prompt_id"], {})[key] = r["response"]
    return out
