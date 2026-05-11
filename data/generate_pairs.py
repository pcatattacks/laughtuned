"""Batch generation of comedy responses from the base Mistral model.

For each train/val prompt, generates **2 responses** (``response_a``,
``response_b``) at each context length. Duplicating the same prompt twice
in the same forward pass produces two independent samples because each
row of the batch draws independently from the multinomial during
sampling — no special seed plumbing required.

For each eval prompt, generates **1 baseline response** per context
length (kept separate from the training pool for final cross-judge eval).

Both flows write to JSONL after every batch so a crashed run resumes
from the last saved batch.

Persisted artifacts:
- ``<drive_root>/data/generations/generations.jsonl``  (train+val pairs)
- ``<drive_root>/data/generations/baselines.jsonl``    (eval singles)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, TypedDict

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from data.build_prompts import PromptRecord


CONTEXT_LENGTHS: List[str] = ["short", "long"]


class GenerationRecord(TypedDict):
    """Two sampled responses for one (prompt, context_length) pair."""

    prompt_id: str
    context_length: str  # "short" or "long"
    response_a: str
    response_b: str
    prompt_text: str


class BaselineRecord(TypedDict):
    """A single sampled response for an eval prompt."""

    prompt_id: str
    context_length: str
    response: str
    prompt_text: str


# ---------------------------------------------------------------------------
# Paths and JSONL helpers
# ---------------------------------------------------------------------------


def _generations_path(config: Dict[str, Any]) -> str:
    return os.path.join(
        config["drive_root"], "data", "generations", "generations.jsonl"
    )


def _baselines_path(config: Dict[str, Any]) -> str:
    return os.path.join(
        config["drive_root"], "data", "generations", "baselines.jsonl"
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


def _prompt_field_for(context_length: str) -> str:
    return f"prompt_text_{context_length}"


def _decode_responses(
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> List[str]:
    """Strip the prompt prefix and decode each generated continuation."""
    n_prompt_tokens = input_ids.shape[1]
    generated = output_ids[:, n_prompt_tokens:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def _prepare_tokenizer_for_generation(tokenizer: PreTrainedTokenizerBase) -> None:
    """Switch to left-padding (required for batched ``generate``)."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Train/val response pairs
# ---------------------------------------------------------------------------


def generate_pairs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Dict[str, Any],
    prompts: List[PromptRecord],
    context_length: str,
    batch_size: int = 4,
    show_progress: bool = True,
) -> List[GenerationRecord]:
    """Generate two responses for every non-eval prompt at one context length.

    Each prompt appears twice in the forward-pass batch; independent
    sampling produces ``response_a`` and ``response_b``. The effective GPU
    batch is therefore ``2 * batch_size`` sequences.

    Args:
        model: Loaded Mistral model (with or without LoRA — for the base
            data pool we use the un-finetuned policy).
        tokenizer: Matching tokenizer. Will be reconfigured to left-pad.
        config: Project config; reads ``drive_root``, ``max_seq_length``,
            ``gen_temperature``, ``gen_top_p``, ``gen_max_new_tokens``.
        prompts: All prompt records (train, val, and eval). Only those
            with ``split != "eval"`` are processed here.
        context_length: ``"short"`` or ``"long"``.
        batch_size: Number of distinct prompts per forward pass. Each
            doubles in the batch, so effective GPU batch = 2*batch_size.
        show_progress: Display a tqdm bar.

    Returns:
        Newly generated records (existing on-disk records are not
        re-listed). Resume-aware: pairs already present in
        ``generations.jsonl`` for this ``context_length`` are skipped.
    """
    if context_length not in CONTEXT_LENGTHS:
        raise ValueError(f"context_length must be one of {CONTEXT_LENGTHS}")

    path = _generations_path(config)
    existing = _load_jsonl(path)
    done = {
        (r["prompt_id"], r["context_length"])
        for r in existing
    }

    to_do = [
        p
        for p in prompts
        if p["split"] != "eval"
        and (p["prompt_id"], context_length) not in done
    ]
    if not to_do:
        print(
            f"[generate_pairs] All {context_length} pairs already on disk; "
            "nothing to do."
        )
        return []

    _prepare_tokenizer_for_generation(tokenizer)
    field = _prompt_field_for(context_length)
    device = next(model.parameters()).device
    max_new = int(config["gen_max_new_tokens"])
    max_prompt_tokens = int(config["max_seq_length"]) - max_new

    new_records: List[GenerationRecord] = []
    iterator: Any = range(0, len(to_do), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"gen pairs [{context_length}]")

    for start in iterator:
        batch = to_do[start : start + batch_size]
        # Duplicate each prompt: positions 2i and 2i+1 share text but draw
        # independent samples, yielding response_a and response_b per prompt.
        prompt_texts = [p[field] for p in batch for _ in range(2)]

        tokenized = tokenizer(
            prompt_texts,
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
        batch_records: List[GenerationRecord] = [
            GenerationRecord(
                prompt_id=p["prompt_id"],
                context_length=context_length,
                response_a=decoded[2 * i].strip(),
                response_b=decoded[2 * i + 1].strip(),
                prompt_text=p[field],
            )
            for i, p in enumerate(batch)
        ]
        _append_jsonl(path, [dict(r) for r in batch_records])
        new_records.extend(batch_records)

    print(
        f"[generate_pairs] Wrote {len(new_records)} new {context_length} pairs "
        f"to {path}."
    )
    return new_records


# ---------------------------------------------------------------------------
# Eval baselines
# ---------------------------------------------------------------------------


def generate_baselines(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Dict[str, Any],
    prompts: List[PromptRecord],
    context_length: str,
    batch_size: int = 8,
    show_progress: bool = True,
) -> List[BaselineRecord]:
    """Generate one response per eval prompt for the held-out comparison set.

    Same generation hyperparameters as ``generate_pairs``; written to a
    separate JSONL so it never gets mixed into the preference-training pool.
    """
    if context_length not in CONTEXT_LENGTHS:
        raise ValueError(f"context_length must be one of {CONTEXT_LENGTHS}")

    path = _baselines_path(config)
    existing = _load_jsonl(path)
    done = {(r["prompt_id"], r["context_length"]) for r in existing}

    to_do = [
        p
        for p in prompts
        if p["split"] == "eval"
        and (p["prompt_id"], context_length) not in done
    ]
    if not to_do:
        print(
            f"[generate_baselines] All {context_length} baselines already on "
            "disk; nothing to do."
        )
        return []

    _prepare_tokenizer_for_generation(tokenizer)
    field = _prompt_field_for(context_length)
    device = next(model.parameters()).device
    max_new = int(config["gen_max_new_tokens"])
    max_prompt_tokens = int(config["max_seq_length"]) - max_new

    new_records: List[BaselineRecord] = []
    iterator: Any = range(0, len(to_do), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"baselines [{context_length}]")

    for start in iterator:
        batch = to_do[start : start + batch_size]
        prompt_texts = [p[field] for p in batch]

        tokenized = tokenizer(
            prompt_texts,
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
        batch_records: List[BaselineRecord] = [
            BaselineRecord(
                prompt_id=p["prompt_id"],
                context_length=context_length,
                response=decoded[i].strip(),
                prompt_text=p[field],
            )
            for i, p in enumerate(batch)
        ]
        _append_jsonl(path, [dict(r) for r in batch_records])
        new_records.extend(batch_records)

    print(
        f"[generate_baselines] Wrote {len(new_records)} new {context_length} "
        f"baselines to {path}."
    )
    return new_records
