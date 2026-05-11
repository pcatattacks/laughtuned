"""Tokenized preference datasets for DPO and KTO training.

Turns the rubric judging output (Step 5) into:
- **DPO examples**: ``(prompt, chosen_response, rejected_response)``,
  driven by the rubric's WINNER.
- **KTO examples**: ``(prompt, response, label ∈ {1.0, 0.0})``, driven
  by the per-response composite score thresholds in ``CONFIG``.

Each example carries the prompt's train/val/eval split inherited from
Step 3. PyTorch ``Dataset`` classes pre-tokenize on construction so the
training loop just indexes tensors at every step.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from data.build_prompts import PromptRecord
from data.judge import RubricRecord


# ---------------------------------------------------------------------------
# Example schemas
# ---------------------------------------------------------------------------


class DPOExample(TypedDict):
    prompt_id: str
    context_length: str
    split: str
    prompt: str
    chosen: str
    rejected: str


class KTOExample(TypedDict):
    prompt_id: str
    context_length: str
    split: str
    prompt: str
    response: str
    label: float  # 1.0 = desirable, 0.0 = undesirable


def build_dpo_examples(
    rubric_records: List[RubricRecord],
    prompts: List[PromptRecord],
) -> List[DPOExample]:
    """For each judged pair (non-eval), produce one (chosen, rejected) example."""
    prompt_lookup = {p["prompt_id"]: p for p in prompts}
    examples: List[DPOExample] = []
    for r in rubric_records:
        p = prompt_lookup.get(r["prompt_id"])
        if p is None or p["split"] == "eval":
            continue
        prompt_text = p[f"prompt_text_{r['context_length']}"]  # type: ignore[literal-required]
        if r["winner"] == "a":
            chosen, rejected = r["response_a"], r["response_b"]
        else:
            chosen, rejected = r["response_b"], r["response_a"]
        examples.append(
            DPOExample(
                prompt_id=r["prompt_id"],
                context_length=r["context_length"],
                split=p["split"],
                prompt=prompt_text,
                chosen=chosen,
                rejected=rejected,
            )
        )
    return examples


def build_kto_examples(
    rubric_records: List[RubricRecord],
    prompts: List[PromptRecord],
    desirable_threshold: float,
    undesirable_threshold: float,
) -> List[KTOExample]:
    """One example per response (a and b separately), excluding dropped labels.

    Labels are derived from each response's composite score at call time
    using the supplied thresholds — the ``kto_label_a`` / ``kto_label_b``
    fields stored in the rubric records are ignored. This decouples
    threshold tuning from the expensive judging step: change the thresholds
    here, rebuild, and skip re-judging.
    """
    from data.judge import compute_kto_label  # late import to avoid cycle

    prompt_lookup = {p["prompt_id"]: p for p in prompts}
    examples: List[KTOExample] = []
    for r in rubric_records:
        p = prompt_lookup.get(r["prompt_id"])
        if p is None or p["split"] == "eval":
            continue
        prompt_text = p[f"prompt_text_{r['context_length']}"]  # type: ignore[literal-required]
        for response, composite in (
            (r["response_a"], r["composite_a"]),
            (r["response_b"], r["composite_b"]),
        ):
            label_str = compute_kto_label(
                composite, desirable_threshold, undesirable_threshold
            )
            if label_str is None:
                continue
            examples.append(
                KTOExample(
                    prompt_id=r["prompt_id"],
                    context_length=r["context_length"],
                    split=p["split"],
                    prompt=prompt_text,
                    response=response,
                    label=1.0 if label_str == "desirable" else 0.0,
                )
            )
    return examples


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _tokenize_prompt_response(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (input_ids, attention_mask, label_mask) of shape (max_length,).

    ``label_mask`` is 1 only on response tokens and 0 on prompt tokens and
    padding. If the concatenation exceeds ``max_length``, the prompt is
    truncated from the FRONT (preserving the response in full) — falling
    back to truncating the response from the right only when the prompt
    is already shorter than ``max_length``.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    overflow = (len(prompt_ids) + len(response_ids)) - max_length
    if overflow > 0:
        if overflow < len(prompt_ids):
            prompt_ids = prompt_ids[overflow:]
        else:
            # Response itself is too long; preserve some prompt, trim response right.
            keep_prompt = max(0, max_length // 2)
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt else []
            remaining = max_length - len(prompt_ids)
            response_ids = response_ids[:remaining]

    combined = prompt_ids + response_ids
    n_prompt = len(prompt_ids)
    n_total = len(combined)
    pad_len = max_length - n_total

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = combined + [pad_id] * pad_len
    attention_mask = [1] * n_total + [0] * pad_len
    label_mask = [0] * n_prompt + [1] * (n_total - n_prompt) + [0] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(label_mask, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# PyTorch Datasets (pre-tokenized for speed)
# ---------------------------------------------------------------------------


class DPODataset(Dataset[Dict[str, torch.Tensor]]):
    """Pre-tokenized DPO dataset; each item has chosen & rejected tensors."""

    def __init__(
        self,
        examples: List[DPOExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.items: List[Dict[str, torch.Tensor]] = []
        for idx, ex in enumerate(examples):
            chosen_ids, chosen_attn, chosen_lbl = _tokenize_prompt_response(
                tokenizer, ex["prompt"], ex["chosen"], max_length
            )
            rejected_ids, rejected_attn, rejected_lbl = _tokenize_prompt_response(
                tokenizer, ex["prompt"], ex["rejected"], max_length
            )
            self.items.append(
                {
                    "chosen_input_ids": chosen_ids,
                    "chosen_attention_mask": chosen_attn,
                    "chosen_label_mask": chosen_lbl,
                    "rejected_input_ids": rejected_ids,
                    "rejected_attention_mask": rejected_attn,
                    "rejected_label_mask": rejected_lbl,
                    "example_idx": torch.tensor(idx, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


class KTODataset(Dataset[Dict[str, torch.Tensor]]):
    """Pre-tokenized KTO dataset; each item is one prompt+response with a label."""

    def __init__(
        self,
        examples: List[KTOExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.items: List[Dict[str, torch.Tensor]] = []
        for idx, ex in enumerate(examples):
            input_ids, attention_mask, label_mask = _tokenize_prompt_response(
                tokenizer, ex["prompt"], ex["response"], max_length
            )
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label_mask": label_mask,
                    "label": torch.tensor(ex["label"], dtype=torch.float32),
                    "example_idx": torch.tensor(idx, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------


def _filter_examples(
    examples: List[Any],
    split: str,
    context_length: Optional[str],
) -> List[Any]:
    out = [e for e in examples if e["split"] == split]
    if context_length is not None:
        out = [e for e in out if e["context_length"] == context_length]
    return out


def make_dpo_loaders(
    examples: List[DPOExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    context_length: Optional[str] = None,
) -> Tuple[DataLoader[Dict[str, torch.Tensor]], DataLoader[Dict[str, torch.Tensor]]]:
    """Return (train_loader, val_loader) for the DPO dataset."""
    train_ex = _filter_examples(examples, "train", context_length)
    val_ex = _filter_examples(examples, "val", context_length)
    train_ds = DPODataset(train_ex, tokenizer, max_length)
    val_ds = DPODataset(val_ex, tokenizer, max_length)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def make_kto_loaders(
    examples: List[KTOExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    context_length: Optional[str] = None,
) -> Tuple[DataLoader[Dict[str, torch.Tensor]], DataLoader[Dict[str, torch.Tensor]]]:
    """Return (train_loader, val_loader) for the KTO dataset."""
    train_ex = _filter_examples(examples, "train", context_length)
    val_ex = _filter_examples(examples, "val", context_length)
    train_ds = KTODataset(train_ex, tokenizer, max_length)
    val_ds = KTODataset(val_ex, tokenizer, max_length)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Disk persistence (for inspection / resume)
# ---------------------------------------------------------------------------


def _dpo_examples_path(config: Dict[str, Any]) -> str:
    return os.path.join(
        config["drive_root"], "data", "preferences", "dpo_examples.jsonl"
    )


def _kto_examples_path(config: Dict[str, Any]) -> str:
    return os.path.join(
        config["drive_root"], "data", "preferences", "kto_examples.jsonl"
    )


def save_examples(
    config: Dict[str, Any],
    dpo_examples: List[DPOExample],
    kto_examples: List[KTOExample],
) -> None:
    """Persist both example lists as JSONL alongside the rubric records."""
    for path, examples in (
        (_dpo_examples_path(config), dpo_examples),
        (_kto_examples_path(config), kto_examples),
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
