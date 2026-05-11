"""Precompute frozen-reference log-probabilities for DPO and KTO training.

Both losses compare the current policy's log-probs against a *frozen*
reference model. Since the reference never updates during training, we
can compute its log-probs once before training and cache them to disk —
then training only needs one model in VRAM (the policy).

The reference is the base Mistral with LoRA adapters disabled. Because
LoRA initializes the B matrix to zero, "fresh LoRA" is mathematically
the same as "base model"; we use PEFT's ``disable_adapter()`` context
manager so the path is explicit.

Outputs are tensors of shape ``(num_examples,)`` keyed by ``example_idx``
from the corresponding ``DPODataset`` / ``KTODataset``. The training
loop looks them up with ``ref_logps[batch["example_idx"]]``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from models.log_probs import compute_log_probs


def _ref_logps_dir(config: Dict[str, Any]) -> str:
    return os.path.join(config["drive_root"], "ref_log_probs")


def _dpo_path(config: Dict[str, Any], split: str, context_length: str) -> str:
    return os.path.join(
        _ref_logps_dir(config),
        f"dpo_{split}_{context_length}.pt",
    )


def _kto_path(config: Dict[str, Any], split: str, context_length: str) -> str:
    return os.path.join(
        _ref_logps_dir(config),
        f"kto_{split}_{context_length}.pt",
    )


def _device_of(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def precompute_dpo_ref_logps(
    model: PreTrainedModel,
    dataloader: DataLoader[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    split: str,
    context_length: str,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute and persist reference log-probs for one DPO split.

    Args:
        model: A LoRA-wrapped PEFT model. We disable the adapter so this
            scores the underlying frozen base.
        dataloader: A loader over ``DPODataset`` items.
        config: Project config; reads ``drive_root``.
        split: One of ``"train"`` or ``"val"`` — used in the output filename.
        context_length: ``"short"`` or ``"long"`` — used in the output filename.
        show_progress: tqdm bar toggle.

    Returns:
        Dict with two ``(N,)`` CPU tensors:
        - ``"chosen"``: ``log π_ref(y_w | x)`` for every example.
        - ``"rejected"``: ``log π_ref(y_l | x)`` for every example.
        Indexed by the dataset's ``example_idx``.
    """
    path = _dpo_path(config, split, context_length)
    device = _device_of(model)
    n_examples = len(dataloader.dataset)  # type: ignore[arg-type]

    if os.path.isfile(path):
        cached = torch.load(path, map_location="cpu")
        valid = (
            isinstance(cached, dict)
            and "chosen" in cached and "rejected" in cached
            and isinstance(cached["chosen"], torch.Tensor)
            and isinstance(cached["rejected"], torch.Tensor)
            and len(cached["chosen"]) == n_examples
            and len(cached["rejected"]) == n_examples
        )
        if valid:
            print(f"[ref_log_probs] Loading cached DPO ref log-probs: {path}")
            return cached
        cached_n = (
            len(cached["chosen"])
            if isinstance(cached, dict) and isinstance(cached.get("chosen"), torch.Tensor)
            else "?"
        )
        print(
            f"[ref_log_probs] Stale DPO cache (cached n={cached_n}, "
            f"expected n={n_examples}); recomputing."
        )
        os.remove(path)

    chosen_buf = torch.full((n_examples,), float("nan"))
    rejected_buf = torch.full((n_examples,), float("nan"))

    model.eval()
    iterator: Any = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc=f"ref dpo [{split}/{context_length}]")

    with torch.no_grad(), model.disable_adapter():
        for batch in iterator:
            chosen_lp = compute_log_probs(
                model,
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
                batch["chosen_label_mask"].to(device),
            )
            rejected_lp = compute_log_probs(
                model,
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
                batch["rejected_label_mask"].to(device),
            )
            idx = batch["example_idx"].cpu()
            chosen_buf[idx] = chosen_lp.cpu().float()
            rejected_buf[idx] = rejected_lp.cpu().float()

    out = {"chosen": chosen_buf, "rejected": rejected_buf}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    print(f"[ref_log_probs] Saved DPO ref log-probs ({n_examples} examples): {path}")
    return out


def precompute_kto_ref_logps(
    model: PreTrainedModel,
    dataloader: DataLoader[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    split: str,
    context_length: str,
    show_progress: bool = True,
) -> torch.Tensor:
    """Compute and persist reference log-probs for one KTO split.

    Returns a ``(N,)`` CPU tensor indexed by ``example_idx``.
    """
    path = _kto_path(config, split, context_length)
    device = _device_of(model)
    n_examples = len(dataloader.dataset)  # type: ignore[arg-type]

    if os.path.isfile(path):
        cached = torch.load(path, map_location="cpu")
        if isinstance(cached, torch.Tensor) and len(cached) == n_examples:
            print(f"[ref_log_probs] Loading cached KTO ref log-probs: {path}")
            return cached
        cached_n = (
            len(cached) if isinstance(cached, torch.Tensor) else "?"
        )
        print(
            f"[ref_log_probs] Stale KTO cache (cached n={cached_n}, "
            f"expected n={n_examples}); recomputing."
        )
        os.remove(path)

    buf = torch.full((n_examples,), float("nan"))

    model.eval()
    iterator: Any = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc=f"ref kto [{split}/{context_length}]")

    with torch.no_grad(), model.disable_adapter():
        for batch in iterator:
            lp = compute_log_probs(
                model,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label_mask"].to(device),
            )
            idx = batch["example_idx"].cpu()
            buf[idx] = lp.cpu().float()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(buf, path)
    print(f"[ref_log_probs] Saved KTO ref log-probs ({n_examples} examples): {path}")
    return buf


def load_dpo_ref_logps(
    config: Dict[str, Any], split: str, context_length: str
) -> Dict[str, torch.Tensor]:
    """Reload precomputed DPO reference log-probs from Drive."""
    return torch.load(_dpo_path(config, split, context_length), map_location="cpu")


def load_kto_ref_logps(
    config: Dict[str, Any], split: str, context_length: str
) -> torch.Tensor:
    """Reload precomputed KTO reference log-probs from Drive."""
    return torch.load(_kto_path(config, split, context_length), map_location="cpu")
