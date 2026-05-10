"""Log-probability helper used by both DPO and KTO loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def compute_log_probs(
    model: PreTrainedModel,
    input_ids_B_T: torch.Tensor,
    attention_mask_B_T: torch.Tensor,
    label_mask_B_T: torch.Tensor,
) -> torch.Tensor:
    """Compute the total log-probability of the response portion only.

    Forward the concatenated (prompt + response) sequence through the
    model, then sum ``log π(token_t | token_<t)`` across the response
    tokens only. Prompt tokens contribute zero to the result.

    Args:
        model: The language model (with or without LoRA adapters).
        input_ids_B_T: Token IDs. Shape (batch, seq_len).
        attention_mask_B_T: Attention mask, 1 for real tokens and 0 for
            padding. Shape (batch, seq_len).
        label_mask_B_T: 1 for response tokens, 0 for prompt tokens (and 0
            for padding). Shape (batch, seq_len).

    Returns:
        log_probs_B: Total ``log π(response | prompt)`` per example. Shape
            (batch,). All values are ≤ 0.
    """
    # get logits
    logits_B_T_V = model(input_ids_B_T, attention_mask=attention_mask_B_T).logits
    # shift for next-token prediction
    logits_B_Tm1_V = logits_B_T_V[:, :-1, :]
    # apply log_softmax over the vocab dimension
    log_probs_B_Tm1_V = F.log_softmax(logits_B_Tm1_V, dim=-1)
    # use torch.gather to pluck out the log-prob of the actual next token at each position
    token_log_probs_B_Tm1 = torch.gather(
        log_probs_B_Tm1_V, dim=-1, index=input_ids_B_T[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    # multiply by the shifted label mask (cast to same dtype) to zero out prompt positions
    mask_B_Tm1 = label_mask_B_T[:, 1:].to(token_log_probs_B_Tm1.dtype)
    # sum across the sequence dimension in float32 to avoid bf16 accumulation drift
    log_probs_B = (token_log_probs_B_Tm1 * mask_B_Tm1).float().sum(dim=-1)

    return log_probs_B
