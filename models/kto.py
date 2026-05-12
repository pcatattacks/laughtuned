"""Kahneman-Tversky Optimization (KTO) loss for binary-labeled responses.

KTO is an asymmetric loss inspired by prospect theory: people are
loss-averse, so undesirable outcomes should be pushed away more
aggressively than desirable ones are pulled in.

For each example with implicit reward ``r_θ(x, y) = log π_θ(y|x) - log π_ref(y|x)``
and a per-batch reference baseline ``z_ref = mean(r_θ).detach()``:

    if y is desirable:    v(x, y) = σ(β · (r_θ - z_ref))
    if y is undesirable:  v(x, y) = σ(β · (z_ref - r_θ))
    Loss = E[ λ_y · (1 - v(x, y)) ]

Weights ``λ_desirable`` and ``λ_undesirable`` are inverse class
frequencies in the batch, so the loss stays balanced even when one
class is rare.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def kto_loss(
    policy_logps_B: torch.Tensor,
    reference_logps_B: torch.Tensor,
    labels_B: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the KTO loss for a batch of binary-labeled responses.

    Implements:
        log_ratio = policy_logps - reference_logps                  # (batch,)
        z_ref     = log_ratio.mean().detach()                        # scalar
        v(d=1)    = sigmoid(β · (log_ratio - z_ref))
        v(d=0)    = sigmoid(β · (z_ref - log_ratio))
        per_example_loss = lambda_y * (1 - v)
        loss      = per_example_loss.mean()

    Args:
        policy_logps_B: ``log π_θ(y | x)``. Shape (batch,).
        reference_logps_B: ``log π_ref(y | x)``. Shape (batch,).
        labels_B: Binary labels — 1.0 for desirable (thumbs up),
            0.0 for undesirable (thumbs down). Shape (batch,).
        beta: Temperature controlling sensitivity. Typical: 0.1.

    Returns:
        Tuple of ``(loss_scalar, metrics_dict)`` where ``loss_scalar`` is
        the mean KTO loss for backpropagation and ``metrics_dict`` is a
        dict of detached floats:

        - ``loss``: same numeric as ``loss_scalar``.
        - ``z_ref``: detached batch-mean log_ratio.
        - ``log_ratio_mean``: mean of ``policy_logps - reference_logps``.
        - ``v_desirable_mean``: mean ``v`` over desirable rows (NaN if none).
        - ``v_undesirable_mean``: mean ``v`` over undesirable rows (NaN if none).
        - ``lambda_desirable``: weight applied to desirable rows.
        - ``lambda_undesirable``: weight applied to undesirable rows.

    Single-class batches:
        When all labels in the batch are the same class, the missing-class
        divisor is clamped to 1.0 (avoiding inf weights) and the missing-
        class metric is reported as NaN. z_ref still collapses onto the
        present class for that batch, which biases the loss; the training
        loop should prefer larger effective batches when class balance is
        skewed to mitigate this.

    Note on gradient accumulation:
        For correctness, ``z_ref`` should ideally be computed across the
        full effective batch (``batch_size × gradient_accumulation_steps``),
        not per-microbatch. The training loop can accumulate ``log_ratio``
        values across microbatches before invoking this loss. Per-microbatch
        ``z_ref`` is simpler but biased when per-microbatch class balance
        is skewed.
    """
    log_ratio_B = policy_logps_B - reference_logps_B
    z_ref = log_ratio_B.mean().detach()
    v_desirable_B = F.sigmoid(beta * (log_ratio_B - z_ref))
    v_undesirable_B = F.sigmoid(beta * (z_ref - log_ratio_B))

    # Inverse-frequency class weights. Clamp the divisors so a microbatch
    # containing only one class doesn't produce inf weights (val passes with
    # eval_steps=10 and small val sets routinely hit single-class batches).
    N = len(labels_B)
    n_desirable = labels_B.sum().clamp(min=1.0)
    n_undesirable = (N - labels_B.sum()).clamp(min=1.0)
    lambda_desirable = N / (2 * n_desirable)
    lambda_undesirable = N / (2 * n_undesirable)

    v_used_B = torch.where(labels_B == 1.0, v_desirable_B, v_undesirable_B)
    weight_B = torch.where(labels_B == 1.0, lambda_desirable, lambda_undesirable)
    per_example_loss_B = weight_B * (1 - v_used_B)

    loss = per_example_loss_B.mean()

    # Mean v restricted to each label class so the metric describes that
    # class only. Returns NaN when the class is absent — caller treats NaN
    # as "no signal" rather than misleading per-batch averages.
    desirable_mask = labels_B == 1.0
    undesirable_mask = ~desirable_mask
    v_desirable_mean = (
        v_desirable_B[desirable_mask].mean().item()
        if desirable_mask.any() else float("nan")
    )
    v_undesirable_mean = (
        v_undesirable_B[undesirable_mask].mean().item()
        if undesirable_mask.any() else float("nan")
    )

    metrics = {
        "loss": loss.item(),
        "z_ref": z_ref.item(),
        "log_ratio_mean": log_ratio_B.mean().item(),
        "v_desirable_mean": v_desirable_mean,
        "v_undesirable_mean": v_undesirable_mean,
        "lambda_desirable": lambda_desirable.item(),
        "lambda_undesirable": lambda_undesirable.item(),
    }
    return loss, metrics