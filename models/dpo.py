"""Direct Preference Optimization (DPO) loss for paired preference data.

The DPO objective is:

    L_DPO = -E[ log σ( β · (r_θ(x, y_w) - r_θ(x, y_l)) ) ]

where the implicit reward is the policy / reference log-prob ratio:

    r_θ(x, y) = log π_θ(y | x) - log π_ref(y | x)

Maximizing the gap between chosen and rejected rewards is equivalent to
maximizing the policy's preference for ``y_w`` over ``y_l`` relative to
the reference distribution.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps_B: torch.Tensor,
    policy_rejected_logps_B: torch.Tensor,
    reference_chosen_logps_B: torch.Tensor,
    reference_rejected_logps_B: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the DPO loss for a batch of preference pairs.

    Implements:
        chosen_reward    = β · (log π_θ(y_w|x) - log π_ref(y_w|x))
        rejected_reward  = β · (log π_θ(y_l|x) - log π_ref(y_l|x))
        loss             = -log σ(chosen_reward - rejected_reward).mean()

    Args:
        policy_chosen_logps_B: ``log π_θ(y_w | x)``. Shape (batch,).
        policy_rejected_logps_B: ``log π_θ(y_l | x)``. Shape (batch,).
        reference_chosen_logps_B: ``log π_ref(y_w | x)``. Shape (batch,).
        reference_rejected_logps_B: ``log π_ref(y_l | x)``. Shape (batch,).
        beta: Temperature controlling separation strength. Typical: 0.1.

    Returns:
        Tuple of ``(loss_scalar, metrics_dict)`` where ``loss_scalar`` is
        the mean DPO loss for backpropagation and ``metrics_dict`` is a
        dict of detached floats:

        - ``loss``: same numeric as ``loss_scalar``.
        - ``chosen_rewards_mean``: mean of ``β · (policy_chosen - ref_chosen)``.
        - ``rejected_rewards_mean``: mean of ``β · (policy_rejected - ref_rejected)``.
        - ``reward_margin``: ``chosen_rewards_mean - rejected_rewards_mean``.
        - ``accuracy``: fraction of pairs where chosen_reward > rejected_reward.

    Raises:
        ValueError: If the four input tensors have mismatched shapes.
    """
    chosen_reward_B = beta * (policy_chosen_logps_B - reference_chosen_logps_B)
    rejected_reward_B = beta * (policy_rejected_logps_B - reference_rejected_logps_B)
    loss = -F.log_sigmoid(chosen_reward_B - rejected_reward_B).mean()
    metrics = {
        "loss": loss.item(),
        "chosen_rewards_mean": chosen_reward_B.mean().item(),
        "rejected_rewards_mean": rejected_reward_B.mean().item(),
        "reward_margin": (chosen_reward_B - rejected_reward_B).mean().item(),
        "accuracy": (chosen_reward_B > rejected_reward_B).count_nonzero().item() / len(chosen_reward_B),
    }
    return loss, metrics
