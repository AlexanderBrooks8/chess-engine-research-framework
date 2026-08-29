from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_cross_entropy_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if policy_logits.shape != target_policy.shape:
        raise ValueError(
            f"policy_logits and target_policy must have same shape. "
            f"Got {policy_logits.shape} and {target_policy.shape}."
        )

    if legal_mask is not None:
        if legal_mask.shape != policy_logits.shape:
            raise ValueError(
                f"legal_mask must have same shape as policy_logits. "
                f"Got {legal_mask.shape} and {policy_logits.shape}."
            )

        policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

    log_probs = F.log_softmax(policy_logits, dim=-1)

    if legal_mask is not None:
        log_probs = log_probs.masked_fill(~legal_mask, 0.0)
        target_policy = target_policy.masked_fill(~legal_mask, 0.0)

    return -(target_policy * log_probs).sum(dim=-1).mean()


def value_mse_loss(
    predicted_value: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    if predicted_value.shape != target_value.shape:
        raise ValueError(
            f"predicted_value and target_value must have same shape. "
            f"Got {predicted_value.shape} and {target_value.shape}."
        )

    return F.mse_loss(predicted_value, target_value)


def combined_policy_value_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    predicted_value: torch.Tensor,
    target_value: torch.Tensor,
    value_weight: float = 1.0,
    legal_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_loss = policy_cross_entropy_loss(
        policy_logits=policy_logits,
        target_policy=target_policy,
        legal_mask=legal_mask,
    )
    value_loss = value_mse_loss(predicted_value, target_value)
    total_loss = policy_loss + value_weight * value_loss

    return total_loss, policy_loss, value_loss