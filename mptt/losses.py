from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NTPLoss:
    loss: torch.Tensor
    final_pass_loss: torch.Tensor
    pass_losses: tuple[torch.Tensor, ...]
    weights: torch.Tensor


def next_token_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Cross-entropy for predicting tokens[:, 1:] from logits[:, :-1]."""
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape [B, T], got {tuple(tokens.shape)}")
    if logits.shape[:2] != tokens.shape:
        raise ValueError(
            f"logits and tokens must agree on [B, T]: "
            f"{tuple(logits.shape[:2])} vs {tuple(tokens.shape)}"
        )
    if tokens.shape[1] < 2:
        raise ValueError("sequence length must be at least 2")

    pred = logits[:, :-1, :].contiguous()
    target = tokens[:, 1:].contiguous()
    ignore_index = -100 if pad_token_id is None else pad_token_id
    return F.cross_entropy(
        pred.view(-1, pred.shape[-1]),
        target.view(-1),
        ignore_index=ignore_index,
    )


def normalize_pass_weights(
    n_pass: int,
    pass_weights: list[float] | tuple[float, ...] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_pass <= 0:
        raise ValueError("n_pass must be positive")

    if pass_weights is None:
        return torch.full((n_pass,), 1.0 / n_pass, device=device, dtype=dtype)

    if len(pass_weights) != n_pass:
        raise ValueError(f"expected {n_pass} pass weights, got {len(pass_weights)}")
    if any(w < 0 for w in pass_weights):
        raise ValueError("pass weights must be non-negative")

    weights = torch.tensor(pass_weights, device=device, dtype=dtype)
    total = weights.sum()
    if total <= 0:
        raise ValueError("at least one pass weight must be positive")
    return weights / total


def pass_weighted_ntp_loss(
    logits_per_pass: tuple[torch.Tensor, ...] | list[torch.Tensor],
    tokens: torch.Tensor,
    *,
    pass_weights: list[float] | tuple[float, ...] | None = None,
    pad_token_id: int | None = None,
) -> NTPLoss:
    """Pass-weighted NTP loss over one or more model passes."""
    if not logits_per_pass:
        raise ValueError("logits_per_pass cannot be empty")

    pass_losses = tuple(
        next_token_loss(logits, tokens, pad_token_id=pad_token_id)
        for logits in logits_per_pass
    )
    stacked = torch.stack(pass_losses)
    weights = normalize_pass_weights(
        len(pass_losses),
        pass_weights,
        device=stacked.device,
        dtype=stacked.dtype,
    )
    loss = (weights * stacked).sum()
    return NTPLoss(
        loss=loss,
        final_pass_loss=pass_losses[-1],
        pass_losses=pass_losses,
        weights=weights,
    )
