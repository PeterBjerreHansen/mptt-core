from __future__ import annotations

from pathlib import Path

import torch


def load_byte_tokens(path: str | Path) -> torch.Tensor:
    """Load a text file as byte-level tokens in [0, 255]."""
    data = Path(path).read_bytes()
    if not data:
        raise ValueError(f"empty data file: {path}")
    return torch.tensor(list(data), dtype=torch.long)


def _ensure_min_length(tokens: torch.Tensor, min_len: int) -> torch.Tensor:
    if tokens.numel() >= min_len:
        return tokens
    repeats = (min_len + tokens.numel() - 1) // tokens.numel()
    return tokens.repeat(repeats)


def get_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample a batch of contiguous token blocks.

    Returns tokens with shape [B, T]. Losses use tokens[:, 1:] as labels.
    """
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    tokens = _ensure_min_length(tokens, block_size + 1)
    max_start = tokens.numel() - block_size
    starts = torch.randint(0, max_start, (batch_size,))
    batch = torch.stack([tokens[i : i + block_size] for i in starts])
    return batch.to(device)
