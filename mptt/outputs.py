from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelOutput:
    """Shared output protocol for single-pass and multi-pass causal LMs."""

    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None

    logits_per_pass: tuple[torch.Tensor, ...] = ()
    hidden_states_per_pass: tuple[torch.Tensor, ...] = ()
    memory_states_per_pass: tuple[torch.Tensor, ...] = ()

    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.logits_per_pass:
            self.logits_per_pass = (self.logits,)
        if self.hidden_states is not None and not self.hidden_states_per_pass:
            self.hidden_states_per_pass = (self.hidden_states,)
