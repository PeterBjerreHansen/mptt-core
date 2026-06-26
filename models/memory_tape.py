from __future__ import annotations

import torch
import torch.nn as nn

from mptt.config import ModelConfig
from mptt.outputs import ModelOutput
from .causal_transformer import TransformerBlock


def shift_right(memory: torch.Tensor) -> torch.Tensor:
    """Causal shift: position t receives memory emitted at t-1."""
    zeros = torch.zeros_like(memory[:, :1, :])
    return torch.cat([zeros, memory[:, :-1, :]], dim=1)


class MemoryTapeTransformer(nn.Module):
    """Minimal multi-pass memory-tape transformer.

    Each pass reuses the same transformer blocks. After each pass, the model emits
    a memory tape. The next pass receives the previous pass's memory shifted right,
    so position t never sees memory emitted from position t or later.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.memory_projection = nn.Linear(config.n_embd, config.n_embd)
        self.memory_gate = nn.Parameter(torch.tensor(float(config.memory_gate_init)))

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.mem_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.n_embd),
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying, GPT-style.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_pass(
        self,
        tokens: torch.Tensor,
        previous_memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, seq_len = tokens.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")

        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]

        if previous_memory is not None:
            memory_in = shift_right(previous_memory)
            x = x + self.memory_gate * self.memory_projection(memory_in)

        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        h = self.ln_f(x)
        logits = self.lm_head(h)
        memory = self.mem_head(h)
        return logits, h, memory

    def forward(self, tokens: torch.Tensor) -> ModelOutput:
        logits_per_pass: list[torch.Tensor] = []
        hidden_per_pass: list[torch.Tensor] = []
        memory_per_pass: list[torch.Tensor] = []

        previous_memory = None
        for _ in range(self.config.n_pass):
            logits, h, memory = self._forward_pass(tokens, previous_memory)
            logits_per_pass.append(logits)
            hidden_per_pass.append(h)
            memory_per_pass.append(memory)
            previous_memory = memory

        return ModelOutput(
            logits=logits_per_pass[-1],
            hidden_states=hidden_per_pass[-1],
            logits_per_pass=tuple(logits_per_pass),
            hidden_states_per_pass=tuple(hidden_per_pass),
            memory_states_per_pass=tuple(memory_per_pass),
        )
