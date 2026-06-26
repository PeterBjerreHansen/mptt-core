from __future__ import annotations

import torch
import torch.nn as nn

from mptt.config import ModelConfig
from mptt.outputs import ModelOutput
from .causal_transformer import TransformerBlock, sample_next_token


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
        if previous_memory is not None and previous_memory.shape[:2] != tokens.shape:
            raise ValueError(
                "previous_memory must align with tokens on [B, T]: "
                f"got {tuple(previous_memory.shape[:2])}, expected {tuple(tokens.shape)}"
            )

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

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        mode: str = "recompute",
    ) -> torch.Tensor:
        """Autoregressive generation for MemoryTape models.

        Modes:
            recompute:
                Recompute all MPTT passes over the cropped context for every
                generated token. This is slow but exactly matches the training
                forward path.

            last_pass_recurrent:
                Run full MPTT on the prompt once, then generate by repeatedly
                running only the last-pass recurrence with the previous final-pass
                memory tape. This is closer to recurrent use of the final memory
                tape and is cheaper than full multi-pass recomputation.
        """
        if mode == "recompute":
            return self.generate_recompute(
                idx,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        if mode == "last_pass_recurrent":
            return self.generate_last_pass_recurrent(
                idx,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        raise ValueError(f"unknown generation mode: {mode}")

    @torch.no_grad()
    def generate_recompute(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate by recomputing all passes at every step."""
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits = self(idx_cond).logits[:, -1, :]
                idx_next = sample_next_token(logits, temperature=temperature, top_k=top_k)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate_last_pass_recurrent(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate using a recurrent final-pass memory tape after prompt prefill.

        The prompt is prefetched with the normal full multi-pass forward. After
        that, each new token is generated by running a single `_forward_pass` over
        the cropped context with the previous final-pass memory. This preserves the
        core causal memory rule: position t reads memory emitted at t-1.
        """
        was_training = self.training
        self.eval()
        try:
            idx_cond = idx[:, -self.config.block_size :]
            output = self(idx_cond)
            memory = output.memory_states_per_pass[-1]
            logits = output.logits[:, -1, :]

            for step in range(max_new_tokens):
                idx_next = sample_next_token(logits, temperature=temperature, top_k=top_k)
                idx = torch.cat((idx, idx_next), dim=1)

                # No next-step logits are needed after the final sample.
                if step == max_new_tokens - 1:
                    break

                # Append a placeholder memory for the newly sampled token, then
                # crop tokens and memories together to the context window.
                placeholder = torch.zeros(
                    memory.shape[0],
                    1,
                    memory.shape[2],
                    dtype=memory.dtype,
                    device=memory.device,
                )
                memory = torch.cat((memory, placeholder), dim=1)

                idx_cond = idx[:, -self.config.block_size :]
                memory_cond = memory[:, -self.config.block_size :, :]
                logits, _, memory = self._forward_pass(idx_cond, memory_cond)
                logits = logits[:, -1, :]

            return idx
        finally:
            if was_training:
                self.train()
