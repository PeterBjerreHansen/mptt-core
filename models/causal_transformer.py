from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mptt.config import ModelConfig
from mptt.outputs import ModelOutput


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Sample one token from final-position logits.

    Args:
        logits: Tensor with shape [B, V].
        temperature: If 0, use greedy argmax. Otherwise divide logits by this value.
        top_k: Optionally keep only the top-k logits before sampling.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, V], got {tuple(logits.shape)}")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.flash = bool(config.use_flash_attention) and hasattr(F, "scaled_dot_product_attention")

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(dim, dim=-1)

        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            # PyTorch SDPA uses Flash Attention CUDA kernels when available.
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
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

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")

        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, tokens: torch.Tensor) -> ModelOutput:
        h = self.forward_hidden(tokens)
        logits = self.lm_head(h)
        return ModelOutput(
            logits=logits,
            hidden_states=h,
            logits_per_pass=(logits,),
            hidden_states_per_pass=(h,),
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation, nanoGPT-style.

        This recomputes the model on the cropped context at every step. It is
        deliberately simple and cache-free so the reference implementation stays
        easy to modify.
        """
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
