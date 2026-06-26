import torch

from mptt.config import ModelConfig
from models import CausalTransformer, MemoryTapeTransformer
from models.causal_transformer import CausalSelfAttention


def test_causal_transformer_generate_shape():
    config = ModelConfig(
        variant="causal_transformer",
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_pass=1,
        use_flash_attention=False,
    )
    model = CausalTransformer(config)
    idx = torch.randint(0, config.vocab_size, (2, 4))

    out = model.generate(idx, max_new_tokens=3, temperature=0.0)

    assert out.shape == (2, 7)
    assert torch.equal(out[:, :4], idx)


def test_memory_tape_generate_recompute_shape():
    config = ModelConfig(
        variant="memory_tape",
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_pass=2,
        use_flash_attention=False,
    )
    model = MemoryTapeTransformer(config)
    idx = torch.randint(0, config.vocab_size, (2, 4))

    out = model.generate(idx, max_new_tokens=2, temperature=0.0, mode="recompute")

    assert out.shape == (2, 6)
    assert torch.equal(out[:, :4], idx)


def test_memory_tape_generate_last_pass_recurrent_shape():
    config = ModelConfig(
        variant="memory_tape",
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_pass=2,
        use_flash_attention=False,
    )
    model = MemoryTapeTransformer(config)
    idx = torch.randint(0, config.vocab_size, (2, 4))

    out = model.generate(idx, max_new_tokens=2, temperature=0.0, mode="last_pass_recurrent")

    assert out.shape == (2, 6)
    assert torch.equal(out[:, :4], idx)


def test_memory_tape_unknown_generation_mode_fails():
    config = ModelConfig(
        variant="memory_tape",
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_pass=2,
        use_flash_attention=False,
    )
    model = MemoryTapeTransformer(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))

    try:
        model.generate(idx, max_new_tokens=1, mode="bad_mode")
    except ValueError as exc:
        assert "unknown generation mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_attention_uses_flash_flag_when_available():
    config = ModelConfig(
        variant="causal_transformer",
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_pass=1,
    )
    attn = CausalSelfAttention(config)
    assert attn.flash == hasattr(torch.nn.functional, "scaled_dot_product_attention")
