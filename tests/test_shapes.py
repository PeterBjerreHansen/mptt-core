import torch

from mptt.config import ModelConfig
from models import CausalTransformer, MemoryTapeTransformer


def test_causal_transformer_shapes():
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
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    out = model(tokens)

    assert out.logits.shape == (2, config.block_size, config.vocab_size)
    assert out.hidden_states.shape == (2, config.block_size, config.n_embd)
    assert len(out.logits_per_pass) == 1
    assert len(out.hidden_states_per_pass) == 1
    assert len(out.memory_states_per_pass) == 0


def test_memory_tape_shapes():
    config = ModelConfig(
        variant="memory_tape",
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_pass=4,
        use_flash_attention=False,
    )
    model = MemoryTapeTransformer(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    out = model(tokens)

    assert out.logits.shape == (2, config.block_size, config.vocab_size)
    assert out.hidden_states.shape == (2, config.block_size, config.n_embd)
    assert len(out.logits_per_pass) == config.n_pass
    assert len(out.hidden_states_per_pass) == config.n_pass
    assert len(out.memory_states_per_pass) == config.n_pass

    for logits in out.logits_per_pass:
        assert logits.shape == (2, config.block_size, config.vocab_size)
    for hidden in out.hidden_states_per_pass:
        assert hidden.shape == (2, config.block_size, config.n_embd)
    for memory in out.memory_states_per_pass:
        assert memory.shape == (2, config.block_size, config.n_embd)
