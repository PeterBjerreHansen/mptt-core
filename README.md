# mptt-core

A minimal, forkable scaffold for **multi-pass transformer training**.

This repo is intentionally small. It contains only:

1. A shared multi-pass `ModelOutput` protocol.
2. Pass-weighted next-token prediction.
3. A simple train/eval/checkpoint loop.
4. Two reference model implementations:
   - `CausalTransformer`
   - `MemoryTapeTransformer`

The goal is to be a nanoGPT-like template for multi-pass / memory-tape experiments: easy to read, copy, fork, and mutate.

## Install

```bash
pip install -e ".[dev]"
```

## Smoke run

```bash
python -m mptt.train --config configs/smoke.yaml
```

or:

```bash
bash scripts/train_smoke.sh
```

## Run tests

```bash
pytest -q
```

## Core idea

A model returns a `ModelOutput`:

```python
@dataclass
class ModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None
    logits_per_pass: tuple[torch.Tensor, ...] = ()
    hidden_states_per_pass: tuple[torch.Tensor, ...] = ()
    memory_states_per_pass: tuple[torch.Tensor, ...] = ()
```

A normal causal transformer has one pass:

```text
logits_per_pass = (logits,)
hidden_states_per_pass = (hidden_states,)
memory_states_per_pass = ()
```

A memory-tape transformer has multiple passes:

```text
logits_per_pass = (logits_1, ..., logits_K)
hidden_states_per_pass = (h_1, ..., h_K)
memory_states_per_pass = (m_1, ..., m_K)
```

The training objective is pass-weighted next-token prediction:

```text
L_NTP = sum_k w_k CE(logits_k[:, :-1], tokens[:, 1:])
```

where the pass weights are normalized internally. If `pass_weights` is unset, all passes are weighted equally.

Examples:

```yaml
objective:
  pass_weights: null          # equal weight over passes
```

```yaml
objective:
  pass_weights: [0, 0, 0.5, 0.5]  # supervise only late passes
```

```yaml
objective:
  pass_weights: [0, 0, 0, 1]      # final-pass only
```
