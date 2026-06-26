from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml
import torch


@dataclass
class ModelConfig:
    variant: str = "causal_transformer"  # causal_transformer | memory_tape
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_pass: int = 1
    dropout: float = 0.0
    memory_gate_init: float = 0.2

    def validate(self) -> None:
        if self.variant not in {"causal_transformer", "memory_tape"}:
            raise ValueError(f"unknown model variant: {self.variant}")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_pass <= 0:
            raise ValueError("n_pass must be positive")
        if self.variant == "causal_transformer" and self.n_pass != 1:
            raise ValueError("causal_transformer must use n_pass=1")


@dataclass
class ObjectiveConfig:
    pass_weights: list[float] | None = None

    def validate(self, n_pass: int) -> None:
        if self.pass_weights is None:
            return
        if len(self.pass_weights) != n_pass:
            raise ValueError(
                f"pass_weights length must equal n_pass: "
                f"got {len(self.pass_weights)}, expected {n_pass}"
            )
        if any(w < 0 for w in self.pass_weights):
            raise ValueError("pass_weights must be non-negative")
        if sum(self.pass_weights) <= 0:
            raise ValueError("at least one pass weight must be positive")


@dataclass
class TrainingConfig:
    train_steps: int = 1000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: bool = False
    output_dir: str = "runs/smoke"

    def validate(self) -> None:
        if self.train_steps <= 0:
            raise ValueError("train_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.eval_interval <= 0 or self.eval_batches <= 0:
            raise ValueError("eval_interval and eval_batches must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


@dataclass
class DataConfig:
    train_file: str = "tests/fixtures/tiny.txt"
    val_file: str = "tests/fixtures/tiny.txt"
    pad_token_id: int | None = None

    def validate(self) -> None:
        if not Path(self.train_file).exists():
            raise FileNotFoundError(f"train_file not found: {self.train_file}")
        if not Path(self.val_file).exists():
            raise FileNotFoundError(f"val_file not found: {self.val_file}")


@dataclass
class Config:
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        self.model.validate()
        self.objective.validate(self.model.n_pass)
        self.training.validate()
        self.data.validate()


def _construct_dataclass(cls: type, values: dict[str, Any]) -> Any:
    kwargs = {}
    for f in fields(cls):
        if f.name not in values:
            continue
        value = values[f.name]
        if f.name == "model":
            kwargs[f.name] = _construct_dataclass(ModelConfig, value)
        elif f.name == "objective":
            kwargs[f.name] = _construct_dataclass(ObjectiveConfig, value)
        elif f.name == "training":
            kwargs[f.name] = _construct_dataclass(TrainingConfig, value)
        elif f.name == "data":
            kwargs[f.name] = _construct_dataclass(DataConfig, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = _construct_dataclass(Config, raw)
    config.validate()
    return config


def config_to_dict(config: Config) -> dict[str, Any]:
    return asdict(config)
