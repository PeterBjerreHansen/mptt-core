from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from mptt.checkpoint import load_checkpoint, save_checkpoint
from mptt.config import Config, load_config
from mptt.data import get_batch, load_byte_tokens
from mptt.losses import pass_weighted_ntp_loss

from models import CausalTransformer, MemoryTapeTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: Config) -> torch.nn.Module:
    if config.model.variant == "causal_transformer":
        return CausalTransformer(config.model)
    if config.model.variant == "memory_tape":
        return MemoryTapeTransformer(config.model)
    raise ValueError(f"unknown model variant: {config.model.variant}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    config: Config,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    final_losses: list[float] = []

    for _ in range(config.training.eval_batches):
        batch = get_batch(
            val_tokens,
            batch_size=config.training.batch_size,
            block_size=config.model.block_size,
            device=config.training.device,
        )
        output = model(batch)
        ntp = pass_weighted_ntp_loss(
            output.logits_per_pass,
            batch,
            pass_weights=config.objective.pass_weights,
            pad_token_id=config.data.pad_token_id,
        )
        losses.append(float(ntp.loss.item()))
        final_losses.append(float(ntp.final_pass_loss.item()))

    model.train()
    mean_loss = sum(losses) / len(losses)
    mean_final = sum(final_losses) / len(final_losses)
    return {
        "loss": mean_loss,
        "final_pass_loss": mean_final,
        "perplexity": math.exp(min(mean_final, 20.0)),
    }


def train(config: Config, *, resume_from: str | None = None) -> float | None:
    set_seed(config.seed)

    device = torch.device(config.training.device)
    train_tokens = load_byte_tokens(config.data.train_file)
    val_tokens = load_byte_tokens(config.data.val_file)

    model = build_model(config).to(device)
    if config.training.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    start_step = 0
    best_val_loss: float | None = None

    if resume_from is not None:
        checkpoint = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_step = int(checkpoint.get("step", 0)) + 1
        best_val_loss = checkpoint.get("best_val_loss")

    run_dir = Path(config.training.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for step in range(start_step, config.training.train_steps):
        batch = get_batch(
            train_tokens,
            batch_size=config.training.batch_size,
            block_size=config.model.block_size,
            device=device,
        )

        output = model(batch)
        ntp = pass_weighted_ntp_loss(
            output.logits_per_pass,
            batch,
            pass_weights=config.objective.pass_weights,
            pad_token_id=config.data.pad_token_id,
        )

        optimizer.zero_grad(set_to_none=True)
        ntp.loss.backward()
        optimizer.step()

        if step % config.training.log_interval == 0:
            pass_loss_str = ", ".join(
                f"p{i}={loss.item():.4f}" for i, loss in enumerate(ntp.pass_losses)
            )
            print(
                f"step {step:05d} | "
                f"loss {ntp.loss.item():.4f} | "
                f"final {ntp.final_pass_loss.item():.4f} | "
                f"{pass_loss_str}"
            )

        if step % config.training.eval_interval == 0 or step == config.training.train_steps - 1:
            metrics = evaluate(model, val_tokens, config)
            val_loss = metrics["final_pass_loss"]
            print(
                f"eval step {step:05d} | "
                f"loss {metrics['loss']:.4f} | "
                f"final {metrics['final_pass_loss']:.4f} | "
                f"ppl {metrics['perplexity']:.2f}"
            )

            save_checkpoint(
                run_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                best_val_loss=best_val_loss,
            )

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    best_val_loss=best_val_loss,
                )

    return best_val_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
