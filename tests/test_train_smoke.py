from pathlib import Path

import torch

from mptt.checkpoint import load_checkpoint, save_checkpoint
from mptt.config import load_config
from mptt.data import get_batch, load_byte_tokens
from mptt.losses import pass_weighted_ntp_loss
from mptt.train import build_model


def test_train_smoke(tmp_path):
    config = load_config("configs/smoke.yaml")
    config.model.block_size = 8
    config.model.n_layer = 1
    config.model.n_head = 1
    config.model.n_embd = 16
    config.training.batch_size = 2
    config.training.output_dir = str(tmp_path / "run")
    config.training.device = "cpu"

    tokens = load_byte_tokens(config.data.train_file)
    batch = get_batch(
        tokens,
        batch_size=config.training.batch_size,
        block_size=config.model.block_size,
        device="cpu",
    )

    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    output = model(batch)
    loss = pass_weighted_ntp_loss(
        output.logits_per_pass,
        batch,
        pass_weights=config.objective.pass_weights,
        pad_token_id=config.data.pad_token_id,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.loss.backward()
    optimizer.step()

    ckpt = Path(config.training.output_dir) / "latest.pt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        step=0,
        config=config,
        best_val_loss=float(loss.final_pass_loss.item()),
    )

    fresh_model = build_model(config)
    load_checkpoint(ckpt, model=fresh_model, map_location="cpu")

    assert ckpt.exists()
