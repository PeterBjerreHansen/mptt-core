import pytest
import torch

from mptt.losses import normalize_pass_weights, pass_weighted_ntp_loss


def test_equal_weights_matches_mean():
    torch.manual_seed(0)
    tokens = torch.randint(0, 10, (2, 6))
    logits = tuple(torch.randn(2, 6, 10) for _ in range(4))

    loss = pass_weighted_ntp_loss(logits, tokens, pass_weights=None)
    expected = torch.stack(loss.pass_losses).mean()

    assert torch.allclose(loss.loss, expected)
    assert torch.allclose(loss.weights, torch.full((4,), 0.25))


def test_final_only_weight_matches_final_pass():
    torch.manual_seed(0)
    tokens = torch.randint(0, 10, (2, 6))
    logits = tuple(torch.randn(2, 6, 10) for _ in range(4))

    loss = pass_weighted_ntp_loss(logits, tokens, pass_weights=[0, 0, 0, 1])

    assert torch.allclose(loss.loss, loss.pass_losses[-1])
    assert torch.allclose(loss.weights, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_pass_weights_are_normalized():
    weights = normalize_pass_weights(
        4,
        [0, 0, 5, 5],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(weights, torch.tensor([0.0, 0.0, 0.5, 0.5]))


def test_invalid_weights_fail():
    tokens = torch.randint(0, 10, (2, 6))
    logits = tuple(torch.randn(2, 6, 10) for _ in range(4))

    with pytest.raises(ValueError):
        pass_weighted_ntp_loss(logits, tokens, pass_weights=[0, 0, 1])

    with pytest.raises(ValueError):
        pass_weighted_ntp_loss(logits, tokens, pass_weights=[0, 0, 0, 0])

    with pytest.raises(ValueError):
        pass_weighted_ntp_loss(logits, tokens, pass_weights=[0, -1, 1, 1])
