import pytest
import torch

from navsim.agents.drivoR.score_module.shared_future_predictor import (
    FutureOccupancyDecoder,
    SharedFutureLatentPredictor,
)


def test_shared_future_predictor_shapes_and_gradients() -> None:
    predictor = SharedFutureLatentPredictor(
        d_model=32,
        num_future_steps=6,
        num_slots=4,
        num_layers=2,
        num_heads=4,
        d_ffn=64,
        dropout=0.0,
    )
    scene_features = torch.randn(2, 5, 32, requires_grad=True)

    future_latents = predictor(scene_features)

    assert future_latents.shape == (2, 6, 4, 32)
    future_latents.square().mean().backward()
    assert scene_features.grad is not None
    assert torch.isfinite(scene_features.grad).all()
    assert predictor.time_queries.grad is not None
    assert predictor.slot_queries.grad is not None


def test_shared_future_predictor_rejects_invalid_scene_shape() -> None:
    predictor = SharedFutureLatentPredictor(d_model=32)

    with pytest.raises(ValueError, match="scene_features"):
        predictor(torch.randn(2, 32))


def test_future_occupancy_decoder_shapes_and_gradients() -> None:
    decoder = FutureOccupancyDecoder(
        d_model=32, num_slots=4, height=32, width=48, base_channels=16
    )
    future_latents = torch.randn(2, 3, 4, 32, requires_grad=True)

    occupancy_logits, velocity = decoder(future_latents)

    assert occupancy_logits.shape == (2, 3, 3, 32, 48)
    assert velocity.shape == (2, 3, 2, 32, 48)
    (occupancy_logits.mean() + velocity.mean()).backward()
    assert future_latents.grad is not None
    assert torch.isfinite(future_latents.grad).all()