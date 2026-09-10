import torch

from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss


def _inputs(occupied: bool):
    occupancy = torch.zeros(2, 3, 3, 8, 8)
    velocity = torch.zeros(2, 3, 2, 8, 8)
    if occupied:
        occupancy[:, :, 0, 2:5, 3:6] = 1
        velocity[:, :, 0, 2:5, 3:6] = 4
    targets = {
        "future_dynamic_occupancy": occupancy,
        "future_dynamic_velocity": velocity,
        "future_dynamic_valid": torch.ones(2, 3, 1, 8, 8),
    }
    predictions = {
        "pred_future_occupancy": torch.zeros_like(occupancy, requires_grad=True),
        "pred_future_velocity": torch.zeros_like(velocity, requires_grad=True),
    }
    return targets, predictions


def test_future_prediction_loss_is_finite_and_has_gradients() -> None:
    loss_module = DrivoRLoss()
    targets, predictions = _inputs(occupied=True)

    occupancy_loss, velocity_loss, future_loss = loss_module.future_prediction_loss(
        targets, predictions
    )

    assert torch.isfinite(occupancy_loss)
    assert torch.isfinite(velocity_loss)
    assert velocity_loss > 0
    future_loss.backward()
    assert predictions["pred_future_occupancy"].grad is not None
    assert predictions["pred_future_velocity"].grad is not None


def test_empty_future_has_zero_velocity_loss() -> None:
    loss_module = DrivoRLoss()
    targets, predictions = _inputs(occupied=False)

    occupancy_loss, velocity_loss, future_loss = loss_module.future_prediction_loss(
        targets, predictions
    )

    assert occupancy_loss > 0
    assert velocity_loss == 0
    assert torch.isfinite(future_loss)