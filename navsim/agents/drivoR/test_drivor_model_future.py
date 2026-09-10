import torch

from navsim.agents.drivoR.drivor_model import DrivoRModel


def test_collision_time_distribution_is_normalized() -> None:
    logits = torch.tensor([[[-20.0, -20.0, -20.0], [20.0, -20.0, -20.0]]])

    distribution = DrivoRModel.collision_time_distribution(logits)

    assert distribution.shape == (1, 2, 4)
    assert torch.allclose(distribution.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
    assert distribution[0, 0, -1] > 0.99
    assert distribution[0, 1, 0] > 0.99