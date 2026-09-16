import torch

from navsim.agents.drivoR.collision_loss import differentiable_collision_loss


def _collision_loss(agent_x: float, valid: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    proposals = torch.zeros((1, 1, 1, 3), requires_grad=True)
    agent_states = torch.tensor([[[[agent_x, 0.0, 0.0, 2.0, 2.0]]]])
    agent_valid = torch.tensor([[[valid]]])
    loss = differentiable_collision_loss(
        proposals,
        agent_states,
        agent_valid,
        margin=0.7,
        filter_radius=10.0,
    )
    return loss, proposals


def test_collision_loss_is_zero_at_margin_and_beyond_filter() -> None:
    # Pacifica front is 4.049 m from rear axle; the agent half-length is 1 m.
    boundary_loss, _ = _collision_loss(5.749)
    far_loss, _ = _collision_loss(20.0)

    assert torch.isclose(boundary_loss, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(far_loss, torch.tensor(0.0), atol=1e-6)


def test_collision_loss_has_finite_proposal_gradient() -> None:
    loss, proposals = _collision_loss(4.0)
    loss.backward()

    assert loss > 0
    assert proposals.grad is not None
    assert torch.isfinite(proposals.grad).all()
    assert proposals.grad[..., 0].abs().sum() > 0


def test_collision_loss_ignores_padding() -> None:
    loss, _ = _collision_loss(0.0, valid=False)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_collision_loss_sums_over_time() -> None:
    proposals = torch.zeros((1, 1, 2, 3))
    agent_states = torch.tensor(
        [[[[4.0, 0.0, 0.0, 2.0, 2.0]], [[4.0, 0.0, 0.0, 2.0, 2.0]]]]
    )
    agent_valid = torch.ones((1, 2, 1), dtype=torch.bool)

    two_pose_loss = differentiable_collision_loss(
        proposals,
        agent_states,
        agent_valid,
        margin=0.7,
        filter_radius=10.0,
    )
    one_pose_loss = differentiable_collision_loss(
        proposals[:, :, :1],
        agent_states[:, :1],
        agent_valid[:, :1],
        margin=0.7,
        filter_radius=10.0,
    )

    assert torch.allclose(two_pose_loss, 2 * one_pose_loss)
