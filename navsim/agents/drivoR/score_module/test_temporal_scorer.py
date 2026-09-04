from types import SimpleNamespace

import torch

from navsim.agents.drivoR.score_module.temporal_scorer import TemporalRiskScorer


def _config() -> SimpleNamespace:
    config = SimpleNamespace(
        tf_d_model=32,
        tf_d_ffn=64,
        num_poses=8,
        trajectory_sampling=SimpleNamespace(interval_length=0.5),
        temporal_scorer_num_heads=4,
        temporal_scorer_dropout=0.0,
        temporal_scorer_num_layers=2,
        scene_scorer_num_layers=1,
        nc_risk_pool_temperature=0.5,
        ttc_risk_pool_temperature=0.75,
        double_score=False,
        agent_pred=False,
        area_pred=False,
        bev_map=False,
        bev_agent=False,
    )
    config.get = lambda name, default=None: getattr(config, name, default)
    return config


def test_temporal_scorer_shapes_and_gradients() -> None:
    scorer = TemporalRiskScorer(_config())
    proposals = torch.randn(2, 3, 8, 3, requires_grad=True)
    scene = torch.randn(2, 5, 32, requires_grad=True)
    ego_token = torch.randn(2, 1, 32, requires_grad=True)
    current_velocity = torch.randn(2, 2)

    logits, clearance = scorer(
        proposals.detach(), scene, ego_token, current_velocity
    )

    assert clearance.shape == (2, 3, 8)
    assert set(logits) == {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "comfort",
    }
    assert all(value.shape == (2, 3) for value in logits.values())
    assert (
        scorer.nc_timestep_risk_head[0].weight
        is not scorer.ttc_timestep_risk_head[0].weight
    )

    (clearance.mean() + sum(value.mean() for value in logits.values())).backward()

    assert proposals.grad is None
    assert scene.grad is not None and torch.isfinite(scene.grad).all()
    assert ego_token.grad is not None and torch.isfinite(ego_token.grad).all()


def test_pose_features_include_unscaled_kinematics() -> None:
    scorer = TemporalRiskScorer(_config())
    proposals = torch.zeros(1, 1, 8, 3)
    proposals[0, 0, :, 0] = torch.arange(1, 9) * 0.5
    current_velocity = torch.tensor([[1.0, 0.0]])

    features = scorer._build_pose_features(proposals, current_velocity)

    assert features.shape == (1, 1, 8, 9)
    assert torch.allclose(features[..., 0], proposals[..., 0])
    assert torch.allclose(features[..., 1], proposals[..., 1])
    assert torch.allclose(features[..., 2], torch.zeros(1, 1, 8))
    assert torch.allclose(features[..., 3], torch.ones(1, 1, 8))
    assert torch.allclose(features[..., 4], torch.ones(1, 1, 8))
    assert torch.allclose(features[..., 5:], torch.zeros(1, 1, 8, 4))


def test_yaw_rate_uses_wrapped_angle_difference() -> None:
    scorer = TemporalRiskScorer(_config())
    proposals = torch.zeros(1, 1, 2, 3)
    proposals[..., 0, 2] = torch.pi - 0.1
    proposals[..., 1, 2] = -torch.pi + 0.1
    current_velocity = torch.zeros(1, 2)

    features = scorer._build_pose_features(proposals, current_velocity)

    assert torch.allclose(features[..., 1, 8], torch.tensor([[0.4]]), atol=1e-5)


def test_first_acceleration_uses_current_ego_velocity() -> None:
    scorer = TemporalRiskScorer(_config())
    proposals = torch.zeros(1, 1, 8, 3)
    proposals[0, 0, :, 0] = torch.arange(1, 9) * 1.5
    current_velocity = torch.tensor([[2.0, 0.0]])

    features = scorer._build_pose_features(proposals, current_velocity)

    assert torch.allclose(features[..., 0, 6], torch.tensor([[2.0]]))
    assert torch.allclose(features[..., 1:, 6:8], torch.zeros(1, 1, 7, 2))

