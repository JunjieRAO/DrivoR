from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

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
        future_post_fusion_temporal_layers=1,
        future_interval=0.5,
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

    logits, clearance, nc_timestep_risk = scorer(
        proposals.detach(), scene, ego_token, current_velocity
    )

    assert clearance.shape == (2, 3, 8)
    assert nc_timestep_risk.shape == (2, 3, 8)
    assert set(logits) == {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "comfort",
    }
    assert all(value.shape == (2, 3) for value in logits.values())
    assert not hasattr(scorer, "ttc_timestep_risk_head")
    assert scorer.nc_pool_query is not scorer.ttc_pool_query
    assert scorer.dac_pool_query is not scorer.ddc_pool_query
    assert scorer.nc_head[0].in_features == 32
    assert scorer.ttc_head[0].in_features == 32

    (clearance.mean() + sum(value.mean() for value in logits.values())).backward()

    assert proposals.grad is None
    assert scene.grad is not None and torch.isfinite(scene.grad).all()
    assert ego_token.grad is not None and torch.isfinite(ego_token.grad).all()


def test_metric_query_pooling_keeps_full_feature_dimension() -> None:
    scorer = TemporalRiskScorer(_config())
    tokens = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8)
    query = torch.zeros(8)

    features = scorer._metric_query_pool(tokens, query)

    assert features.shape == (1, 1, 8)
    assert torch.allclose(features, tokens.mean(dim=-2))


def test_collision_hazard_does_not_control_subscore_pooling() -> None:
    scorer = TemporalRiskScorer(_config()).eval()
    inputs = (
        torch.randn(1, 2, 8, 3),
        torch.randn(1, 5, 32),
        torch.randn(1, 1, 32),
        torch.randn(1, 2),
        torch.randn(1, 8, 4, 32),
    )

    logits_before, _, risk_before = scorer(*inputs)
    with torch.no_grad():
        scorer.nc_timestep_risk_head[-1].bias.add_(20.0)
    logits_after, _, risk_after = scorer(*inputs)

    assert not torch.allclose(risk_before, risk_after)
    for name in logits_before:
        assert torch.allclose(logits_before[name], logits_after[name])


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


def test_time_aligned_future_fusion_shapes_and_gradients() -> None:
    scorer = TemporalRiskScorer(_config())
    proposals = torch.randn(2, 3, 8, 3)
    scene = torch.randn(2, 5, 32, requires_grad=True)
    ego_token = torch.randn(2, 1, 32, requires_grad=True)
    current_velocity = torch.randn(2, 2)
    future_latents = torch.randn(2, 8, 4, 32, requires_grad=True)

    logits, clearance, timestep_risk = scorer(
        proposals, scene, ego_token, current_velocity, future_latents
    )

    assert clearance.shape == (2, 3, 8)
    assert timestep_risk.shape == (2, 3, 8)
    (clearance.mean() + sum(value.mean() for value in logits.values())).backward()
    assert future_latents.grad is not None
    assert torch.isfinite(future_latents.grad).all()


def test_future_latents_require_exact_timestep_alignment() -> None:
    scorer = TemporalRiskScorer(_config())

    with pytest.raises(ValueError, match="Expected 8 future steps"):
        scorer(
            torch.randn(1, 2, 8, 3),
            torch.randn(1, 5, 32),
            torch.randn(1, 1, 32),
            torch.randn(1, 2),
            torch.randn(1, 7, 4, 32),
        )


def test_future_attention_queries_pre_scene_proposal_tokens() -> None:
    class AddSceneContext(nn.Module):
        def forward(self, tokens, scene_features):
            return tokens + 10.0

    class CaptureFutureAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = None

        def forward(self, proposal_tokens, future_latents):
            self.queries = proposal_tokens.detach().clone()
            return torch.zeros_like(proposal_tokens)

    scorer = TemporalRiskScorer(_config())
    scorer.scene_layers = nn.ModuleList([AddSceneContext()])
    capture = CaptureFutureAttention()
    scorer.future_attention = capture
    proposals = torch.randn(1, 2, 8, 3)
    velocity = torch.randn(1, 2)
    ego_token = torch.randn(1, 1, 32)
    pose_features = scorer._build_pose_features(proposals, velocity)
    expected = scorer.pose_encoder(pose_features)
    expected = expected + scorer.temporal_embedding + ego_token[:, None, :, :]
    expected = scorer.temporal_encoder(expected.reshape(2, 8, 32)).reshape(
        1, 2, 8, 32
    )

    scorer(
        proposals,
        torch.randn(1, 5, 32),
        ego_token,
        velocity,
        torch.randn(1, 8, 4, 32),
    )

    assert torch.allclose(capture.queries, expected)


def test_scorer_rejects_mismatched_future_interval() -> None:
    config = _config()
    config.future_interval = 0.25

    with pytest.raises(ValueError, match="same interval"):
        TemporalRiskScorer(config)

