import torch

from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss


def test_clearance_loss_is_zero_for_exact_prediction_and_clips_target() -> None:
    loss_fn = DrivoRLoss(clearance_clip_min=-1.0, clearance_clip_max=5.0)
    target = torch.tensor([[[-0.2, 5.0]]])

    assert torch.allclose(loss_fn.clearance_loss(target, target), torch.tensor(0.0))

    outside_loss = loss_fn.clearance_loss(
        torch.tensor([[[4.0]]]), torch.tensor([[[9.0]]])
    )
    boundary_loss = loss_fn.clearance_loss(
        torch.tensor([[[4.0]]]), torch.tensor([[[5.0]]])
    )
    assert torch.allclose(outside_loss, boundary_loss)


def test_near_collision_target_has_larger_gradient_weight() -> None:
    loss_fn = DrivoRLoss(
        clearance_clip_min=-1.0,
        clearance_clip_max=5.0,
        clearance_far_weight=0.25,
        clearance_near_alpha=1.75,
        clearance_near_tau=1.0,
    )
    prediction = torch.tensor([[[0.8, 6.0]]], requires_grad=True)
    target = torch.tensor([[[-0.2, 5.0]]])

    loss_fn.clearance_loss(prediction, target).backward()

    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0] > prediction.grad[0, 0, 1]


def test_weighted_collision_sign_loss_prioritizes_collisions() -> None:
    loss_fn = DrivoRLoss(
        collision_sign_temperature=1.0,
        collision_positive_weight=5.0,
        collision_near_weight=2.0,
        collision_far_weight=0.25,
        collision_near_threshold=1.0,
    )
    prediction = torch.tensor([[[1.0, -1.0, -1.0]]], requires_grad=True)
    target = torch.tensor([[[-0.5, 0.5, 2.0]]])

    loss_fn.collision_sign_loss(prediction, target).backward()

    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0].abs() > prediction.grad[0, 0, 1].abs()
    assert prediction.grad[0, 0, 1].abs() > prediction.grad[0, 0, 2].abs()


def test_clearance_metrics_match_known_predictions() -> None:
    loss_fn = DrivoRLoss(
        collision_sign_temperature=1.0,
        collision_near_threshold=1.0,
    )
    prediction = torch.tensor([[[-0.5, 0.2, -0.1, 3.0]]])
    target = torch.tensor([[[-0.5, -0.2, 0.5, 2.0]]])

    metrics = loss_fn.clearance_metrics(prediction, target)

    assert torch.allclose(metrics["collision_positive_ratio"], torch.tensor(0.5))
    assert torch.allclose(metrics["collision_precision"], torch.tensor(0.5))
    assert torch.allclose(metrics["collision_recall"], torch.tensor(0.5))
    assert torch.allclose(metrics["collision_auprc"], torch.tensor(5.0 / 6.0))
    assert torch.allclose(metrics["clearance_sign_accuracy"], torch.tensor(0.5))
    assert torch.allclose(metrics["clearance_mae_collision"], torch.tensor(0.2))
    assert torch.allclose(metrics["clearance_mae_near"], torch.tensor(1.0 / 3.0))


def test_loss_forward_consumes_clearance_targets() -> None:
    loss_fn = DrivoRLoss(inter_weight=0.0, clearance_weight=0.2)
    proposals = torch.randn(1, 2, 8, 3, requires_grad=True)
    pred_clearance = torch.randn(1, 2, 8, requires_grad=True)
    metric_names = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "comfort",
    )
    predictions = {
        "proposals": proposals,
        "proposal_list": [proposals],
        "pred_logit": {
            name: torch.zeros(1, 2, requires_grad=True) for name in metric_names
        },
        "pred_clearance": pred_clearance,
        "pred_logit2": None,
        "pred_agents_states": None,
        "pred_area_logit": None,
        "agent_states": None,
        "agent_labels": None,
        "bev_semantic_map": None,
        "pdm_score": torch.zeros(1, 2),
    }
    targets = {"trajectory": torch.zeros(1, 8, 3)}

    def scoring_function(targets, proposals, test):
        del targets, test
        target_scores = torch.ones(proposals.shape[0], proposals.shape[1], 7)
        return (
            target_scores[..., -1],
            target_scores[..., -1].amax(dim=-1),
            target_scores,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros_like(pred_clearance),
        )

    loss_dict = loss_fn(targets, predictions, None, scoring_function)
    loss_dict["loss"].backward()

    assert torch.isfinite(loss_dict["loss"])
    assert torch.isfinite(loss_dict["clearance_loss"])
    assert torch.isfinite(loss_dict["collision_sign_loss"])
    assert "collision_positive_ratio" in loss_dict
    assert "collision_auprc" in loss_dict
    assert pred_clearance.grad is not None