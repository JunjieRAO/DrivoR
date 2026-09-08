from types import SimpleNamespace
from itertools import combinations

import torch
import torch.nn as nn

from navsim.agents.drivoR.drivor_model import DrivoRModel


class _IdentityTrajectoryDecoder(nn.Module):
    def forward(self, tokens, scene_features):
        return tokens.unsqueeze(0)


class _CountingScorerAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, tokens, scene_features):
        self.calls.append(tokens.shape)
        return tokens


class _StubScorer(nn.Module):
    def forward(self, proposals, proposal_features):
        logits = {
            name: proposal_features[..., 0]
            for name in (
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "driving_direction_compliance",
                "time_to_collision_within_bound",
                "ego_progress",
                "comfort",
            )
        }
        return logits, None, None, None, None, None, None


def _augmentation_config():
    return SimpleNamespace(
        scorer_augmentation=True,
        scorer_aug_selected_num=16,
        scorer_aug_variants_num=4,
        scorer_aug_small_axes=[1.0, 2.0],
        scorer_aug_large_axes=[3.0, 5.0],
        scorer_aug_min_radius=0.3,
    )


def _straight_proposals(batch_size=2, proposal_num=64, num_poses=8):
    proposals = torch.zeros(batch_size, proposal_num, num_poses, 3)
    proposals[..., 0] = torch.arange(1, num_poses + 1) * 5.0
    return proposals


def test_scorer_proposal_augmentation_geometry():
    model = DrivoRModel.__new__(DrivoRModel)
    nn.Module.__init__(model)
    model._config = _augmentation_config()
    proposals = _straight_proposals()
    original = proposals.clone()

    torch.manual_seed(7)
    augmented = model._augment_scorer_proposals(proposals)

    assert augmented.shape == (2, 64, 8, 3)
    assert torch.equal(proposals, original)

    endpoint_offsets = augmented[:, :, -1, :2] - proposals[:, :1, -1, :2]
    grouped_offsets = endpoint_offsets.reshape(2, 16, 4, 2)
    small_axes = torch.tensor([1.0, 2.0])
    large_axes = torch.tensor([3.0, 5.0])

    for group in grouped_offsets.flatten(0, 1):
        valid_assignments = 0
        for small_indices in combinations(range(4), 2):
            small_indices = set(small_indices)
            sectors = []
            valid = True
            for variant_idx, offset in enumerate(group):
                axes = small_axes if variant_idx in small_indices else large_axes
                lateral, longitudinal = offset[1], offset[0]
                normalized = torch.stack((lateral / axes[0], longitudinal / axes[1]))
                radius = normalized.norm()
                if not 0.3 - 1e-5 <= radius <= 1.0 + 1e-5:
                    valid = False
                    break
                angle = torch.atan2(normalized[0], normalized[1]) % (2 * torch.pi)
                sectors.append(int(torch.floor(angle / (torch.pi / 2))))
            if valid and sorted(sectors) == [0, 1, 2, 3]:
                valid_assignments += 1
        assert valid_assignments >= 1

    first_offsets = augmented[:, :, 0, :2] - proposals[:, :1, 0, :2]
    first_weight = 3.0 / 8**2 - 2.0 / 8**3
    assert torch.allclose(first_offsets, endpoint_offsets * first_weight, atol=1e-5)

    previous_xy = torch.cat((torch.zeros(2, 64, 1, 2), augmented[:, :, :-1, :2]), dim=2)
    augmented_steps = augmented[..., :2] - previous_xy
    expected_headings = torch.atan2(augmented_steps[..., 1], augmented_steps[..., 0])
    assert torch.allclose(augmented[..., 2], expected_headings)
    assert torch.isfinite(augmented).all()
    assert torch.all((augmented[..., 2] >= -torch.pi) & (augmented[..., 2] <= torch.pi))


def _routing_model():
    model = DrivoRModel.__new__(DrivoRModel)
    nn.Module.__init__(model)
    model._frozen_backbones = set()
    model._config = SimpleNamespace(
        **vars(_augmentation_config()),
        full_history_status=False,
        ref_num=1,
        noc=1.0,
        dac=1.0,
        ddc=0.0,
        ttc=1.0,
        ep=1.0,
        comfort=1.0,
    )
    model.num_cams = 1
    model.num_lidar = 0
    model.poses_num = 8
    model.state_size = 3
    model.hist_encoding = nn.Linear(11, 24)
    model.init_feature = nn.Embedding(64, 24)
    model.scene_embeds = nn.Parameter(torch.zeros(1, 1, 1, 24))
    model.image_backbone = lambda image, scene_tokens: scene_tokens.reshape(image.shape[0], -1, 24)
    model.traj_head = nn.ModuleList((nn.Linear(24, 24), nn.Linear(24, 24)))
    model.trajectory_decoder = _IdentityTrajectoryDecoder()
    model.pos_embed = nn.Linear(24, 24)
    model.scorer_attention = _CountingScorerAttention()
    model.scorer = _StubScorer()
    model._augment_scorer_proposals = lambda proposals: proposals.detach().clone()
    return model


def test_training_and_inference_scorer_group_routing():
    model = _routing_model()
    features = {
        "ego_status": torch.zeros(1, 2, 11),
        "image": torch.zeros(1, 1),
    }

    model.train()
    training_output = model(features)
    assert [shape[1] for shape in model.scorer_attention.calls] == [64, 64]
    assert training_output["proposals"].shape == (1, 128, 8, 3)
    assert training_output["pdm_score"].shape == (1, 128)
    assert all(stage.shape[1] == 64 for stage in training_output["proposal_list"])

    model.eval()
    model.scorer_attention.calls.clear()
    inference_output = model(features)
    assert [shape[1] for shape in model.scorer_attention.calls] == [64]
    assert inference_output["proposals"].shape == (1, 64, 8, 3)
    assert inference_output["pdm_score"].shape == (1, 64)
