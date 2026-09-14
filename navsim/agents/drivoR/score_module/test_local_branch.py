import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from navsim.agents.drivoR.score_module.spatial_sampler import (
    project_poses_to_camera,
    LocalPatchSampler,
)
from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer
from navsim.agents.drivoR.drivor_model import DrivoRModel


def test_spatial_projection_shape_and_bounds():
    B, N, T, C = 2, 64, 8, 4
    proposals = torch.randn(B, N, T, 3)
    # Forward-looking waypoints (x > 0)
    proposals[..., 0] = torch.abs(proposals[..., 0]) + 1.0

    cam_K = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, C, 1, 1)
    cam_K[:, :, 0, 0] = 500.0  # fx
    cam_K[:, :, 1, 1] = 500.0  # fy
    cam_K[:, :, 0, 2] = 574.0  # cx
    cam_K[:, :, 1, 2] = 336.0  # cy

    world_2_cam = torch.eye(4).reshape(1, 1, 4, 4).repeat(B, C, 1, 1)

    uv_norm, depth, valid_mask = project_poses_to_camera(
        proposals, cam_K, world_2_cam, image_size=(1148, 672)
    )

    assert uv_norm.shape == (B, N, T, C, 2)
    assert depth.shape == (B, N, T, C)
    assert valid_mask.shape == (B, N, T, C)


def test_zero_init_equivalence():
    """Verify H_fused == H_global at initialization step."""
    B, N, d_model = 2, 64, 256
    C, H_grid, W_grid = 4, 48, 82

    config = OmegaConf.create({
        "refiner_num_heads": 4,
        "refiner_ls_values": 0.0,
        "use_local_patch_branch": True,
        "local_fusion_alpha": 1.0,
        "spatial_patch_num_heads": 4,
        "spatial_patch_sigma": 0.15,
        "num_poses": 8,
        "image_size": [1148, 672],
    })

    scorer_decoder = TransformerDecoderScorer(
        num_layers=2, d_model=d_model, proj_drop=0.1, drop_path=0.0, config=config
    )
    scorer_decoder.eval()

    x = torch.randn(B, N, d_model)
    x_cross = torch.randn(B, C * 16, d_model)
    proposals = torch.randn(B, N, 8, 3)
    patch_features = torch.randn(B, C, H_grid, W_grid, d_model)
    cam_K = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, C, 1, 1)
    world_2_cam = torch.eye(4).reshape(1, 1, 4, 4).repeat(B, C, 1, 1)

    # 1. Output without local patch branch inputs (H_global)
    h_global = scorer_decoder(x, x_cross)

    # 2. Output with local patch branch inputs (H_fused)
    h_fused = scorer_decoder(
        x, x_cross, proposals=proposals, patch_features=patch_features, cam_K=cam_K, world_2_cam=world_2_cam
    )

    # 3. Check exact numerical equivalence (atol=1e-6)
    assert torch.allclose(h_fused, h_global, atol=1e-6), "H_fused should equal H_global at initialization step!"


def test_gradient_flow_local_branch():
    """Verify gradients propagate back to local fusion projection weights and patch features."""
    B, N, d_model = 2, 16, 256
    C, H_grid, W_grid = 2, 12, 12

    config = OmegaConf.create({
        "refiner_num_heads": 2,
        "refiner_ls_values": 0.0,
        "use_local_patch_branch": True,
        "local_fusion_alpha": 1.0,
        "spatial_patch_num_heads": 2,
        "spatial_patch_sigma": 0.15,
        "num_poses": 8,
        "image_size": [1148, 672],
    })

    scorer_decoder = TransformerDecoderScorer(
        num_layers=1, d_model=d_model, proj_drop=0.0, drop_path=0.0, config=config
    )

    x = torch.randn(B, N, d_model, requires_grad=True)
    x_cross = torch.randn(B, C * 16, d_model, requires_grad=True)
    proposals = torch.randn(B, N, 8, 3)
    patch_features = torch.randn(B, C, H_grid, W_grid, d_model, requires_grad=True)
    cam_K = torch.eye(3).reshape(1, 1, 3, 3).repeat(B, C, 1, 1)
    world_2_cam = torch.eye(4).reshape(1, 1, 4, 4).repeat(B, C, 1, 1)

    h_fused = scorer_decoder(
        x, x_cross, proposals=proposals, patch_features=patch_features, cam_K=cam_K, world_2_cam=world_2_cam
    )

    loss = h_fused.sum()
    loss.backward()

    assert scorer_decoder.local_fusion_proj.weight.grad is not None
    assert patch_features.grad is not None
