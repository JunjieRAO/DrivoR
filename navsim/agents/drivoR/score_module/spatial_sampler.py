from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from navsim.agents.drivoR.layers.utils.mlp import MLP

def project_poses_to_camera(
    proposals: torch.Tensor,
    cam_K: torch.Tensor,
    world_2_cam: torch.Tensor,
    image_size: Tuple[int, int] = (1148, 672),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D proposal poses (x, y, theta) into 2D camera image coordinates.

    Args:
        proposals: (B, N, T, 3) tensor of (x, y, theta) in vehicle/ego frame (Z=0).
        cam_K: (B, C, 3, 3) camera intrinsic matrices.
        world_2_cam: (B, C, 4, 4) transformation from ego/lidar to camera 3D frame.
        image_size: (W, H) tuple of image dimensions.

    Returns:
        uv_norm: (B, N, T, C, 2) normalized pixel coordinates in [0, 1].
        depth: (B, N, T, C) Z_c camera depth.
        valid_mask: (B, N, T, C) boolean mask indicating valid projection in front of camera.
    """
    B, N, T, _ = proposals.shape
    C = cam_K.shape[1]
    W_img, H_img = image_size

    # Ensure cam_K and world_2_cam match proposals device and dtype (e.g. float16/float32 instead of double)
    cam_K = cam_K.to(device=proposals.device, dtype=proposals.dtype)
    world_2_cam = world_2_cam.to(device=proposals.device, dtype=proposals.dtype)

    # 1. Expand proposals to homogeneous 3D coordinates (x, y, 0, 1)
    pts_ego = torch.zeros((B, N, T, 4), device=proposals.device, dtype=proposals.dtype)
    pts_ego[..., 0] = proposals[..., 0]  # X
    pts_ego[..., 1] = proposals[..., 1]  # Y
    pts_ego[..., 2] = 0.0                # Z = 0
    pts_ego[..., 3] = 1.0                # Homogeneous coordinate

    # 2. Transform ego -> camera 3D coords: P_cam = P_ego @ world_2_cam^T
    # pts_ego: (B, 1, N, T, 4), world_2_cam: (B, C, 1, 1, 4, 4) -> (B, C, N, T, 4)
    pts_ego_exp = pts_ego.unsqueeze(1)  # (B, 1, N, T, 4)
    world_2_cam_exp = world_2_cam.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1, 4, 4)

    # Matrix multiplication along 4D vector
    pts_cam = torch.matmul(world_2_cam_exp, pts_ego_exp.unsqueeze(-1)).squeeze(-1)  # (B, C, N, T, 4)
    # Permute to (B, N, T, C, 4)
    pts_cam = pts_cam.permute(0, 2, 3, 1, 4)  # (B, N, T, C, 4)

    X_c = pts_cam[..., 0]
    Y_c = pts_cam[..., 1]
    Z_c = pts_cam[..., 2]

    # 3. Project 3D camera coords -> 2D image plane via intrinsics K
    # cam_K: (B, C, 3, 3) -> reshape to (B, 1, 1, C, 3, 3)
    cam_K_exp = cam_K.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, C, 3, 3)
    pts_cam_3d = torch.stack([X_c, Y_c, Z_c], dim=-1)  # (B, N, T, C, 3)

    pts_img_3d = torch.matmul(cam_K_exp, pts_cam_3d.unsqueeze(-1)).squeeze(-1)  # (B, N, T, C, 3)

    x_px = pts_img_3d[..., 0]
    y_px = pts_img_3d[..., 1]
    z_px = pts_img_3d[..., 2]

    z_clamp = torch.clamp(z_px, min=1e-4)
    u_px = x_px / z_clamp
    v_px = y_px / z_clamp

    u_norm = u_px / float(W_img)
    v_norm = v_px / float(H_img)

    uv_norm = torch.stack([u_norm, v_norm], dim=-1)  # (B, N, T, C, 2)
    depth = Z_c

    valid_mask = (Z_c > 0.1) & (u_norm >= -0.2) & (u_norm <= 1.2) & (v_norm >= -0.2) & (v_norm <= 1.2)

    return uv_norm, depth, valid_mask


class LocalPatchSampler(nn.Module):
    """
    Gaussian-weighted spatial cross-attention over multi-camera feature patches.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_poses: int = 8,
        sigma: float = 0.15,
        image_size: Tuple[int, int] = (1148, 672),
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_poses = num_poses
        self.sigma = sigma
        self.image_size = image_size
        self.head_dim = d_model // num_heads

        # Waypoint pose & step encodings
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.step_embedder = nn.Embedding(num_poses, d_model)

        # Multi-head Cross Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Step aggregation layer (8 waypoints -> 1 token)
        self.step_aggregate = nn.Linear(num_poses * d_model, d_model)

    def forward(
        self,
        proposal_tokens: torch.Tensor,
        proposals: torch.Tensor,
        patch_features: torch.Tensor,
        cam_K: torch.Tensor,
        world_2_cam: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            proposal_tokens: (B, N, d_model) baseline query tokens.
            proposals: (B, N, T, 3) trajectory poses (x, y, theta).
            patch_features: (B, C, H_grid, W_grid, d_model) image patch tokens.
            cam_K: (B, C, 3, 3) camera intrinsics.
            world_2_cam: (B, C, 4, 4) camera extrinsics.

        Returns:
            H_local: (B, N, d_model) local spatial patch features.
        """
        B, N, d_model = proposal_tokens.shape
        _, _, T, _ = proposals.shape
        _, C, H_grid, W_grid, _ = patch_features.shape

        # 1. Project proposal waypoints to camera 2D coordinates
        # uv_norm: (B, N, T, C, 2), depth: (B, N, T, C), valid_mask: (B, N, T, C)
        uv_norm, depth, valid_mask = project_poses_to_camera(
            proposals, cam_K, world_2_cam, image_size=self.image_size
        )
        uv_norm = uv_norm.to(dtype=patch_features.dtype)

        # 2. Compute Patch Grid Centers (H_grid, W_grid, 2)
        device = patch_features.device
        dtype = patch_features.dtype

        grid_y = (torch.arange(H_grid, device=device, dtype=dtype) + 0.5) / float(H_grid)
        grid_x = (torch.arange(W_grid, device=device, dtype=dtype) + 0.5) / float(W_grid)
        grid_vy, grid_ux = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid_centers = torch.stack([grid_ux, grid_vy], dim=-1)  # (H_grid, W_grid, 2)
        grid_centers_flat = grid_centers.reshape(H_grid * W_grid, 2)  # (K_patches, 2)

        # 3. Compute spatial Gaussian distance bias between waypoints (B, N, T, C, 2) and patches (K_patches, 2)
        # uv_norm: (B, N, T, C, 1, 2), grid_centers_flat: (1, 1, 1, 1, K_patches, 2)
        u_diff = uv_norm[..., 0:1] - grid_centers_flat[:, 0]  # (B, N, T, C, K_patches)
        v_diff = uv_norm[..., 1:2] - grid_centers_flat[:, 1]  # (B, N, T, C, K_patches)

        spatial_dist_sq = (u_diff ** 2 + v_diff ** 2) / (2.0 * (self.sigma ** 2))  # (B, N, T, C, K_patches)
        spatial_bias = -spatial_dist_sq  # Higher (closer to 0) for nearby patches

        # Mask out invalid/behind-camera waypoints by making bias very negative
        spatial_bias = torch.where(
            valid_mask.unsqueeze(-1),
            spatial_bias,
            torch.full_like(spatial_bias, -1e4)
        )

        # Reshape spatial_bias across all cameras and patches: (B, N, T, C * K_patches)
        spatial_bias_all = spatial_bias.reshape(B, N, T, C * H_grid * W_grid)

        # 4. Construct spatio-temporal Query (B, N, T, d_model)
        step_idx = torch.arange(T, device=device)
        step_embed = self.step_embedder(step_idx)  # (T, d_model)

        pose_embed = self.pose_encoder(proposals)  # (B, N, T, d_model)
        query = proposal_tokens.unsqueeze(2) + pose_embed + step_embed.unsqueeze(0).unsqueeze(0)  # (B, N, T, d_model)

        # 5. Prepare Key & Value from patch features
        # patch_features: (B, C, H_grid, W_grid, d_model) -> (B, C * H_grid * W_grid, d_model)
        kv_flat = patch_features.reshape(B, C * H_grid * W_grid, d_model)

        # Multi-Head Projections
        # Q: (B, N, T, h, d_k) -> permute to (B, h, N*T, d_k)
        Q = self.q_proj(query).reshape(B, N * T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # K: (B, K_all, h, d_k) -> permute to (B, h, K_all, d_k)
        K = self.k_proj(kv_flat).reshape(B, C * H_grid * W_grid, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # V: (B, K_all, h, d_k) -> permute to (B, h, K_all, d_k)
        V = self.v_proj(kv_flat).reshape(B, C * H_grid * W_grid, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention + Spatial Bias
        # attn_scores: (B, h, N*T, K_all)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Reshape spatial_bias_all to match attn_scores: (B, 1, N*T, K_all)
        bias_exp = spatial_bias_all.reshape(B, 1, N * T, C * H_grid * W_grid)
        attn_scores = attn_scores + bias_exp

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, h, N*T, K_all)
        out = torch.matmul(attn_weights, V)  # (B, h, N*T, d_k)

        # Reshape back: (B, N, T, d_model)
        out = out.permute(0, 2, 1, 3).reshape(B, N, T, d_model)
        out = self.out_proj(out)

        # 6. Aggregate step tokens across time steps (B, N, T*d_model) -> (B, N, d_model)
        h_local = self.step_aggregate(out.reshape(B, N, T * d_model))

        return h_local
