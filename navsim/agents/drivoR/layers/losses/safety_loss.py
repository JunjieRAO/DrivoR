import torch
import torch.nn.functional as F


def interpolate_poses(proposals, steps):
    initial = torch.zeros_like(proposals[:, :, :1])
    poses = torch.cat((initial, proposals), dim=2)
    positions = F.interpolate(
        poses[..., :2].flatten(0, 1).transpose(1, 2),
        size=steps + 1, mode="linear", align_corners=True,
    ).transpose(1, 2).reshape(*proposals.shape[:2], steps + 1, 2)
    angles = poses[..., 2]
    differences = angles[..., 1:] - angles[..., :-1]
    differences = torch.atan2(differences.sin(), differences.cos())
    unwrapped = torch.cat((angles[..., :1], differences.cumsum(-1)), dim=-1)
    headings = F.interpolate(
        unwrapped.flatten(0, 1).unsqueeze(1),
        size=steps + 1, mode="linear", align_corners=True,
    ).reshape(*proposals.shape[:2], steps + 1)
    return torch.cat((positions, headings.unsqueeze(-1)), dim=-1)[:, :, 1:]


def footprint_points(poses, vehicle, samples=True):
    longitudinal = torch.linspace(-0.5, 0.5, 5 if samples else 2, device=poses.device)
    lateral = torch.linspace(-0.5, 0.5, 3 if samples else 2, device=poses.device)
    longitudinal, lateral = torch.meshgrid(longitudinal, lateral, indexing="ij")
    local = torch.stack((longitudinal.flatten(), lateral.flatten()), dim=-1)
    local = local[None, None, None] * vehicle[:, None, None, None, :2]
    local_x = local[..., 0] + vehicle[:, None, None, None, 2]
    local_y = local[..., 1]
    cosine = poses[..., 2].cos().unsqueeze(-1)
    sine = poses[..., 2].sin().unsqueeze(-1)
    return torch.stack((
        poses[..., 0, None] + cosine * local_x - sine * local_y,
        poses[..., 1, None] + sine * local_x + cosine * local_y,
    ), dim=-1)


def safety_losses(proposals, targets, dac_margin=0.2, collision_margin=0.2,
                  worst_weight=1.0, compute_dac=True, compute_collision=True):
    proposals = proposals.float()
    steps = targets["obstacle_valid"].shape[1]
    poses = interpolate_poses(proposals, steps)
    vehicle = targets["vehicle"].to(proposals)
    zero = proposals.sum() * 0.0
    dac_loss = collision_loss = zero
    if compute_dac:
        points = footprint_points(poses, vehicle)
        bounds = targets["sdf_bounds"].to(proposals)[:, None, None, None]
        lower, upper = bounds[..., :2], bounds[..., 2:]
        clamped = torch.maximum(lower, torch.minimum(points, upper))
        grid = 2.0 * (clamped - lower) / (upper - lower) - 1.0
        batch_size, candidates, time_steps, point_count, _ = points.shape
        distances = F.grid_sample(
            targets["drivable_sdf"].to(proposals).unsqueeze(1),
            grid.reshape(batch_size, candidates * time_steps, point_count, 2),
            mode="bilinear", padding_mode="border", align_corners=True,
        ).reshape(batch_size, candidates, time_steps, point_count)
        outside = torch.linalg.vector_norm(points - clamped, dim=-1)
        penalty = F.relu(dac_margin - distances).square() + outside.square()
        trajectory_loss = penalty.mean((-1, -2)) + worst_weight * penalty.amax((-1, -2))
        dac_loss = trajectory_loss.mean()
    if compute_collision:
        ego = footprint_points(poses, vehicle, samples=False)
        heading = poses[..., 2]
        ego_axes = torch.stack((
            torch.stack((heading.cos(), heading.sin()), dim=-1),
            torch.stack((-heading.sin(), heading.cos()), dim=-1),
        ), dim=-2)
        obstacles = targets["obstacle_corners"].to(proposals)
        valid = targets["obstacle_valid"].to(device=proposals.device)
        total = torch.zeros_like(heading)
        worst = torch.zeros_like(heading)
        counts = valid.sum(-1)[:, None].expand_as(heading).clamp(min=1)
        for start in range(0, obstacles.shape[2], 16):
            boxes = obstacles[:, :, start:start + 16]
            mask = valid[:, :, start:start + 16]
            edges = torch.stack((boxes[..., 1, :] - boxes[..., 0, :],
                                 boxes[..., 3, :] - boxes[..., 0, :]), dim=-2)
            actor_axes = F.normalize(edges, dim=-1, eps=1e-6)
            axes = torch.cat((
                ego_axes.unsqueeze(3).expand(-1, -1, -1, boxes.shape[2], -1, -1),
                actor_axes[:, None].expand(-1, proposals.shape[1], -1, -1, -1, -1),
            ), dim=-2)
            ego_projection = torch.einsum("bntpd,bntakd->bntakp", ego, axes)
            actor_projection = torch.einsum("btapd,bntakd->bntakp", boxes, axes)
            gap = torch.maximum(
                actor_projection.amin(-1) - ego_projection.amax(-1),
                ego_projection.amin(-1) - actor_projection.amax(-1),
            ).amax(-1)
            penalty = F.relu(collision_margin - gap).square() * mask[:, None]
            total = total + penalty.sum(-1)
            worst = torch.maximum(worst, penalty.amax(-1))
        trajectory_loss = (total / counts).mean(-1) + worst_weight * worst.amax(-1)
        collision_loss = trajectory_loss.mean()
    return dac_loss, collision_loss