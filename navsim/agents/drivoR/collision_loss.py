import numpy as np
import torch
import torch.nn.functional as F

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


def differentiable_collision_loss(
    proposals: torch.Tensor,
    agent_states: torch.Tensor,
    agent_valid: torch.Tensor,
    margin: float,
    filter_radius: float,
) -> torch.Tensor:
    """Compute a hinge loss from the signed SAT separation of ego and agent OBBs."""
    vehicle = get_pacifica_parameters()
    heading = proposals[..., 2]
    ego_longitudinal = torch.stack((heading.cos(), heading.sin()), dim=-1)
    ego_lateral = torch.stack((-heading.sin(), heading.cos()), dim=-1)
    ego_center = proposals[..., :2] + vehicle.rear_axle_to_center * ego_longitudinal

    agent_heading = agent_states[..., 2]
    agent_longitudinal = torch.stack((agent_heading.cos(), agent_heading.sin()), dim=-1)
    agent_lateral = torch.stack((-agent_heading.sin(), agent_heading.cos()), dim=-1)
    agent_center = agent_states[..., :2]

    ego_center = ego_center.unsqueeze(-2)
    ego_longitudinal = ego_longitudinal.unsqueeze(-2)
    ego_lateral = ego_lateral.unsqueeze(-2)
    agent_center = agent_center.unsqueeze(1)
    agent_longitudinal = agent_longitudinal.unsqueeze(1)
    agent_lateral = agent_lateral.unsqueeze(1)
    relative_center = agent_center - ego_center

    expanded_shape = relative_center.shape
    ego_longitudinal = ego_longitudinal.expand(expanded_shape)
    ego_lateral = ego_lateral.expand(expanded_shape)
    agent_longitudinal = agent_longitudinal.expand(expanded_shape)
    agent_lateral = agent_lateral.expand(expanded_shape)
    axes = torch.stack((ego_longitudinal, ego_lateral, agent_longitudinal, agent_lateral), dim=-2)
    center_projection = (relative_center.unsqueeze(-2) * axes).sum(dim=-1).abs()
    ego_radius = (
        vehicle.half_length * (axes * ego_longitudinal.unsqueeze(-2)).sum(dim=-1).abs()
        + vehicle.half_width * (axes * ego_lateral.unsqueeze(-2)).sum(dim=-1).abs()
    )
    agent_half_length = (agent_states[..., 3] / 2).unsqueeze(1).unsqueeze(-1)
    agent_half_width = (agent_states[..., 4] / 2).unsqueeze(1).unsqueeze(-1)
    agent_radius = (
        agent_half_length * (axes * agent_longitudinal.unsqueeze(-2)).sum(dim=-1).abs()
        + agent_half_width * (axes * agent_lateral.unsqueeze(-2)).sum(dim=-1).abs()
    )
    signed_separation = (center_projection - ego_radius - agent_radius).amax(dim=-1)

    center_distance = torch.linalg.vector_norm(relative_center, dim=-1)
    agent_circle_radius = torch.sqrt(agent_half_length.squeeze(-1).square() + agent_half_width.squeeze(-1).square())
    broad_phase_radius = torch.maximum(
        torch.full_like(agent_circle_radius, filter_radius),
        torch.full_like(
            agent_circle_radius,
            np.hypot(vehicle.half_length, vehicle.half_width),
        ) + agent_circle_radius + margin,
    )
    pair_valid = agent_valid.unsqueeze(1) & (center_distance <= broad_phase_radius)
    per_agent_loss = F.relu(margin - signed_separation).masked_fill(~pair_valid, 0.0)
    per_pose_loss = per_agent_loss.amax(dim=-1)
    return per_pose_loss.sum(dim=-1).mean()
