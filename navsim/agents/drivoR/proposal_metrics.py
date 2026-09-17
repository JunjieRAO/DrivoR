from typing import Dict

import torch


def wta_loss_schedule(
    current_epoch: int,
    start_epoch: int = 5,
    end_epoch: int = 8,
    minimum_weight: float = 0.2,
) -> float:
    """Linearly decay WTA weight from 1 before start_epoch to minimum_weight at end_epoch."""
    if current_epoch < start_epoch:
        return 1.0
    if current_epoch >= end_epoch:
        return minimum_weight
    progress = (current_epoch - start_epoch + 1) / (end_epoch - start_epoch + 1)
    return 1.0 - progress * (1.0 - minimum_weight)


@torch.no_grad()
def wta_imitation_weights(
    wta_indices: torch.Tensor,
    proposal_pdms: torch.Tensor,
    proposal_scores: torch.Tensor,
    gt_pdms: torch.Tensor,
    scheduled_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-scene weights and the mask for safe WTA proposals that outperform GT."""
    wta_pdms = proposal_pdms.gather(1, wta_indices[:, None]).squeeze(1)
    wta_scores = proposal_scores.gather(
        1, wta_indices[:, None, None].expand(-1, 1, proposal_scores.shape[-1])
    ).squeeze(1)
    safe = (
        torch.isclose(wta_scores[:, 0], torch.ones_like(wta_scores[:, 0]))
        & torch.isclose(wta_scores[:, 1], torch.ones_like(wta_scores[:, 1]))
        & torch.isclose(wta_scores[:, 3], torch.ones_like(wta_scores[:, 3]))
    )
    eligible = safe & (wta_pdms > gt_pdms)
    weights = torch.where(
        eligible,
        torch.full_like(wta_pdms, scheduled_weight),
        torch.ones_like(wta_pdms),
    )
    return weights, eligible


SAFETY_METRIC_NAMES = (
    "proposal_nc_zero_ratio",
    "proposal_dac_zero_ratio",
    "scene_any_nc_zero_ratio",
    "scene_any_dac_zero_ratio",
    "proposal_ttc_lt1_ratio",
    "proposal_safe_ratio",
    "scene_no_safe_proposal_ratio",
    "safe_proposal_ep_mean",
    "avoidable_unsafe_selection_ratio",
    "wta_pdms_better_than_gt_ratio",
    "wta_better_than_gt_min_loss_ratio",
)


@torch.no_grad()
def proposal_safety_statistics(
    target_scores: torch.Tensor, selected_indices: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Return numerator/denominator pairs for proposal-level safety metrics."""
    nc, dac, ep, ttc = (target_scores[..., index] for index in range(4))
    valid = ~torch.isclose(ttc, torch.full_like(ttc, 2.0))
    valid_scene = valid.any(dim=1)
    nc_zero = valid & torch.isclose(nc, torch.zeros_like(nc))
    dac_zero = valid & torch.isclose(dac, torch.zeros_like(dac))
    safe = (
        valid
        & torch.isclose(nc, torch.ones_like(nc))
        & torch.isclose(dac, torch.ones_like(dac))
        & torch.isclose(ttc, torch.ones_like(ttc))
    )
    scene_has_safe = safe.any(dim=1)
    selected_safe = safe.gather(1, selected_indices[:, None]).squeeze(1)

    valid_count = valid.sum().to(torch.float64)
    valid_scene_count = valid_scene.sum().to(torch.float64)
    safe_count = safe.sum().to(torch.float64)
    safe_scene_count = scene_has_safe.sum().to(torch.float64)

    def pair(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.stack((numerator.to(torch.float64), denominator.to(torch.float64)))

    return {
        "proposal_nc_zero_ratio": pair(nc_zero.sum(), valid_count),
        "proposal_dac_zero_ratio": pair(dac_zero.sum(), valid_count),
        "scene_any_nc_zero_ratio": pair((nc_zero.any(dim=1) & valid_scene).sum(), valid_scene_count),
        "scene_any_dac_zero_ratio": pair((dac_zero.any(dim=1) & valid_scene).sum(), valid_scene_count),
        "proposal_ttc_lt1_ratio": pair((valid & (ttc < 1.0)).sum(), valid_count),
        "proposal_safe_ratio": pair(safe_count, valid_count),
        "scene_no_safe_proposal_ratio": pair((valid_scene & ~scene_has_safe).sum(), valid_scene_count),
        "safe_proposal_ep_mean": pair(ep.masked_select(safe).sum(), safe_count),
        "avoidable_unsafe_selection_ratio": pair((scene_has_safe & ~selected_safe).sum(), safe_scene_count),
    }


@torch.no_grad()
def wta_gt_statistics(
    proposals: torch.Tensor,
    target_trajectory: torch.Tensor,
    proposal_pdms: torch.Tensor,
    gt_pdms: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Measure how often the imitation WTA beats GT and its share of WTA loss."""
    per_proposal_loss = torch.linalg.vector_norm(
        proposals - target_trajectory[:, None], ord=1, dim=-1
    ).mean(dim=-1)
    wta_loss, wta_indices = per_proposal_loss.min(dim=1)
    wta_pdms = proposal_pdms.gather(1, wta_indices[:, None]).squeeze(1)
    better_than_gt = wta_pdms > gt_pdms
    scene_count = torch.tensor(float(proposals.shape[0]), device=proposals.device, dtype=torch.float64)

    return {
        "wta_pdms_better_than_gt_ratio": torch.stack(
            (better_than_gt.sum().to(torch.float64), scene_count)
        ),
        "wta_better_than_gt_min_loss_ratio": torch.stack(
            (
                wta_loss.masked_select(better_than_gt).sum().to(torch.float64),
                wta_loss.sum().to(torch.float64),
            )
        ),
    }
