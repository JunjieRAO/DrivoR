from typing import Dict, Sequence

import torch
from torch import Tensor


def select_topk_proposal_indices(
    no_collision_scores: Tensor,
    drivable_area_scores: Tensor,
    pdm_scores: Tensor,
    topk_values: Sequence[int],
) -> Dict[int, Tensor]:
    """Select the highest-PDMS proposal from each NC/DAC top-k union."""
    expected_shape = tuple(pdm_scores.shape)
    if pdm_scores.ndim != 2:
        raise ValueError(f"Expected pdm_scores shape [B, N], got {expected_shape}")
    if tuple(no_collision_scores.shape) != expected_shape:
        raise ValueError(
            f"Expected no_collision_scores shape {expected_shape}, "
            f"got {tuple(no_collision_scores.shape)}"
        )
    if tuple(drivable_area_scores.shape) != expected_shape:
        raise ValueError(
            f"Expected drivable_area_scores shape {expected_shape}, "
            f"got {tuple(drivable_area_scores.shape)}"
        )
    if pdm_scores.shape[1] == 0:
        raise ValueError("Cannot select from zero proposals")

    normalized_topk_values = tuple(int(k) for k in topk_values)
    if any(k <= 0 for k in normalized_topk_values):
        raise ValueError(f"Expected positive top-k values, got {normalized_topk_values}")

    nc_order = torch.argsort(no_collision_scores, dim=1, descending=True, stable=True)
    dac_order = torch.argsort(drivable_area_scores, dim=1, descending=True, stable=True)
    score_order = torch.argsort(pdm_scores, dim=1, descending=True, stable=True)
    batch_indices = torch.arange(pdm_scores.shape[0], device=pdm_scores.device)

    selected_indices: Dict[int, Tensor] = {}
    for k in normalized_topk_values:
        effective_k = min(k, pdm_scores.shape[1])
        candidate_mask = torch.zeros_like(pdm_scores, dtype=torch.bool)
        candidate_mask.scatter_(1, nc_order[:, :effective_k], True)
        candidate_mask.scatter_(1, dac_order[:, :effective_k], True)
        ranked_candidates = candidate_mask.gather(1, score_order)
        first_candidate_rank = ranked_candidates.to(torch.int64).argmax(dim=1)
        selected_indices[k] = score_order[batch_indices, first_candidate_rank]

    return selected_indices