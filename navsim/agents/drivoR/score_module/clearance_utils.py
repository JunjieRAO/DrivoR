from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon


def temporal_sampling_indices(
    simulation_num_poses: int, model_num_poses: int
) -> npt.NDArray[np.int64]:
    if simulation_num_poses % model_num_poses != 0:
        raise ValueError(
            f"Cannot align {model_num_poses} model poses with "
            f"{simulation_num_poses} simulation poses."
        )
    sampling_stride = simulation_num_poses // model_num_poses
    return np.arange(
        sampling_stride, simulation_num_poses + 1, sampling_stride, dtype=np.int64
    )


def _polygon_axes(vertices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    edges = np.roll(vertices, -1, axis=0) - vertices
    axes = np.stack((-edges[:, 1], edges[:, 0]), axis=-1)
    norms = np.linalg.norm(axes, axis=-1)
    return axes[norms > 0] / norms[norms > 0, None]


def _penetration_depth(first: Polygon, second: Polygon) -> float:
    first_vertices = np.asarray(first.convex_hull.exterior.coords[:-1], dtype=np.float64)
    second_vertices = np.asarray(second.convex_hull.exterior.coords[:-1], dtype=np.float64)
    axes = np.concatenate((_polygon_axes(first_vertices), _polygon_axes(second_vertices)))

    minimum_overlap = np.inf
    for axis in axes:
        first_projection = first_vertices @ axis
        second_projection = second_vertices @ axis
        overlap = min(
            first_projection.max() - second_projection.min(),
            second_projection.max() - first_projection.min(),
        )
        minimum_overlap = min(minimum_overlap, overlap)

    return max(float(minimum_overlap), 0.0)


def signed_polygon_clearance(first: Polygon, second: Polygon) -> float:
    """Returns positive separation or negative convex-polygon penetration depth."""
    if first.is_empty or second.is_empty:
        raise ValueError("Clearance cannot be computed for an empty polygon.")

    distance = first.distance(second)
    if distance > 0:
        return float(distance)
    if first.touches(second):
        return 0.0
    return -_penetration_depth(first, second)


def minimum_signed_clearance(
    ego_polygon: Polygon,
    object_polygons: Iterable[Polygon],
    maximum_clearance: float,
) -> float:
    clearances = [
        signed_polygon_clearance(ego_polygon, object_polygon)
        for object_polygon in object_polygons
    ]
    return min(clearances, default=maximum_clearance)


def compute_temporal_clearance_targets(
    ego_polygons: npt.NDArray[np.object_],
    observation,
    clip_min: float,
    clip_max: float,
    observation_indices: Sequence[int],
) -> npt.NDArray[np.float32]:
    """Computes minimum clearance for proposal polygons shaped (N, T)."""
    if ego_polygons.ndim != 2:
        raise ValueError(f"Expected ego polygons with shape (N, T), got {ego_polygons.shape}.")
    if clip_min >= clip_max:
        raise ValueError("clearance clip_min must be smaller than clip_max.")

    num_proposals, num_poses = ego_polygons.shape
    if len(observation_indices) != num_poses:
        raise ValueError(
            f"Expected {num_poses} observation indices, got {len(observation_indices)}."
        )
    targets = np.full((num_proposals, num_poses), clip_max, dtype=np.float32)
    red_light_prefix = observation.red_light_token

    for time_idx, observation_idx in enumerate(observation_indices):
        occupancy_map = observation[observation_idx]
        object_polygons = [
            occupancy_map[token]
            for token in occupancy_map.tokens
            if not token.startswith(red_light_prefix)
        ]
        if not object_polygons:
            continue

        for proposal_idx in range(num_proposals):
            targets[proposal_idx, time_idx] = minimum_signed_clearance(
                ego_polygons[proposal_idx, time_idx], object_polygons, clip_max
            )

    return np.clip(targets, clip_min, clip_max)