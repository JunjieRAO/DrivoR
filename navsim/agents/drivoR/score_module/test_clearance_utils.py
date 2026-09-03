import numpy as np
import pytest
from shapely.affinity import rotate
from shapely.geometry import box

from navsim.agents.drivoR.score_module.clearance_utils import (
    compute_temporal_clearance_targets,
    minimum_signed_clearance,
    signed_polygon_clearance,
    temporal_sampling_indices,
)
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)


def test_signed_polygon_clearance_for_separation_touch_and_overlap() -> None:
    ego = box(0.0, 0.0, 2.0, 2.0)

    assert signed_polygon_clearance(ego, box(5.0, 0.0, 7.0, 2.0)) == pytest.approx(3.0)
    assert signed_polygon_clearance(ego, box(2.0, 0.0, 4.0, 2.0)) == pytest.approx(0.0)
    assert signed_polygon_clearance(ego, box(1.5, 0.0, 3.5, 2.0)) == pytest.approx(-0.5)


def test_signed_polygon_clearance_for_rotated_overlap() -> None:
    ego = box(-1.0, -1.0, 1.0, 1.0)
    other = rotate(box(0.5, -0.5, 2.5, 0.5), 30.0, origin="centroid")

    clearance = signed_polygon_clearance(ego, other)

    assert np.isfinite(clearance)
    assert clearance < 0.0


def test_signed_polygon_clearance_for_contained_polygon() -> None:
    outer = box(-2.0, -2.0, 2.0, 2.0)
    inner = box(-0.5, -0.5, 0.5, 0.5)

    assert signed_polygon_clearance(outer, inner) == pytest.approx(-2.5)


def test_minimum_signed_clearance_and_empty_objects() -> None:
    ego = box(0.0, 0.0, 2.0, 2.0)
    objects = [box(5.0, 0.0, 7.0, 2.0), box(2.5, 0.0, 4.5, 2.0)]

    assert minimum_signed_clearance(ego, objects, 5.0) == pytest.approx(0.5)
    assert minimum_signed_clearance(ego, [], 5.0) == pytest.approx(5.0)


def test_temporal_sampling_indices_align_model_and_simulation() -> None:
    assert temporal_sampling_indices(40, 8).tolist() == [5, 10, 15, 20, 25, 30, 35, 40]

    with pytest.raises(ValueError, match="Cannot align"):
        temporal_sampling_indices(40, 6)


def test_temporal_targets_query_nearby_objects_and_exclude_red_lights() -> None:
    class Observation:
        red_light_token = "red_light"

        def __init__(self):
            self.maps = {
                5: PDMOccupancyMap(
                    ["near", "far", "red_light_lane"],
                    np.asarray(
                        [
                            box(2.5, 0.0, 3.5, 1.0),
                            box(20.0, 0.0, 21.0, 1.0),
                            box(0.25, 0.25, 0.75, 0.75),
                        ],
                        dtype=object,
                    ),
                )
            }

        def __getitem__(self, index):
            return self.maps[index]

    ego_polygons = np.asarray(
        [[box(0.0, 0.0, 1.0, 1.0)], [box(2.75, 0.0, 3.75, 1.0)]], dtype=object
    )

    targets = compute_temporal_clearance_targets(
        ego_polygons, Observation(), -1.0, 5.0, [5]
    )

    assert targets.shape == (2, 1)
    assert targets[0, 0] == pytest.approx(1.5)
    assert targets[1, 0] < 0.0