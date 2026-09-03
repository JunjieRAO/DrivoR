import numpy as np
import pytest
from shapely.affinity import rotate
from shapely.geometry import box

from navsim.agents.drivoR.score_module.clearance_utils import (
    minimum_signed_clearance,
    signed_polygon_clearance,
    temporal_sampling_indices,
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