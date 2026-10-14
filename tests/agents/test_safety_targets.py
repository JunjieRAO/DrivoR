import unittest
from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon, box
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.agents.drivoR.score_module.safety_targets import build_safety_targets


class Occupancy:
    tokens = ["actor", "red_light_lane", "past_collision"]

    def __getitem__(self, token):
        return box(10, 19, 12, 21)


class Observation:
    _sample_interval = 0.1
    collided_track_ids = ["past_collision"]
    red_light_token = "red_light"

    def __getitem__(self, index):
        return Occupancy()


class DrivableMap:
    tokens = ["road"]

    def get_indices_of_map_type(self, layers):
        if SemanticMapLayer.ROADBLOCK not in layers:
            raise AssertionError("Missing original DAC layers")
        return [0]

    def __getitem__(self, token):
        return box(5, 15, 15, 25)


class SafetyTargetsTest(unittest.TestCase):
    def cache(self):
        return SimpleNamespace(
            ego_state=SimpleNamespace(
                rear_axle=SimpleNamespace(x=10., y=20., heading=np.pi / 2),
                car_footprint=SimpleNamespace(vehicle_parameters=SimpleNamespace(
                    length=4., width=2., rear_axle_to_center=1.,
                )),
            ),
            drivable_area_map=DrivableMap(), observation=Observation(),
        )

    def test_local_frame_layers_and_exclusions(self):
        targets = build_safety_targets(self.cache(), (-8, -8, 8, 8), 0.5, 0.2)
        self.assertGreater(targets["drivable_sdf"][16, 16], 0)
        self.assertLess(targets["drivable_sdf"][0, 0], 0)
        np.testing.assert_allclose(targets["obstacle_corners"][0, 0].mean(0), [0, -1], atol=1e-6)
        self.assertEqual(targets["obstacle_valid"].sum(), 2)
        np.testing.assert_allclose(targets["vehicle"], [4, 2, 1])

    def test_invalid_map_polygon(self):
        cache = self.cache()
        class InvalidMap(DrivableMap):
            def __getitem__(self, token):
                return Polygon([(5, 15), (15, 25), (5, 25), (15, 15)])
        cache.drivable_area_map = InvalidMap()
        targets = build_safety_targets(cache, (-8, -8, 8, 8), 0.5, 0.2)
        self.assertTrue(np.isfinite(targets["drivable_sdf"]).all())


if __name__ == "__main__":
    unittest.main()