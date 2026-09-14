"""Run in the existing Linux DrivoR environment; no real dataset required.

Windows cannot import the vendored nuPlan map stack (fcntl). Skipping there is
explicit, not evidence that the formal adapter passed its integration test.
"""
import sys
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="vendored nuPlan requires Linux fcntl")


def test_official_pair_scoring_recording_and_replay():
    from shapely.geometry import Polygon
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, pdm_score
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap, PDMOccupancyMap
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.offline_search.official import OfficialEvaluator, local_to_world, world_to_local
    from navsim.offline_search.geometry import vertices
    from navsim.offline_search.core import Config
    origin = np.array([1000., 2000., .2])
    vehicle = get_pacifica_parameters()
    ego = EgoState.build_from_rear_axle(StateSE2(*origin), StateVector2D(6., 0.),
        StateVector2D(0., 0.), 0., TimePoint(1_000_000), vehicle)
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    gt_local = np.c_[np.arange(1, 9) * 3., np.zeros((8, 2))]
    gt = Trajectory(gt_local)
    path = local_to_world(np.array([[0.,0.,0.],[50.,0.,0.]]), origin)
    assert np.allclose(world_to_local(local_to_world(gt_local, origin), origin), gt_local)
    centerline = PDMPath([StateSE2(*p) for p in path])
    road = Polygon(vertices(origin, np.array([[-20.,-4.],[100.,-4.],[100.,4.],[-20.,4.]])))
    dm = PDMDrivableMap(['road', 'lane'], [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.LANE], [road, road])
    observation = PDMObservation(sampling, sampling, 100, observation_sample_res=1)
    observation._occupancy_maps = [PDMOccupancyMap([], []) for _ in range(51)]
    observation._unique_objects = {}
    observation._initialized = True
    cache = SimpleNamespace(ego_state=ego, trajectory=transform_trajectory(Trajectory(gt_local * [1.08,1,1]), ego),
        observation=observation, centerline=centerline, route_lane_ids=['lane'], drivable_area_map=dm)
    future = local_to_world(np.c_[np.arange(11) * 3., np.zeros((11, 2))], origin)
    frames = [SimpleNamespace(timestamp=1_000_000+500_000*i, ego_status=SimpleNamespace(ego_pose=p)) for i,p in enumerate(future)]
    scene = SimpleNamespace(scene_metadata=SimpleNamespace(num_history_frames=1), frames=frames,
        get_future_trajectory=lambda n: gt,
        map_api=SimpleNamespace(get_map_object=lambda *args: SimpleNamespace(speed_limit_mps=13.9)))
    evaluator = OfficialEvaluator(scene, cache, Config(population=4, generations=1))
    plain = pdm_score(cache, gt, sampling, PDMSimulator(sampling), PDMScorer(sampling, vehicle_parameters=vehicle))
    assert abs(plain.score - evaluator.gt.score) < 1e-10
    assert evaluator.replay_error < 1e-8
    assert evaluator.gt.executed.shape == (41, 11)
    assert evaluator.gt.controls.shape == (40, 2)
    assert evaluator.gt.passed
    first = evaluator.evaluate_theta(np.r_[np.full(5, .1), np.zeros(5)])
    evaluator.evaluate_theta(np.r_[np.full(5, -.1), np.zeros(5)])
    again = evaluator.evaluate_poses(first.poses)
    assert abs(first.score - again.score) < 1e-10
    assert np.allclose(first.executed, again.executed, atol=1e-9, rtol=0)
    assert observation.collided_track_ids == []
    assert not first.export_certified
