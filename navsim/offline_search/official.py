"""V1 adapter: calls the unchanged public pdm_score for every candidate separately."""
from __future__ import annotations

from dataclasses import asdict
import copy
import lzma
import pickle

import numpy as np
from scipy.optimize import lsq_linear
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union

from nuplan.common.actor_state.state_representation import TimeDuration
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from .core import Candidate, candidate_id, BOUNDS, basis, residual
from .geometry import NominalValidator, angle_delta, vertices, terminal_heading_check, terminal_speed_check
from .adapter_data import cached_actors, frame_clock, speed_limit_at, experiment_speed_limit


class RecordingTracker(BatchLQRTracker):
    def __init__(self):
        super().__init__()
        self.commands = []

    def update(self, states):
        self.commands = []
        return super().update(states)

    def track_trajectory(self, *args, **kwargs):
        command = super().track_trajectory(*args, **kwargs)
        self.commands.append(command.copy())
        return command


class RecordingSimulator(PDMSimulator):
    def __init__(self, sampling):
        super().__init__(sampling)
        self._tracker = RecordingTracker()
        self.executed = None

    def simulate_proposals(self, states, initial_ego_state):
        result = super().simulate_proposals(states, initial_ego_state)
        self.executed = result.copy()
        return result


def world_to_local(poses, origin):
    out = np.asarray(poses, dtype=float).copy()
    c, s = np.cos(origin[2]), np.sin(origin[2])
    out[:, :2] = (out[:, :2] - origin[:2]) @ np.array([[c, -s], [s, c]])
    out[:, 2] = angle_delta(origin[2], out[:, 2])
    return out


def local_to_world(poses, origin):
    out = np.asarray(poses, dtype=float).copy()
    c, s = np.cos(origin[2]), np.sin(origin[2])
    out[:, :2] = out[:, :2] @ np.array([[c, s], [-s, c]]) + origin[:2]
    out[:, 2] = angle_delta(0, out[:, 2] + origin[2])
    return out


class OfficialEvaluator:
    def __init__(self, scene, cache, config):
        self.scene, self.cache, self.config = scene, cache, config
        self.sampling = TrajectorySampling(num_poses=40, interval_length=.1)
        self.input_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
        self.simulator = RecordingSimulator(self.sampling)
        self.scorer = PDMScorer(self.sampling, vehicle_parameters=cache.ego_state.car_footprint.vehicle_parameters)
        self.origin = np.array(cache.ego_state.rear_axle.serialize())
        self.vehicle = cache.ego_state.car_footprint.vehicle_parameters
        i = scene.scene_metadata.num_history_frames - 1
        frames = scene.frames[i:i + 11]
        if len(frames) != 11:
            raise ValueError("Need current frame plus 10 future frames (5 seconds)")
        self.time_diagnostics = frame_clock([f.timestamp for f in frames])
        pose_error = np.linalg.norm(frames[0].ego_status.ego_pose[:2] - self.origin[:2])
        yaw_error = abs(angle_delta(frames[0].ego_status.ego_pose[2], self.origin[2]))
        if pose_error > .001 or yaw_error > .001 or abs(frames[0].timestamp - cache.ego_state.time_point.time_us) > 2000:
            raise ValueError("scene/cache initial pose or timestamp mismatch")
        self.gt_poses = np.asarray(scene.get_future_trajectory(8).poses, dtype=float)
        self.raw_path = np.asarray([f.ego_status.ego_pose for f in frames], dtype=float)
        self.gt_path = LineString(self.raw_path[:, :2])
        footprint = cache.ego_state.car_footprint.geometry
        corners = np.asarray(footprint.exterior.coords)[:-1]
        # Rotate corners into the rear-axle frame; center and rear axle differ.
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        self.ego_local = (corners - self.origin[:2]) @ np.array([[c, -s], [s, c]])
        self.actors, self.unknown = cached_actors(cache.observation)
        dm = cache.drivable_area_map
        layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                  SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA]
        self.road = unary_union([dm[dm.tokens[j]] for j in dm.get_indices_of_map_type(layers)])
        if self.road.is_empty or not self.road.is_valid:
            raise ValueError("missing/invalid drivable union")
        self.intersections = unary_union([dm[dm.tokens[j]] for j in
            dm.get_indices_of_map_type([SemanticMapLayer.INTERSECTION])])
        # Use the available five-second GT reference so faster four-second
        # candidates are not rejected merely for passing the four-second endpoint.
        speed_sampling = TrajectorySampling(num_poses=50, interval_length=.1)
        speed_input = Trajectory(np.asarray(scene.get_future_trajectory(10).poses, dtype=float),
                                 TrajectorySampling(num_poses=10, interval_length=.5))
        speed_reference = get_trajectory_as_array(transform_trajectory(speed_input, cache.ego_state),
                                                  speed_sampling, cache.ego_state.time_point)
        speed_simulator = RecordingSimulator(speed_sampling)
        self.gt_speed_states = speed_simulator.simulate_proposals(
            speed_reference[None], cache.ego_state)[0].copy()
        self.lanes = []
        for j in dm.get_indices_of_map_type([SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]):
            token, layer = dm.tokens[j], dm.map_types[j]
            lane = scene.map_api.get_map_object(token, layer)
            limit = getattr(lane, "speed_limit_mps", None)
            self.lanes.append({'id': token, 'layer': layer.name, 'polygon': dm[token],
                               'on_route': token in cache.route_lane_ids,
                               'limit_mps': float(limit) if limit is not None and np.isfinite(limit) and limit > 0 else None})
        self.gt_terminal_speed = None
        self.validator = NominalValidator(config, self.ego_local, self.actors, self.road,
                self.gt_path, self.vehicle.wheel_base, self.unknown, self.speed_limit,
                lambda state: terminal_heading_check(state, self.gt_speed_states, cache.centerline.linestring),
                lambda state: terminal_speed_check(state, self.gt_terminal_speed))
        self.evaluations = 0
        self.gt = self.evaluate_poses(self.gt_poses)
        self.gt.island = "GT"
        self.base_commands = self.gt.controls.copy()
        replay = self.replay(self.base_commands)
        self.replay_error = float(abs(replay - self.gt.executed).max())
        if self.replay_error > 1e-8:
            raise ValueError(f"GT command replay mismatch: {self.replay_error}")
        self.roundtrip = self.evaluate_theta(np.zeros(10))
        self.roundtrip.island = "roundtrip_zero"
        # Repeat after another trajectory, using the same mutable scorer/simulator.
        repeat = self.evaluate_poses(self.gt_poses)
        if not np.allclose(repeat.executed, self.gt.executed, atol=1e-9, rtol=0):
            raise ValueError("GT rollout changed after another candidate")
        if any(abs(repeat.metrics[k] - self.gt.metrics[k]) > 1e-10 for k in self.gt.metrics):
            raise ValueError("GT scoring changed after another candidate")

    def speed_limit(self, state):
        center = Polygon(vertices(state[:3], self.ego_local)).centroid
        sample = speed_limit_at(self.lanes, center)
        return experiment_speed_limit(sample, state[:2], self.intersections.covers(center),
                                      self.gt_speed_states)

    def evaluate_poses(self, poses):
        poses = np.asarray(poses, dtype=float)
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise ValueError("invalid 8-pose candidate")
        trajectory = Trajectory(poses.copy(), self.input_sampling)
        # Each call evaluates [fixed PDM reference, one candidate], never the search pool.
        result = pdm_score(self.cache, trajectory, self.sampling, self.simulator, self.scorer)
        metrics = {k: float(v) for k, v in asdict(result).items()}
        if not np.isfinite(list(metrics.values())).all() or any(v < -1e-6 or v > 1 + 1e-6 for v in metrics.values()):
            raise ValueError("invalid official score; no fallback labels")
        executed = self.simulator.executed[1].copy()
        commands = np.stack(self.simulator._tracker.commands)[:, 1].copy()
        ref = get_trajectory_as_array(transform_trajectory(trajectory, self.cache.ego_state),
                                     self.sampling, self.cache.ego_state.time_point)
        if self.gt_terminal_speed is None:
            self.gt_terminal_speed = float(executed[-1, 3])
        gate = self.validator.check(executed, metrics)
        gate["tracking_rms_xy"] = float(np.sqrt(np.mean(np.sum((ref[:, :2] - executed[:, :2]) ** 2, axis=1))))
        self.evaluations += 1
        return Candidate(candidate_id(poses), poses.copy(), ref, executed, metrics, gate, controls=commands)

    def replay(self, commands):
        model = copy.deepcopy(self.simulator._motion_model)
        model._vehicle = self.vehicle
        out = np.empty((41, 11))
        out[0] = ego_state_to_state_array(self.cache.ego_state)
        for t in range(40):
            out[t + 1] = model.propagate_state(out[t:t + 1].copy(), commands[t:t + 1].copy(),
                                               TimeDuration.from_s(.1))[0]
        return out

    def evaluate_theta(self, theta):
        generated = self.replay(self.base_commands + residual(theta))
        poses = world_to_local(generated[5::5, :3], self.origin)
        candidate = self.evaluate_poses(poses)
        candidate.generated, candidate.theta = generated, theta.copy()
        return candidate

    def seeds(self):
        seeds, notes = {"B0": np.zeros(10)}, []
        schedules = {"B1": [0, .6, .6, 0, 0, 0], "B2": [0, 0, 0, 0, .6, 0],
                     "B3": [0, -.6, -.6, 0, 0, 0], "B4": [0, 0, 0, 0, -.6, 0],
                     "B5": [0, .4, .4, 0, -.4, 0]}
        # Same geometric path with a changed speed schedule. No endpoint clipping.
        path = self.raw_path
        length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
        unique = np.r_[True, np.diff(length) > 1e-6]
        path, length = path[unique], length[unique]
        for name, schedule in schedules.items():
            reason = "seed_path_unavailable"
            if len(path) >= 2:
                for scale in [1., .5, .25]:
                    da = basis() @ (np.asarray(schedule[1:]) * scale)
                    speed = self.gt.executed[:, 3] + np.r_[0., np.cumsum(da) * .1]
                    distance = np.r_[0., np.cumsum(speed[:-1]) * .1]
                    if speed.min() < 0 or distance.max() > length[-1]:
                        reason = "seed_path_coverage_or_reverse"
                        continue
                    reference = np.column_stack([np.interp(distance, length, path[:, j]) for j in range(2)] +
                                                [np.interp(distance, length, np.unwrap(path[:, 2]))])
                    target = self.evaluate_poses(world_to_local(reference[5::5], self.origin))
                    delta = target.controls - self.base_commands
                    theta = np.r_[lsq_linear(basis(), delta[:, 0], bounds=(-1, 1)).x,
                                  lsq_linear(basis(), delta[:, 1], bounds=(-.03, .03)).x]
                    fitted = self.evaluate_theta(theta)
                    error = np.sqrt(np.mean(np.sum((fitted.executed[:, :2] - target.executed[:, :2]) ** 2, axis=1)))
                    target_path = LineString(target.executed[:, :2])
                    lateral = max(Point(*s[:2]).distance(target_path) for s in fitted.executed)
                    if error <= .25 and lateral <= .5:
                        seeds[name] = theta
                        notes.append({"island": name, "scale": scale, "rms_xy": float(error), "max_lateral": lateral})
                        break
                    reason = "seed_fit_distortion"
            if name not in seeds:
                notes.append({"island": name, "unavailable": reason})
        for name, sign in [("B6", 1), ("B7", -1)]:
            seeds[name] = np.r_[np.zeros(5), sign * np.array([.015, .015, -.015, -.015, 0])]
        return seeds, notes
