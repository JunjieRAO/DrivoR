import hashlib
import json
import math

import numpy as np
import torch
from shapely import affinity
from shapely.geometry import LineString, Point, box
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.agents.drivoR.scene_geometry.geometry_targets import (
    AGENT_CLASSES, local_geometry, vectorize_lines,
)
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


MAP_TYPES = (
    "lane_center", "connector_center", "lane_left", "lane_right", "connector_left",
    "connector_right", "stop_line", "crosswalk", "walkway", "intersection_boundary",
)
MAP_TYPE_QUOTAS = (12, 8, 8, 8, 6, 6, 6, 4, 3, 3)


class StructuredSceneFeatureBuilder(AbstractFeatureBuilder):
    requires_scene = True

    def __init__(self, config):
        options = config.structured_input
        self.roi = list(options.get("roi", [-32., -32., 64., 32.]))
        self.max_agents = int(options.get("max_agents", 128))
        self.max_map_lines = int(options.get("max_map_lines", 256))
        self.map_points = int(options.get("map_points", 20))
        self.max_lights = int(options.get("max_lights", 16))
        self.max_segment_length = float(options.get("max_segment_length", 20.0))
        self.agent_selection = options.get("agent_selection", "balanced")
        self.agent_risk_horizon = float(options.get("agent_risk_horizon", 5.0))
        self.agent_risk_distance = float(options.get("agent_risk_distance", 2.0))
        self.map_selection = options.get("map_selection", "balanced")
        self.map_type_quotas = [float(value) for value in options.get("map_type_quotas", MAP_TYPE_QUOTAS)]
        self.map_segments_per_source = int(options.get("map_segments_per_source", 2))
        self.light_selection = options.get("light_selection", "relevant")
        self.modalities = {name: bool(options.get(f"use_{name}", True))
                           for name in ("agents", "map", "traffic_lights")}
        if min(self.max_agents, self.max_map_lines, self.max_lights) < 1 or self.map_points < 2:
            raise ValueError("Structured input capacities must be positive and map_points must be at least two")
        if len(self.roi) != 4 or self.roi[0] >= self.roi[2] or self.roi[1] >= self.roi[3]:
            raise ValueError("Invalid structured input ROI")
        if self.max_segment_length <= 0:
            raise ValueError("Structured map segments must have positive length")
        if self.agent_selection not in ("balanced", "nearest"):
            raise ValueError("agent_selection must be balanced or nearest")
        if self.agent_risk_horizon <= 0 or self.agent_risk_distance < 0:
            raise ValueError("Invalid structured agent risk horizon or distance")
        if self.map_selection not in ("balanced", "nearest") or self.light_selection not in ("relevant", "nearest"):
            raise ValueError("Invalid structured map/light selection mode")
        if (len(self.map_type_quotas) != len(MAP_TYPES) or
                any(not math.isfinite(value) or value < 0 for value in self.map_type_quotas) or
                sum(self.map_type_quotas) <= 0 or self.map_segments_per_source < 1):
            raise ValueError("Invalid structured map quotas or source-segment budget")
        specification = {key: value for key, value in vars(self).items()}
        digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:12]
        self._name = f"structured_current_v2_{digest}"

    def get_unique_name(self):
        return self._name

    def _in_roi(self, position):
        return self.roi[0] <= position[0] <= self.roi[2] and self.roi[1] <= position[1] <= self.roi[3]

    @staticmethod
    def _velocities(frame):
        velocities = getattr(frame.annotations, "velocity_3d", None)
        if velocities is None:
            return np.full((len(frame.annotations.boxes), 2), np.nan)
        return np.asarray(velocities, dtype=np.float64)[:, :2]

    def _select_agents(self, rows):
        nearest = sorted(rows, key=lambda row: (row["distance"], row["token"]))
        if self.agent_selection == "nearest":
            return nearest[:self.max_agents]
        selected = nearest[:(self.max_agents + 1) // 2]
        seen = {row["token"] for row in selected}
        risks = sorted((row for row in rows if row["risk"] is not None and row["token"] not in seen),
                       key=lambda row: (*row["risk"], row["distance"], row["token"]))
        selected.extend(risks[:self.max_agents // 4])
        seen.update(row["token"] for row in selected)
        sectors = [[] for _ in range(8)]
        for row in nearest:
            if row["token"] not in seen:
                angle = math.atan2(row["values"][1], row["values"][0])
                sector = int((angle + math.pi) * 8 / (2 * math.pi)) % 8
                sectors[sector].append(row)
        while len(selected) < self.max_agents:
            added = False
            for sector in (4, 3, 5, 2, 6, 1, 7, 0):
                if sectors[sector] and len(selected) < self.max_agents:
                    selected.append(sectors[sector].pop(0))
                    added = True
            if not added:
                break
        return selected

    def _agents(self, scene, current_index, result):
        frame = scene.frames[current_index]
        velocities = self._velocities(frame)
        ego_velocity = np.asarray(getattr(frame.ego_status, "ego_velocity", [np.nan, np.nan]))[:2]
        vehicle = get_pacifica_parameters()
        ego_center = np.array([vehicle.rear_axle_to_center, 0.0])
        ego_polygon = box(ego_center[0] - vehicle.length / 2, -vehicle.width / 2,
                          ego_center[0] + vehicle.length / 2, vehicle.width / 2)
        previous_tracks = {}
        elapsed = 0.0
        if current_index > 0:
            previous = scene.frames[current_index - 1]
            elapsed = (frame.timestamp - previous.timestamp) * 1e-6
            angle = previous.ego_status.ego_pose[2] - frame.ego_status.ego_pose[2]
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            previous_velocities = self._velocities(previous)
            previous_tracks = {str(token): rotation @ velocity for token, velocity in
                               zip(previous.annotations.track_tokens, previous_velocities)
                               if np.isfinite(velocity).all()}
        rows = []
        for index, (bounds, name) in enumerate(zip(frame.annotations.boxes, frame.annotations.names)):
            if name not in AGENT_CLASSES or not np.isfinite(bounds[[0, 1, 3, 4, 6]]).all():
                continue
            if min(bounds[3:5]) <= 0 or not self._in_roi(bounds[:2]):
                continue
            velocity_valid = bool(np.isfinite(velocities[index]).all())
            velocity = velocities[index] if velocity_valid else np.zeros(2)
            token = str(frame.annotations.track_tokens[index])
            acceleration_valid = velocity_valid and elapsed > 0 and token in previous_tracks
            acceleration = (velocity - previous_tracks[token]) / elapsed if acceleration_valid else np.zeros(2)
            values = np.r_[bounds[:2], np.sin(bounds[6]), np.cos(bounds[6]), bounds[4], bounds[3],
                           velocity, acceleration, float(velocity_valid), float(acceleration_valid)]
            polygon = affinity.rotate(box(-bounds[3] / 2, -bounds[4] / 2, bounds[3] / 2, bounds[4] / 2),
                                      bounds[6], use_radians=True)
            polygon = affinity.translate(polygon, bounds[0], bounds[1])
            distance = ego_polygon.distance(polygon)
            risk = None
            if velocity_valid and np.isfinite(ego_velocity).all():
                relative_velocity = velocity - ego_velocity
                speed_squared = float(relative_velocity @ relative_velocity)
                time = (float(np.clip(-(bounds[:2] - ego_center) @ relative_velocity / speed_squared,
                                      0, self.agent_risk_horizon)) if speed_squared > 1e-6 else 0.0)
                closest = affinity.translate(polygon, *(relative_velocity * time))
                closest_gap = ego_polygon.distance(closest)
                if closest_gap <= self.agent_risk_distance:
                    risk = (closest_gap, time)
            rows.append({"distance": distance, "token": token, "risk": risk,
                         "category": AGENT_CLASSES[name], "values": values})
        for index, row in enumerate(self._select_agents(rows)):
            result["structured_agents"][index] = torch.tensor(row["values"], dtype=torch.float32)
            result["structured_agent_types"][index] = row["category"]
            result["structured_agent_valid"][index] = True

    def _map_quotas(self):
        scaled = np.asarray(self.map_type_quotas) * self.max_map_lines / sum(self.map_type_quotas)
        quotas = np.floor(scaled).astype(int)
        order = sorted(range(len(quotas)), key=lambda category: (-(scaled[category] - quotas[category]), category))
        for category in order[:self.max_map_lines - int(quotas.sum())]:
            quotas[category] += 1
        return quotas

    def _select_map(self, candidates):
        if self.map_selection == "nearest":
            return sorted(candidates, key=lambda row: (row["distance"], row["category"], row["source"],
                                                       tuple(row["points"].ravel())))[:self.max_map_lines]
        counts = {}
        for row in sorted(candidates, key=lambda row: (*row["priority"], row["category"], row["source"],
                                                       tuple(row["points"].ravel()))):
            key = (row["category"], row["source"])
            row["round"] = counts.get(key, 0) // self.map_segments_per_source
            counts[key] = counts.get(key, 0) + 1
        ordered = sorted(candidates, key=lambda row: (row["round"], *row["priority"], row["category"],
                                                      row["source"], tuple(row["points"].ravel())))
        selected = []
        seen = set()
        quotas = self._map_quotas()
        for category, quota in enumerate(quotas):
            matching = [row for row in ordered if row["category"] == category][:quota]
            selected.extend(matching)
            seen.update(id(row) for row in matching)
        selected.extend(row for row in ordered if id(row) not in seen)
        return selected[:self.max_map_lines]

    @staticmethod
    def _reference_lane(objects, pose):
        lanes = [(str(lane.id), local_geometry(lane.baseline_path.linestring, pose))
                 for lane in objects.get(SemanticMapLayer.LANE, [])]
        if not lanes:
            return None
        return min(lanes, key=lambda item: (item[1].distance(Point(0., 0.)), item[0]))[1]

    def _map_and_lights(self, scene, frame, result):
        if scene.map_api is None:
            raise ValueError("Structured map/traffic-light input requires the current scene map")
        supported = scene.map_api.get_available_map_objects()
        required = [SemanticMapLayer.LANE_CONNECTOR]
        if self.modalities["map"]:
            required.append(SemanticMapLayer.LANE)
        if any(layer not in supported for layer in required):
            raise ValueError("Structured input requires lane/lane-connector map geometry")
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.STOP_LINE,
                  SemanticMapLayer.CROSSWALK, SemanticMapLayer.WALKWAYS, SemanticMapLayer.INTERSECTION]
        if not self.modalities["map"]:
            layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        requested = [layer for layer in layers if layer in supported]
        pose = frame.ego_status.ego_pose
        radius = math.hypot(max(abs(self.roi[0]), abs(self.roi[2])),
                            max(abs(self.roi[1]), abs(self.roi[3]))) + 20.0
        objects = scene.map_api.get_proximal_map_objects(Point2D(*pose[:2]), radius, requested)
        reference = self._reference_lane(objects, pose)

        if self.modalities["map"]:
            lines = {category: [] for category in range(len(MAP_TYPES))}
            for layer, center, left, right in (
                (SemanticMapLayer.LANE, 0, 2, 3), (SemanticMapLayer.LANE_CONNECTOR, 1, 4, 5),
            ):
                for lane in objects[layer]:
                    for category, geometry in (
                        (center, lane.baseline_path), (left, lane.left_boundary), (right, lane.right_boundary),
                    ):
                        source = f"{category}:{getattr(geometry, 'id', lane.id)}"
                        lines[category].append((source, local_geometry(geometry.linestring, pose)))
            for category, layer in ((6, SemanticMapLayer.STOP_LINE), (7, SemanticMapLayer.CROSSWALK),
                                    (8, SemanticMapLayer.WALKWAYS), (9, SemanticMapLayer.INTERSECTION)):
                lines[category].extend((f"{category}:{getattr(item, 'id', item.polygon.wkb_hex)}",
                                        local_geometry(item.polygon.boundary, pose))
                                       for item in objects.get(layer, []))
            candidates, seen = [], set()
            for category, geometries in lines.items():
                for source, geometry in sorted(geometries, key=lambda item: item[0]):
                    for points in vectorize_lines([geometry], self.roi, self.map_points,
                                                  self.max_segment_length, directed=category in (0, 1)):
                        fingerprint = tuple(np.round(points, 3).ravel())
                        if category not in (0, 1):
                            fingerprint = min(fingerprint, tuple(np.round(points[::-1], 3).ravel()))
                        if (category, fingerprint) in seen:
                            continue
                        seen.add((category, fingerprint))
                        line = LineString(points)
                        distance = line.distance(Point(0., 0.))
                        corridor_gap = line.distance(reference) if reference is not None else abs(points[:, 1]).min()
                        score = distance + .5 * min(corridor_gap, 20.) + (15. if points[:, 0].max() < -5. else 0.)
                        candidates.append({"category": category, "points": points, "source": source,
                                           "distance": distance, "priority": (score, distance)})
            for index, row in enumerate(self._select_map(candidates)):
                result["structured_map_points"][index] = torch.tensor(row["points"], dtype=torch.float32)
                result["structured_map_types"][index] = row["category"]
                result["structured_map_valid"][index] = True

        if self.modalities["traffic_lights"]:
            connectors = {str(item.id): item for item in objects[SemanticMapLayer.LANE_CONNECTOR]}
            lights = []
            statuses = {}
            for token, is_red in frame.traffic_lights:
                statuses[str(token)] = statuses.get(str(token), False) or bool(is_red)
            for token, is_red in statuses.items():
                connector = connectors.get(token)
                if connector is None:
                    continue
                position = np.asarray(local_geometry(connector.baseline_path.linestring, pose).coords[0])[:2]
                if self._in_roi(position):
                    direction = np.asarray(local_geometry(connector.baseline_path.linestring, pose).coords[-1])[:2] - position
                    gap = Point(*position).distance(reference) if reference is not None else abs(position[1])
                    relevant = position[0] >= -5. and gap <= 8. and direction[0] >= 0.
                    lights.append({"distance": float(np.linalg.norm(position)), "token": token,
                                   "relevant": relevant, "values": [*position, float(is_red)]})
            lights.sort(key=lambda row: ((not row["relevant"] if self.light_selection == "relevant" else False),
                                         row["distance"], -row["values"][2], row["token"]))
            for index, row in enumerate(lights[:self.max_lights]):
                result["structured_lights"][index] = torch.tensor(row["values"], dtype=torch.float32)
                result["structured_light_valid"][index] = True

    def compute_features(self, scene):
        if not hasattr(scene, "scene_metadata") or not hasattr(scene, "frames"):
            raise ValueError("Structured current-frame GT features require Scene, not sensor-only AgentInput")
        current_index = scene.scene_metadata.num_history_frames - 1
        if current_index < 0:
            raise ValueError("Structured input requires at least one current/history frame")
        result = {
            "structured_agents": torch.zeros(self.max_agents, 12),
            "structured_agent_types": torch.zeros(self.max_agents, dtype=torch.long),
            "structured_agent_valid": torch.zeros(self.max_agents, dtype=torch.bool),
            "structured_map_points": torch.zeros(self.max_map_lines, self.map_points, 2),
            "structured_map_types": torch.zeros(self.max_map_lines, dtype=torch.long),
            "structured_map_valid": torch.zeros(self.max_map_lines, dtype=torch.bool),
            "structured_lights": torch.zeros(self.max_lights, 3),
            "structured_light_valid": torch.zeros(self.max_lights, dtype=torch.bool),
        }
        frame = scene.frames[current_index]
        if self.modalities["agents"]:
            self._agents(scene, current_index, result)
        if self.modalities["map"] or self.modalities["traffic_lights"]:
            self._map_and_lights(scene, frame, result)
        return result