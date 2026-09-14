"""Conservative swept-footprint checks for explicitly piecewise-SE2 replay."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon, MultiPoint, LineString, Point
from shapely.ops import unary_union


def angle_delta(a, b):
    return np.arctan2(np.sin(b - a), np.cos(b - a))


def interpolate_pose(a, b, t):
    out = (1 - t) * np.asarray(a) + t * np.asarray(b)
    out[2] = a[2] + t * angle_delta(a[2], b[2])
    return out


def vertices(pose, local):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.asarray(local) @ np.array([[c, s], [-s, c]]) + pose[:2]


def inflated(poly, radius):
    # Shapely approximates circles by inscribed polygons. Inflate sufficiently
    # that the polygonal approximation contains the intended circular buffer.
    return poly.buffer((radius + 1e-9) / np.cos(np.pi / 64), quad_segs=16)


def envelope(a, b, local, margin=0.):
    hull = MultiPoint(np.vstack([vertices(a, local), vertices(b, local)])).convex_hull
    radius = np.linalg.norm(local, axis=1).max()
    error = radius * angle_delta(a[2], b[2]) ** 2 / 8 + 1e-6
    return inflated(hull, margin + error)


def certify_separation(a, b, local, c, d, other_local, dt=.1, minimum=.0125, margin=.25):
    x, y = envelope(a, b, local, margin), envelope(c, d, other_local)
    if x.disjoint(y):
        return True, float(x.distance(y))
    if dt <= minimum + 1e-12:
        return False, 0.
    m, n = interpolate_pose(a, b, .5), interpolate_pose(c, d, .5)
    left, dl = certify_separation(a, m, local, c, n, other_local, dt / 2, minimum, margin)
    right, dr = certify_separation(m, b, local, n, d, other_local, dt / 2, minimum, margin)
    return left and right, min(dl, dr)


def certify_containment(a, b, local, road, dt=.1, minimum=.0125, margin=.1):
    swept = envelope(a, b, local, margin)
    if road.contains(swept) and road.boundary.disjoint(swept):
        return True
    if dt <= minimum + 1e-12:
        return False
    mid = interpolate_pose(a, b, .5)
    return (certify_containment(a, mid, local, road, dt / 2, minimum, margin)
            and certify_containment(mid, b, local, road, dt / 2, minimum, margin))


@dataclass
class Actor:
    token: str
    poses: np.ndarray  # World box-center SE2 at the same 0.1 s nodes as ego.
    local: np.ndarray


class NominalValidator:
    def __init__(self, config, ego_local, actors, road, gt_path, wheelbase,
                 unknown_reasons=(), speed_limit_fn=None):
        self.config = config
        self.ego_local = np.asarray(ego_local)
        self.actors, self.road = actors, road
        self.gt_path = gt_path
        self.wheelbase = wheelbase
        self.unknown = list(unknown_reasons)
        self.speed_limit_fn = speed_limit_fn

    def check(self, states, metrics):
        cfg = self.config
        reasons, events = list(self.unknown), []
        if states.shape != (41, 11) or not np.isfinite(states).all():
            return {"checked_nominal_pass": False, "data_verifiable": False,
                    "reasons": ["invalid_states"], "violation": [999] * 6, "events": []}
        required = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
                    "comfort", "driving_direction_compliance"]
        if not all(np.isfinite(metrics.get(k, np.nan)) and abs(metrics[k] - 1.) <= 1e-6 for k in required):
            reasons.append("official_submetrics")
        v, a, delta = states[:, 3], states[:, 5], states[:, 7]
        lat = v ** 2 * np.tan(delta) / self.wheelbase
        jerk = np.diff(a) / cfg.dt
        rate = np.diff(delta) / cfg.dt
        bounds = {"reverse": max(0., float(-v.min())),
                  "acceleration": max(0., float(a.max() / 2 - 1), float(-a.min() / 3.5 - 1)),
                  "lateral_acceleration": max(0., float(abs(lat).max() / 3 - 1)),
                  "jerk": max(0., float(abs(jerk).max() / 3.5 - 1)),
                  "steering_angle": max(0., float(abs(delta).max() / .5 - 1)),
                  "steering_rate": max(0., float(max(abs(rate).max(), abs(states[:, 8]).max()) / .3 - 1))}
        limits = None
        if self.speed_limit_fn is not None:
            limits = np.asarray([self.speed_limit_fn(s) for s in states], dtype=float)
            if not np.isfinite(limits).all() or (limits <= 0).any():
                reasons.append("speed_limit_unverifiable")
            else:
                bounds["speed_limit"] = max(0., float(np.max(v / limits) - 1))
        else:
            reasons.append("speed_limit_unverifiable")
        reasons.extend(k for k, amount in bounds.items() if amount > 1e-8)
        collisions, road_fail = 0, 0
        min_distance = float("inf")
        for t in range(40):
            for actor in self.actors:
                if len(actor.poses) < 41 or not np.isfinite(actor.poses[t:t + 2]).all():
                    reasons.append("actor_lifecycle_unverifiable")
                    continue
                ok, distance = certify_separation(states[t, :3], states[t + 1, :3], self.ego_local,
                    actor.poses[t], actor.poses[t + 1], actor.local, cfg.dt, cfg.min_interval, cfg.collision_margin)
                min_distance = min(min_distance, distance)
                if not ok:
                    collisions += 1
                    events.append({"kind": "collision_or_margin_uncertified", "t": t * cfg.dt, "actor": actor.token})
            if self.road is None or not certify_containment(states[t, :3], states[t + 1, :3],
                        self.ego_local, self.road, cfg.dt, cfg.min_interval, cfg.road_margin):
                road_fail += 1
                events.append({"kind": "road_envelope_uncertified", "t": t * cfg.dt})
        if collisions:
            reasons.append("collision_or_margin_uncertified")
        if road_fail:
            reasons.append("road_envelope_uncertified")
        # Continuous rear-axle path, not just eight input waypoints.
        path = LineString(states[:, :2])
        if not self.gt_path.buffer(cfg.gt_path_margin).covers(path):
            reasons.append("gt_path_corridor")
        data_ok = not self.unknown and not any("unverifiable" in x for x in reasons)
        return {"checked_nominal_pass": not reasons, "data_verifiable": data_ok,
                "reasons": sorted(set(reasons)), "events": events,
                "clearance_lower_bound": float(min_distance) if np.isfinite(min_distance) else 1e6,
                "clearance_no_actors": not bool(self.actors),
                "max_jerk": float(abs(jerk).max()), "max_lateral_acceleration": float(abs(lat).max()),
                "violation": [collisions, 0 if not collisions else 1, road_fail,
                              int("gt_path_corridor" in reasons), max(bounds.values()),
                              int("official_submetrics" in reasons)],
                "pending_gates": ["raw_actor_coverage", "ordered_route_topology", "time_varying_traffic_lights",
                                  "one_second_tail", "13_case_stress", "anchor_stop_clustering"],
                "export_certified": False}
