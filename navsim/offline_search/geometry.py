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

    @property
    def valid(self):
        return np.isfinite(self.poses).all(axis=1)


def missing_interval_region(actor, index, speed_bound, dt):
    """Over-approximate an unknown footprint under an explicit speed bound.

    Use nearest observed centers on both sides when available. Each disk covers
    the WHOLE interval, including arbitrary heading via the box circumradius.
    No pose is extrapolated or held stationary. Disjointness is conditional on
    the stated engineering bound; it is not a real-world certificate.
    """
    known = np.flatnonzero(actor.valid)
    before = known[known <= index]
    after = known[known >= index + 1]
    anchors = ([before[-1]] if len(before) else []) + ([after[0]] if len(after) else [])
    if not anchors:
        return None
    radius = float(np.linalg.norm(actor.local, axis=1).max())
    region = None
    for j in anchors:
        reach = speed_bound * max(abs(index - j), abs(index + 1 - j)) * dt + radius
        disk = inflated(Point(*actor.poses[j, :2]), reach)
        region = disk if region is None else region.intersection(disk)
    return region


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
        self.actor_regions = {}
        self.lifecycle_diagnostics = []
        for actor in actors:
            known = np.flatnonzero(actor.valid)
            observed_speed = (float(np.max(np.linalg.norm(np.diff(actor.poses[known, :2], axis=0), axis=1)
                              / (np.diff(known) * config.dt))) if len(known) > 1 else 0.)
            bound = max(config.missing_actor_speed_bound_mps, observed_speed * 1.01)
            missing_intervals = []
            for t in range(40):
                if not actor.valid[t:t + 2].all():
                    self.actor_regions[actor.token, t] = None  # Missing intervals are diagnostic only.
                    missing_intervals.append(t)
            self.lifecycle_diagnostics.append({'token': actor.token, 'observed_indices': known.tolist(),
                'missing_interval_indices': missing_intervals, 'engineering_speed_bound_mps': bound,
                'observed_max_center_speed_mps': observed_speed,
                'partial': bool(missing_intervals)})

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
        speed_samples = []
        if self.speed_limit_fn is not None:
            for state in states:
                value = self.speed_limit_fn(state)
                if not isinstance(value, dict):
                    value = {'limit_mps': float(value) if value is not None and np.isfinite(value) and value > 0 else None,
                             'source': 'provided_limit', 'matches': []}
                speed_samples.append(value)
            limits = np.array([sample['limit_mps'] if sample['limit_mps'] is not None else np.nan for sample in speed_samples])
            known_limits = np.isfinite(limits) & (limits > 0)
            if any(x["limit_mps"] is None and not x.get("allowed_unbounded", False) for x in speed_samples):
                reasons.append("speed_limit_unverifiable")
            # Preserve known speeding even if another timestamp has no limit.
            if known_limits.any():
                bounds["speed_limit"] = max(0., float(np.max(v[known_limits] / limits[known_limits]) - 1))
        else:
            reasons.append("speed_limit_unverifiable")
        reasons.extend(k for k, amount in bounds.items() if amount > 1e-8)
        collisions, road_fail = 0, 0
        uncertain_intervals, excluded_intervals = 0, 0
        min_distance = float("inf")
        for t in range(40):
            ego_swept = envelope(states[t, :3], states[t + 1, :3], self.ego_local, cfg.collision_margin)
            for actor in self.actors:
                if len(actor.poses) < 41 or not np.isfinite(actor.poses[t:t + 2]).all():
                    for j in (t, t + 1):
                        if actor.valid[j] and inflated(Polygon(vertices(states[j, :3], self.ego_local)), cfg.collision_margin).intersects(
                                Polygon(vertices(actor.poses[j], actor.local))):
                            collisions += 1
                            events.append({'kind': 'observed_partial_actor_contact', 't': t * cfg.dt, 'actor': actor.token})
                    uncertain_intervals += 1
                    events.append({'kind': 'missing_actor_interval_not_checked',
                                   't': t * cfg.dt, 'actor': actor.token})
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
                "lifecycle_uncertain_interval_count": uncertain_intervals,
                "lifecycle_excluded_interval_count": excluded_intervals,
                "lifecycle_bound_assumption_used": False,
                "missing_actor_policy": "observed_intervals_only",
                "max_speed_mps": float(v.max()),
                "missing_actor_speed_bound_mps": cfg.missing_actor_speed_bound_mps,
                "speed_limit_diagnostics": {'samples': speed_samples,
                    'unknown_indices': [i for i, x in enumerate(speed_samples) if x['limit_mps'] is None],
                    'exceeded_indices': [i for i, x in enumerate(speed_samples) if x['limit_mps'] is not None and v[i] > x['limit_mps'] + 1e-8]},
                "violation": [collisions, 0 if not collisions else 1, road_fail,
                              int("gt_path_corridor" in reasons), max(bounds.values()),
                              int("official_submetrics" in reasons)],
                "pending_gates": ["raw_actor_coverage", "ordered_route_topology", "time_varying_traffic_lights",
                                  "one_second_tail", "13_case_stress", "anchor_stop_clustering",
                                  "missing_actor_motion_bound_validation"],
                "export_certified": False}
