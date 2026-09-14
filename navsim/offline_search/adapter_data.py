"""Cache reconstruction and clock diagnostics without importing the Linux map stack."""
from __future__ import annotations

import numpy as np
from .geometry import Actor, angle_delta, vertices


class DataAlignmentError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def frame_clock(timestamps):
    """Use NAVSIM's nominal frame clock, retaining actual acquisition timestamps.

    A deviation of half a frame or more is ambiguous (possible missing frames),
    so it remains an error. Millisecond acquisition jitter is not a new clock.
    """
    stamp = np.asarray(timestamps, dtype=np.int64)
    actual = (stamp - stamp[0]) / 1e6
    nominal = np.arange(len(stamp)) * .5
    gaps = np.diff(actual)
    info = {'clock_policy': 'navsim_nominal_frame_clock_0.5s',
            'raw_timestamps_us': stamp.tolist(), 'actual_relative_s': actual.tolist(),
            'nominal_relative_s': nominal.tolist(), 'intervals_s': gaps.tolist(),
            'max_absolute_jitter_s': float(np.max(abs(actual - nominal))),
            'old_2ms_check_would_fail': bool(np.any(abs(actual - nominal) > .002))}
    if len(stamp) != 11 or np.any(gaps <= 0):
        raise DataAlignmentError('invalid/nonmonotonic raw frame timestamps', info)
    if np.any(abs(actual - nominal) >= .25) or np.any(abs(gaps - .5) >= .25):
        raise DataAlignmentError('raw frame spacing ambiguous relative to nominal 2 Hz; inspect diagnostics', info)
    return info


def cached_actors(observation):
    """Preserve partial tracks with NaN gaps; never freeze or delete them."""
    if observation._observation_sample_res != 1 or abs(observation._sample_interval - .1) > 1e-9:
        raise ValueError('Interval checks require 10 Hz cache with observation_sample_res=1')
    maps = [observation[i] for i in range(41)]
    token_sets = [set(m.tokens) for m in maps]
    ids = sorted(set().union(*token_sets))
    actors, unknown = [], []
    for token in ids:
        if token.startswith(observation.red_light_token):
            continue
        observed = [i for i in range(41) if token in token_sets[i]]
        first = maps[observed[0]][token]
        xy = np.asarray(first.exterior.coords)[:-1]
        if xy.shape != (4, 2) or not first.is_valid:
            unknown.append('non_box_actor_unverifiable:' + token)
            continue
        center = np.asarray(first.centroid.coords)[0]
        local = xy - center
        base_angle = np.arctan2(local[0, 1], local[0, 0])
        poses = np.full((41, 3), np.nan)
        failed = False
        for i in observed:
            polygon = maps[i][token]
            curr = np.asarray(polygon.exterior.coords)[:-1]
            c = np.asarray(polygon.centroid.coords)[0]
            if curr.shape != (4, 2) or not polygon.is_valid:
                failed = True
                break
            yaw = angle_delta(base_angle, np.arctan2(*(curr[0] - c)[::-1]))
            pose = np.array([*c, yaw])
            # Refuse a shape/vertex-order mismatch rather than guessing a pose.
            reconstructed = vertices(pose, local)
            if np.max(np.abs(reconstructed - curr)) > 1e-4:
                failed = True
                break
            poses[i] = pose
        if failed:
            unknown.append('non_rigid_actor_unverifiable:' + token)
        else:
            actors.append(Actor(token, poses, local))
    return actors, unknown


def speed_limit_at(lanes, point):
    matches = [lane for lane in lanes if lane['polygon'].covers(point)]
    on_route = [lane for lane in matches if lane['on_route']]
    relevant = on_route or matches
    details = [{k: v for k, v in lane.items() if k != 'polygon'} for lane in matches]
    limits = [lane['limit_mps'] for lane in relevant]
    known = bool(limits) and all(x is not None and np.isfinite(x) and x > 0 for x in limits)
    return {'limit_mps': float(min(limits)) if known else None,
            'source': 'minimum_matching_on_route_map_limits' if on_route else 'minimum_matching_map_limits',
            'reason': None if known else 'missing_map_limit' if matches else 'no_matching_lane',
            'matches': details}
