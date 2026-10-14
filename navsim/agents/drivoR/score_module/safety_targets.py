from collections import OrderedDict

import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely import contains_xy
from shapely.affinity import affine_transform
from shapely.ops import unary_union
from shapely.validation import make_valid

from nuplan.common.maps.maps_datatypes import SemanticMapLayer


_TARGET_CACHE = OrderedDict()


def build_safety_targets(metric_cache, bounds, resolution, time_horizon):
    ego = metric_cache.ego_state
    heading = ego.rear_axle.heading
    cosine, sine = np.cos(heading), np.sin(heading)
    origin = np.array([ego.rear_axle.x, ego.rear_axle.y])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    transform = [cosine, sine, -sine, cosine,
                 -cosine * origin[0] - sine * origin[1],
                 sine * origin[0] - cosine * origin[1]]
    area_map = metric_cache.drivable_area_map
    indices = area_map.get_indices_of_map_type([
        SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA,
    ])
    if not indices:
        raise ValueError("Metric cache contains no DAC drivable polygons")
    polygons = [area_map[area_map.tokens[index]] for index in indices]
    polygons = [polygon if polygon.is_valid else make_valid(polygon) for polygon in polygons]
    geometry = affine_transform(unary_union(polygons), transform)
    lower_x, lower_y, upper_x, upper_y = bounds
    if resolution <= 0 or upper_x <= lower_x or upper_y <= lower_y:
        raise ValueError("Invalid safety distance-field resolution or bounds")
    width = int(round((upper_x - lower_x) / resolution)) + 1
    height = int(round((upper_y - lower_y) / resolution)) + 1
    coordinates_x, coordinates_y = np.meshgrid(
        np.linspace(lower_x, upper_x, width), np.linspace(lower_y, upper_y, height),
    )
    inside = contains_xy(geometry, coordinates_x, coordinates_y)
    spacing = ((upper_y - lower_y) / (height - 1), (upper_x - lower_x) / (width - 1))
    if not inside.any():
        raise ValueError("No drivable pixels in safety distance field; check map and bounds")
    padded = np.pad(inside, 1, constant_values=False)
    sdf = (distance_transform_edt(padded, sampling=spacing)
           - distance_transform_edt(~padded, sampling=spacing))[1:-1, 1:-1]
    observation = metric_cache.observation
    interval = observation._sample_interval
    steps = int(round(time_horizon / interval))
    if not np.isclose(steps * interval, time_horizon):
        raise ValueError("Safety horizon must align with observation timestamps")
    excluded = set(observation.collided_track_ids)
    per_step = []
    for time_index in range(1, steps + 1):
        occupancy = observation[time_index]
        boxes = []
        for token in occupancy.tokens:
            if observation.red_light_token in token or token in excluded:
                continue
            polygon = occupancy[token].minimum_rotated_rectangle
            if polygon.geom_type != "Polygon":
                raise ValueError(f"Obstacle {token} has no rectangular footprint")
            corners = np.asarray(polygon.exterior.coords)[:4, :2]
            boxes.append((corners - origin).dot(rotation))
        per_step.append(boxes)
    count = max(1, max(len(boxes) for boxes in per_step))
    corners = np.zeros((steps, count, 4, 2), dtype=np.float32)
    valid = np.zeros((steps, count), dtype=bool)
    for time_index, boxes in enumerate(per_step):
        if boxes:
            corners[time_index, :len(boxes)] = boxes
            valid[time_index, :len(boxes)] = True
    vehicle = ego.car_footprint.vehicle_parameters
    return {
        "drivable_sdf": sdf.astype(np.float32),
        "sdf_bounds": np.asarray(bounds, dtype=np.float32),
        "vehicle": np.array([vehicle.length, vehicle.width, vehicle.rear_axle_to_center], dtype=np.float32),
        "obstacle_corners": corners,
        "obstacle_valid": valid,
    }


def cached_safety_targets(path, metric_cache, config):
    key = (str(path), tuple(config["bounds"]), config["resolution"], config["time_horizon"])
    if key not in _TARGET_CACHE:
        _TARGET_CACHE[key] = build_safety_targets(
            metric_cache, config["bounds"], config["resolution"], config["time_horizon"],
        )
        if len(_TARGET_CACHE) > 32:
            _TARGET_CACHE.popitem(last=False)
    _TARGET_CACHE.move_to_end(key)
    return _TARGET_CACHE[key]