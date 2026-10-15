"""Deterministic current-frame labels in the NAVSIM ego/rear-axle frame."""
import math
import numpy as np
import torch
from shapely.geometry import box, LineString
from shapely.ops import unary_union, transform, substring

SCHEMA_VERSION = "scene_geometry_v1.0"
AGENT_CLASSES = {"vehicle": 0, "pedestrian": 1, "bicycle": 2,
                 "traffic_cone": 3, "barrier": 3, "czone_sign": 3, "generic_object": 3}


def normalize_xy(xy, roi):
    lower, upper = np.array(roi[:2]), np.array(roi[2:])
    return (xy - (lower + upper) / 2) / ((upper - lower) / 2)


def local_geometry(geometry, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return transform(lambda a, b, z=None: ((np.asarray(a) - x) * c + (np.asarray(b) - y) * s,
                                          -(np.asarray(a) - x) * s + (np.asarray(b) - y) * c), geometry)


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [LineString(geometry)]
    return [line for part in getattr(geometry, "geoms", []) for line in line_parts(part)]


def vectorize_lines(lines, roi, points, max_length, directed=False):
    result, seen = [], set()
    for line in lines:
        # Canonicalize closed rings before splitting; no duplicate closing point segments.
        coords = np.asarray(line.coords)[:, :2]
        if line.is_ring and len(coords) > 2:
            coords = coords[:-1]
            first = np.lexsort((coords[:, 1], coords[:, 0]))[0]
            coords = np.roll(coords, -first, axis=0)
            line = LineString(np.vstack((coords, coords[0])))
        for clipped in line_parts(line.intersection(box(*roi))):
            if clipped.length < .1:
                continue
            n = max(1, math.ceil(clipped.length / max_length))
            for i in range(n):
                segment = substring(clipped, clipped.length * i / n, clipped.length * (i + 1) / n)
                xy = np.asarray([segment.interpolate(t, normalized=True).coords[0]
                                 for t in np.linspace(0, 1, points)])[:, :2]
                key = tuple(np.round(xy, 3).ravel())
                if not directed:
                    key = min(key, tuple(np.round(xy[::-1], 3).ravel()))
                if key not in seen:
                    seen.add(key)
                    result.append(xy)
    return result


def boundary_lines(polygons):
    # Union first, boundary second, crop last: preserves holes and removes lane seams.
    if not polygons:
        return []
    valid = [p if p.is_valid else p.buffer(0) for p in polygons if not p.is_empty]
    return line_parts(unary_union(valid).boundary) if valid else []


def build_geometry_targets(scene, cfg):
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    a, m, k = cfg.num_agent_queries, cfg.num_map_queries, cfg.map_points
    result = {}
    for kind, count, width in (("agent", a, 6), ("map", m, k * 2)):
        result.update({f"sg_{kind}_geometry": torch.zeros(count, width),
                       f"sg_{kind}_mask": torch.zeros(count, width, dtype=torch.bool),
                       f"sg_{kind}_class": torch.zeros(count, dtype=torch.long),
                       f"sg_{kind}_valid": torch.zeros(count, dtype=torch.bool),
                       f"sg_{kind}_available": torch.tensor(True),
                       f"sg_{kind}_overflow": torch.tensor(0)})
    roi = list(cfg.roi)
    agents, unknown = [], 0
    for index, (bbox, name) in enumerate(zip(frame.annotations.boxes, frame.annotations.names)):
        if name not in AGENT_CLASSES:
            unknown += 1
            continue
        if not np.isfinite(bbox[[0, 1, 3, 4, 6]]).all() or min(bbox[3:5]) <= 0:
            continue
        if not (roi[0] <= bbox[0] <= roi[2] and roi[1] <= bbox[1] <= roi[3]):
            continue
        geometry = np.r_[normalize_xy(bbox[:2], roi), np.log(bbox[3:5]), np.sin(bbox[6]), np.cos(bbox[6])]
        mask = np.ones(6, dtype=bool)
        if name in ("traffic_cone", "generic_object", "czone_sign"):
            mask[4:] = False
        agents.append((float(np.linalg.norm(bbox[:2])), index, AGENT_CLASSES[name], geometry, mask))
    agents.sort(key=lambda item: item[:2])
    result["sg_agent_overflow"] = torch.tensor(max(0, len(agents) - a))
    result["sg_agent_unknown"] = torch.tensor(unknown)
    for i, (_, _, cls, geometry, mask) in enumerate(agents[:a]):
        result["sg_agent_geometry"][i] = torch.tensor(geometry, dtype=torch.float32)
        result["sg_agent_mask"][i] = torch.tensor(mask)
        result["sg_agent_class"][i], result["sg_agent_valid"][i] = cls, True

    if scene.map_api is None:
        result["sg_map_available"] = torch.tensor(False)
        return result
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    polygon_layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                      SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA]
    line_layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    pose = frame.ego_status.ego_pose
    radius = math.hypot(max(abs(roi[0]), abs(roi[2])), max(abs(roi[1]), abs(roi[3]))) + 20
    # Errors are propagated: unavailable map data must not become an empty negative label.
    supported = scene.map_api.get_available_map_objects()
    requested = [layer for layer in polygon_layers + line_layers if layer in supported]
    objects = scene.map_api.get_proximal_map_objects(Point2D(*pose[:2]), radius, requested)
    polygons = []
    for layer in polygon_layers:
        if layer in supported:
            polygons.extend(local_geometry(obj.polygon, pose) for obj in objects[layer])
        elif layer == SemanticMapLayer.DRIVABLE_AREA:
            # NuPlanMap exposes the union's source polygons as a vector layer only.
            vectors = scene.map_api._get_vector_map_layer(layer)
            patch = box(pose[0] - radius, pose[1] - radius, pose[0] + radius, pose[1] + radius)
            indices = list(vectors.sindex.intersection(patch.bounds))
            polygons.extend(local_geometry(polygon, pose) for polygon in vectors.iloc[indices].geometry
                            if polygon.intersects(patch))
        else:
            raise ValueError(f"Required geometry map layer unavailable: {layer}")
    if any(layer not in supported for layer in line_layers):
        raise ValueError("Map is missing lane/lane-connector geometry")
    boundaries = vectorize_lines(boundary_lines(polygons), roi, k, cfg.max_segment_length)
    centers = [local_geometry(obj.baseline_path.linestring, pose) for layer in line_layers for obj in objects[layer]]
    centers = vectorize_lines(centers, roi, k, cfg.max_segment_length, directed=True)
    candidates = [(cls, xy) for cls, lines in enumerate((boundaries, centers)) for xy in lines]
    candidates.sort(key=lambda item: (float(np.linalg.norm(item[1], axis=1).min()), item[0], tuple(item[1].ravel())))
    result["sg_map_overflow"] = torch.tensor(max(0, len(candidates) - m))
    for i, (cls, xy) in enumerate(candidates[:m]):
        result["sg_map_geometry"][i] = torch.tensor(normalize_xy(xy, roi).ravel(), dtype=torch.float32)
        result["sg_map_mask"][i] = True
        result["sg_map_class"][i], result["sg_map_valid"][i] = cls, True
    return result
