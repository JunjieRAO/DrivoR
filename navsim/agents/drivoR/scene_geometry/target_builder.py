"""Independent, versioned geometry cache. Never overwrites drivor_target.gz."""
import gzip
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder
from .geometry_targets import build_geometry_targets, SCHEMA_VERSION, AGENT_CLASSES


class SceneGeometryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        self.cfg = config.scene_geometry
        self.spec = {k: list(self.cfg[k]) if k == "roi" else self.cfg[k] for k in (
            "roi", "num_agent_queries", "num_map_queries", "map_points", "max_segment_length", "data_version")}
        self.spec.update(schema=SCHEMA_VERSION, classes=AGENT_CLASSES,
                         source_sha256=hashlib.sha256(Path(__file__).with_name("geometry_targets.py").read_bytes()).hexdigest())
        self.spec_hash = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()[:16]
        self.root = Path(self.cfg.cache_root) / self.spec_hash if self.cfg.cache_root else None
        if self.cfg.cache_mode not in ("strict", "on_the_fly"):
            raise ValueError("geometry cache_mode must be strict or on_the_fly")
        if self.cfg.cache_mode == "strict" and self.root is None:
            raise ValueError("Strict scene geometry requires cache_root")

    def get_unique_name(self):
        return "scene_geometry_" + self.spec_hash

    def cache_path(self, log_name, token):
        if self.root is None:
            raise ValueError("cache_root is required to read/write geometry caches")
        return self.root / log_name / (token + ".gz")

    def load_cached(self, log_name, token):
        path = self.cache_path(log_name, token)
        if not path.is_file():
            raise FileNotFoundError(f"Missing geometry cache {path}; run run_scene_geometry_caching.py")
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        if payload["spec_hash"] != self.spec_hash or payload["token"] != token or payload["log_name"] != log_name:
            raise ValueError(f"Geometry cache identity mismatch: {path}")
        return payload["targets"]

    def compute_targets(self, scene):
        metadata = scene.scene_metadata
        if self.cfg.cache_mode == "strict":
            return self.load_cached(metadata.log_name, metadata.initial_token)
        return build_geometry_targets(scene, self.cfg)

    def cache_scene(self, scene):
        metadata = scene.scene_metadata
        path = self.cache_path(metadata.log_name, metadata.initial_token)
        targets = build_geometry_targets(scene, self.cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"spec_hash": self.spec_hash, "token": metadata.initial_token,
                   "log_name": metadata.log_name, "targets": targets}
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1) as stream:
                pickle.dump(payload, stream)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return targets
