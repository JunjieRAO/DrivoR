"""Export inference-only code into NAVSIM-v2 without shadowing its navsim package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from omegaconf import OmegaConf
import yaml


def load_training_agent_config(path):
    # Saved Hydra files also contain large scene lists. Only wrap the agent subtree
    # in OmegaConf; constructing millions of unrelated config nodes is unnecessary.
    with open(path) as stream:
        raw = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return OmegaConf.create({"agent": raw["agent"]}).agent.config


WRAPPER = '''import torch
from omegaconf import OmegaConf
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from .drivor_model import DrivoRModel
from .drivor_features import DrivoRFeatureBuilder

class DrivoRSceneGeometryAgent(AbstractAgent):
    def __init__(self, config, checkpoint_path, trajectory_sampling):
        super().__init__(trajectory_sampling)
        self._config = config
        self._checkpoint_path = checkpoint_path
        self._drivor_model = DrivoRModel(config)

    def name(self):
        return self.__class__.__name__

    def initialize(self):
        if not self._checkpoint_path:
            raise ValueError("Geometry evaluation requires a full trained checkpoint")
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.replace("agent._drivor_model", "_drivor_model"): value for key, value in state.items()}
        self.load_state_dict(state, strict=True)

    def get_sensor_config(self):
        return SensorConfig(**{key: OmegaConf.to_container(self._config[key]) for key in
            ("cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "lidar_pc")})

    def get_feature_builders(self):
        return [DrivoRFeatureBuilder(self._config)]

    def get_target_builders(self):
        return []

    def forward(self, features):
        return self._drivor_model(features)

    def compute_loss(self, *args, **kwargs):
        raise RuntimeError("Inference-only export; train in the DrivoR repository")

    def get_optimizers(self):
        raise RuntimeError("Inference-only export")
'''


def export(destination, mode, training_config=None):
    root = Path(__file__).resolve().parents[1]
    source = root / "navsim/agents/drivoR"
    destination = Path(destination).resolve()
    if not (destination / "navsim/agents/abstract_agent.py").is_file():
        raise ValueError("Destination must be a NAVSIM repository")
    package = Path("navsim/agents/drivor_scene_geometry")
    files = {}
    for name in ("drivor_model.py", "transformer_decoder.py", "timm_layers.py",
                 "score_module/scorer.py", "layers/utils/mlp.py",
                 "layers/image_encoder/dinov2_lora.py", "layers/image_encoder/grid_mask.py",
                 "scene_geometry/geometry_encoder.py"):
        content = (source / name).read_text()
        content = content.replace("from navsim.agents.drivoR.utils import pylogger",
                                  "from navsim.agents.drivor_scene_geometry import logging_utils as pylogger")
        files[package / name] = content
    feature = (source / "drivor_features.py").read_text().split("class DrivoRFeatureBuilder", 1)[1]
    feature = "class DrivoRFeatureBuilder" + feature.split("class DrivoRTargetBuilder", 1)[0]
    files[package / "drivor_features.py"] = '''from typing import Dict
import numpy as np
import torch
from PIL import Image
from navsim.common.dataclasses import AgentInput
from navsim.common.enums import LidarIndex
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder
''' + feature
    files[package / "logging_utils.py"] = "import logging\ndef get_pylogger(name):\n    return logging.getLogger(name)\n"
    files[package / "drivor_agent.py"] = WRAPPER
    for directory in ("", "score_module", "layers", "layers/utils", "layers/image_encoder", "scene_geometry"):
        files[package / directory / "__init__.py"] = '"""DrivoR scene geometry inference export."""\n'
    if training_config:
        config = load_training_agent_config(training_config)
    else:
        base = OmegaConf.load(root / "navsim/planning/script/config/common/agent/drivoR.yaml")
        extra = OmegaConf.load(root / "navsim/planning/script/config/common/agent/drivoR_scene_geometry.yaml")
        config = OmegaConf.merge(base.config, extra.config)
    config.scene_geometry.mode = mode
    config.image_backbone.model_weights = str(root / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")
    # Formal evaluation uses one common ranking rule across experiments.
    for name, value in dict(noc=10., dac=13., ddc=6., ttc=14., ep=15., comfort=2.).items():
        config[name] = value
    config.scene_geometry.cache_root = None
    exported = OmegaConf.create({"_target_": "navsim.agents.drivor_scene_geometry.drivor_agent.DrivoRSceneGeometryAgent",
                                "config": OmegaConf.to_container(config, resolve=False),
                                "checkpoint_path": "", "trajectory_sampling": "${agent.config.trajectory_sampling}"})
    config_path = Path("navsim/planning/script/config/common/agent/drivoR_scene_geometry.yaml")
    files[config_path] = OmegaConf.to_yaml(exported)
    hashes = {str(path): hashlib.sha256(content.encode()).hexdigest() for path, content in files.items()}
    manifest_path = destination / package / "export_manifest.json"
    previous = json.loads(manifest_path.read_text())["files"] if manifest_path.exists() else {}
    for path, content in files.items():
        target = destination / path
        if target.exists():
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual not in (previous.get(str(path)), hashes[str(path)]):
                raise RuntimeError(f"Refusing to overwrite independent changes: {target}")
    for path, content in files.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=target.parent)
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(temp, target)
    manifest = {"mode": mode, "source": str(root), "files": hashes,
                "code_hash": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    parser.add_argument("--mode", choices=["off", "aux", "gen"], default="gen")
    parser.add_argument("--training-config", help="Hydra config.yaml for non-default architecture")
    args = parser.parse_args()
    export(args.destination, args.mode, args.training_config)
