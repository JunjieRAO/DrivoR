import torch
from torch import nn


class StructuredSceneEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        options = config.structured_input
        self.modalities = [name for name in ("agents", "map", "traffic_lights")
                           if options.get(f"use_{name}", True)]
        if not self.modalities:
            raise ValueError("Structured input needs at least one enabled modality")
        self.map_points = int(options.get("map_points", 20))
        if self.map_points < 2:
            raise ValueError("Structured map_points must be at least two")
        dimensions = config.tf_d_model
        self.encoders = nn.ModuleDict()
        for name in self.modalities:
            input_size = {"agents": 12, "map": self.map_points * 2, "traffic_lights": 3}[name]
            self.encoders[name] = nn.Sequential(
                nn.Linear(input_size, dimensions), nn.ReLU(), nn.Linear(dimensions, dimensions),
            )
        if "agents" in self.modalities:
            self.agent_types = nn.Embedding(4, dimensions)
        if "map" in self.modalities:
            self.map_types = nn.Embedding(10, dimensions)
        self.register_buffer("agent_scale", torch.tensor(
            [64., 64., 1., 1., 10., 10., 20., 20., 10., 10., 1., 1.],
        ), persistent=False)

    def forward(self, features):
        entities = []
        masks = []
        for name in self.modalities:
            if name == "agents":
                valid = features["structured_agent_valid"]
                values = torch.where(valid[..., None], features["structured_agents"], 0.0)
                types = torch.where(valid, features["structured_agent_types"], 0)
                tokens = self.encoders[name](values / self.agent_scale) + self.agent_types(types)
            elif name == "map":
                valid = features["structured_map_valid"]
                points = torch.where(valid[..., None, None], features["structured_map_points"], 0.0)
                if points.shape[-2] != self.map_points:
                    raise ValueError(f"Expected {self.map_points} ordered samples per structured polyline")
                types = torch.where(valid, features["structured_map_types"], 0)
                tokens = self.encoders[name]((points / 64.0).flatten(-2)) + self.map_types(types)
            else:
                valid = features["structured_light_valid"]
                values = torch.where(valid[..., None], features["structured_lights"], 0.0)
                scale = values.new_tensor([64., 64., 1.])
                tokens = self.encoders[name](values / scale)
            entities.append(torch.where(valid[..., None], tokens, 0.0))
            masks.append(valid)
        return torch.cat(entities, dim=1), torch.cat(masks, dim=1)