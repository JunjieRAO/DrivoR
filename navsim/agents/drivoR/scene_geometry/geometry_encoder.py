"""SparseDrive-inspired query refinement on DrivoR scene tokens (no temporal bank)."""
import math
import torch
from torch import nn


def geometry_mode(config):
    mode = config.get("scene_geometry", {}).get("mode", "off")
    if mode not in ("off", "aux", "gen", "gen_scorer"):
        raise ValueError(f"Unknown scene_geometry.mode: {mode}")
    return mode


class QueryBranch(nn.Module):
    def __init__(self, count, classes, geometry_dim, cfg, dim, ffn):
        super().__init__()
        self.content = nn.Embedding(count, dim)
        self.reference = nn.Parameter(torch.zeros(count, geometry_dim))
        nn.init.normal_(self.reference, std=0.1)
        self.position = nn.Sequential(nn.Linear(geometry_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(dim, cfg.num_heads, ffn, cfg.dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(cfg.num_layers)])
        self.classifiers = nn.ModuleList([nn.Linear(dim, classes) for _ in self.layers])
        self.regressors = nn.ModuleList([nn.Linear(dim, geometry_dim) for _ in self.layers])
        for head in self.classifiers:
            nn.init.constant_(head.bias, -math.log(99))

    def forward(self, scene):
        content = self.content.weight[None].expand(scene.shape[0], -1, -1)
        reference = self.reference[None].expand(scene.shape[0], -1, -1)
        outputs = []
        for layer, cls, reg in zip(self.layers, self.classifiers, self.regressors):
            content = layer(content + self.position(reference), scene)
            reference = reference + reg(content)
            outputs.append({"tokens": content, "geometry": reference, "logits": cls(content)})
        return outputs


class SceneGeometryEncoder(nn.Module):
    def __init__(self, config, scene_token_count):
        super().__init__()
        cfg = config.scene_geometry
        dim = config.tf_d_model
        # Scene slot identity is branch-local; the original scorer memory is untouched.
        self.scene_position = nn.Parameter(torch.zeros(1, scene_token_count, dim))
        nn.init.normal_(self.scene_position, std=0.01)
        self.agent = QueryBranch(cfg.num_agent_queries, 4, 6, cfg, dim, config.tf_d_ffn)
        self.map = QueryBranch(cfg.num_map_queries, 2, cfg.map_points * 2, cfg, dim, config.tf_d_ffn)

    def forward(self, scene):
        memory = scene + self.scene_position
        return {"agent": self.agent(memory), "map": self.map(memory)}


class GeometryAttention(nn.Module):
    def __init__(self, dim, geometry_dim, heads, confidence_floor):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.position = nn.Sequential(nn.Linear(geometry_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.null = nn.Parameter(torch.zeros(1, 1, dim))
        self.gate = nn.Parameter(torch.zeros(()))
        self.heads = heads
        self.confidence_floor = confidence_floor

    def forward(self, query, prediction):
        # Geometry/confidence are read-only to the planning branch, semantic values are not.
        value = self.value_norm(prediction["tokens"])
        key = value + self.position(prediction["geometry"].detach())
        null = self.null.expand(query.shape[0], -1, -1)
        key, value = torch.cat((key, null), 1), torch.cat((value, null), 1)
        confidence = prediction["logits"].detach().sigmoid().amax(-1).clamp_min(self.confidence_floor)
        bias = torch.cat((confidence.log(), confidence.new_zeros((query.shape[0], 1))), -1)
        bias = bias[:, None, None].expand(-1, self.heads, query.shape[1], -1)
        bias = bias.reshape(query.shape[0] * self.heads, query.shape[1], -1).to(query.dtype)
        update = self.attn(self.query_norm(query), key, value, attn_mask=bias, need_weights=False)[0]
        return query + self.gate.tanh() * update


class GeometryAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config.scene_geometry
        self.agent = GeometryAttention(config.tf_d_model, 6, cfg.num_heads, cfg.confidence_floor)
        self.map = GeometryAttention(config.tf_d_model, cfg.map_points * 2, cfg.num_heads, cfg.confidence_floor)

    def forward(self, query, geometry):
        query = self.agent(query, geometry["agent"][-1])
        return self.map(query, geometry["map"][-1])
