import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class SceneCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.scene_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, scene_features: torch.Tensor) -> torch.Tensor:
        attended = self.attention(
            self.query_norm(tokens),
            self.scene_norm(scene_features),
            self.scene_norm(scene_features),
            need_weights=False,
        )[0]
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.ffn(self.ffn_norm(tokens))


class TemporalRiskScorer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        unsupported = [
            name
            for name in ("double_score", "agent_pred", "area_pred", "bev_map", "bev_agent")
            if config.get(name, False)
        ]
        if unsupported:
            raise ValueError(
                "TemporalRiskScorer does not support enabled legacy heads: "
                + ", ".join(unsupported)
            )

        d_model = config.tf_d_model
        d_ffn = config.tf_d_ffn
        num_heads = config.temporal_scorer_num_heads
        dropout = config.temporal_scorer_dropout

        self.num_poses = config.num_poses
        self.d_model = d_model
        self.pose_interval = config.trajectory_sampling.interval_length
        self.nc_temperature = config.nc_risk_pool_temperature
        self.ttc_temperature = config.ttc_risk_pool_temperature
        if self.nc_temperature <= 0 or self.ttc_temperature <= 0:
            raise ValueError("NC and TTC risk pooling temperatures must be positive.")
        if self.pose_interval <= 0:
            raise ValueError("trajectory sampling interval must be positive.")

        self.pose_encoder = nn.Sequential(
            nn.Linear(9, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        position = torch.arange(self.num_poses, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        temporal_embedding = torch.zeros(1, 1, self.num_poses, d_model)
        temporal_embedding[0, 0, :, 0::2] = torch.sin(position * div_term)
        temporal_embedding[0, 0, :, 1::2] = torch.cos(position * div_term)
        self.temporal_embedding = nn.Parameter(temporal_embedding)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=config.temporal_scorer_num_layers
        )
        self.scene_layers = nn.ModuleList(
            SceneCrossAttentionBlock(d_model, num_heads, d_ffn, dropout)
            for _ in range(config.scene_scorer_num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)

        self.clearance_head = self._head(d_model, d_ffn)
        self.nc_timestep_risk_head = self._head(d_model, d_ffn)
        self.ttc_timestep_risk_head = self._head(d_model, d_ffn)
        self.nc_head = nn.Linear(1, 1)
        self.ttc_head = nn.Linear(1, 1)

        self.global_pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.global_heads = nn.ModuleDict(
            {
                name: self._head(d_model, d_ffn)
                for name in (
                    "drivable_area_compliance",
                    "ego_progress",
                    "driving_direction_compliance",
                    "comfort",
                )
            }
        )

    @staticmethod
    def _head(d_model: int, d_ffn: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )

    @staticmethod
    def _risk_pool(risk: torch.Tensor, temperature: float) -> torch.Tensor:
        return temperature * (
            torch.logsumexp(risk / temperature, dim=-1) - math.log(risk.shape[-1])
        )

    def _build_pose_features(self, proposals: torch.Tensor) -> torch.Tensor:
        position = proposals[..., :2]
        yaw = proposals[..., 2:3]

        previous_position = torch.cat(
            (torch.zeros_like(position[..., :1, :]), position[..., :-1, :]), dim=-2
        )
        velocity = (position - previous_position) / self.pose_interval

        previous_velocity = torch.cat(
            (velocity[..., :1, :], velocity[..., :-1, :]), dim=-2
        )
        acceleration = (velocity - previous_velocity) / self.pose_interval

        previous_yaw = torch.cat(
            (torch.zeros_like(yaw[..., :1, :]), yaw[..., :-1, :]), dim=-2
        )
        yaw_delta = yaw - previous_yaw
        yaw_rate = torch.atan2(yaw_delta.sin(), yaw_delta.cos()) / self.pose_interval

        return torch.cat(
            (
                position,
                yaw.sin(),
                yaw.cos(),
                velocity,
                acceleration,
                yaw_rate,
            ),
            dim=-1,
        )

    def forward(
        self,
        proposals: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_size, proposal_num, num_poses, _ = proposals.shape
        if num_poses != self.num_poses:
            raise ValueError(f"Expected {self.num_poses} poses, got {num_poses}.")
        if (
            scene_features.ndim != 3
            or scene_features.shape[0] != batch_size
            or scene_features.shape[-1] != self.d_model
        ):
            raise ValueError(
                f"Expected scene_features shape (B, S, {self.d_model}) with batch "
                f"size {batch_size}, got {tuple(scene_features.shape)}."
            )
        if ego_token.shape != (batch_size, 1, self.d_model):
            raise ValueError(
                f"Expected ego_token shape {(batch_size, 1, self.d_model)}, "
                f"got {tuple(ego_token.shape)}."
            )

        pose_features = self._build_pose_features(proposals)
        tokens = self.pose_encoder(pose_features)
        tokens = tokens + self.temporal_embedding + ego_token[:, None, :, :]
        tokens = tokens.reshape(batch_size * proposal_num, num_poses, -1)
        tokens = self.temporal_encoder(tokens)

        scene = scene_features[:, None].expand(-1, proposal_num, -1, -1)
        scene = scene.reshape(batch_size * proposal_num, scene_features.shape[1], -1)
        for layer in self.scene_layers:
            tokens = layer(tokens, scene)

        tokens = tokens.reshape(batch_size, proposal_num, num_poses, -1)
        tokens = self.output_norm(tokens)
        pred_clearance = self.clearance_head(tokens).squeeze(-1)

        nc_risk = self.nc_timestep_risk_head(tokens).squeeze(-1)
        ttc_risk = self.ttc_timestep_risk_head(tokens).squeeze(-1)
        nc_logit = self.nc_head(
            self._risk_pool(nc_risk, self.nc_temperature).unsqueeze(-1)
        ).squeeze(-1)
        ttc_logit = self.ttc_head(
            self._risk_pool(ttc_risk, self.ttc_temperature).unsqueeze(-1)
        ).squeeze(-1)

        pool_weights = torch.softmax(
            torch.einsum("bntd,d->bnt", tokens, self.global_pool_query)
            / math.sqrt(tokens.shape[-1]),
            dim=-1,
        )
        global_features = torch.einsum("bnt,bntd->bnd", pool_weights, tokens)

        pred_logit = {
            name: head(global_features).squeeze(-1)
            for name, head in self.global_heads.items()
        }
        pred_logit["no_at_fault_collisions"] = nc_logit
        pred_logit["time_to_collision_within_bound"] = ttc_logit
        return pred_logit, pred_clearance