import math
from typing import Dict, Optional, Tuple

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


class TimeAlignedFutureAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.future_norm = nn.LayerNorm(d_model)
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

    def forward(
        self, proposal_tokens: torch.Tensor, future_latents: torch.Tensor
    ) -> torch.Tensor:
        batch_size, proposal_num, num_poses, d_model = proposal_tokens.shape
        num_slots = future_latents.shape[2]
        queries = proposal_tokens.permute(0, 2, 1, 3).reshape(
            batch_size * num_poses, proposal_num, d_model
        )
        future = future_latents.reshape(
            batch_size * num_poses, num_slots, d_model
        )
        attended = self.attention(
            self.query_norm(queries),
            self.future_norm(future),
            self.future_norm(future),
            need_weights=False,
        )[0]
        attended = attended + self.ffn(self.ffn_norm(attended))
        return self.attention_dropout(attended).reshape(
            batch_size, num_poses, proposal_num, d_model
        ).permute(0, 2, 1, 3)


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
        self.future_interval = config.get("future_interval", self.pose_interval)
        self.nc_temperature = config.nc_risk_pool_temperature
        self.ttc_temperature = config.ttc_risk_pool_temperature
        if self.nc_temperature <= 0 or self.ttc_temperature <= 0:
            raise ValueError("NC and TTC risk pooling temperatures must be positive.")
        if self.pose_interval <= 0 or self.future_interval <= 0:
            raise ValueError("Trajectory and future sampling intervals must be positive.")
        if self.future_interval != self.pose_interval:
            raise ValueError(
                "Future and proposal timestamps must use the same interval, got "
                f"{self.future_interval} and {self.pose_interval}."
            )

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
        self.future_attention = TimeAlignedFutureAttentionBlock(
            d_model, num_heads, d_ffn, dropout
        )
        self.future_gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.future_gate.weight)
        nn.init.constant_(self.future_gate.bias, -1.0)
        post_fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.post_fusion_temporal_encoder = nn.TransformerEncoder(
            post_fusion_layer,
            num_layers=config.get("future_post_fusion_temporal_layers", 1),
        )
        self.output_norm = nn.LayerNorm(d_model)

        self.clearance_head = self._head(d_model, d_ffn)
        self.nc_timestep_risk_head = self._head(d_model, d_ffn)
        self.nc_head = self._head(d_model, d_ffn)
        self.ttc_head = self._head(d_model, d_ffn)

        self.dac_pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.ddc_pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
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
        self.ego_progress_summary = nn.Linear(2 * d_model, d_model)
        self.comfort_summary = nn.Linear(2 * d_model, d_model)
        self.comfort_kinematics_summary = nn.Linear(4, d_model, bias=False)

    @staticmethod
    def _head(d_model: int, d_ffn: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )

    @staticmethod
    def _risk_aware_feature_pool(
        tokens: torch.Tensor, risk_logits: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(risk_logits / temperature, dim=-1)
        features = torch.einsum("bnt,bntd->bnd", weights, tokens)
        return features, weights

    def _build_pose_features(
        self, proposals: torch.Tensor, current_velocity: torch.Tensor
    ) -> torch.Tensor:
        position = proposals[..., :2]
        yaw = proposals[..., 2:3]

        previous_position = torch.cat(
            (torch.zeros_like(position[..., :1, :]), position[..., :-1, :]), dim=-2
        )
        velocity = (position - previous_position) / self.pose_interval

        initial_velocity = current_velocity[:, None, None, :].expand(
            -1, proposals.shape[1], -1, -1
        )
        previous_velocity = torch.cat(
            (initial_velocity, velocity[..., :-1, :]), dim=-2
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

    @staticmethod
    def _metric_query_pool(
        tokens: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        weights = torch.softmax(
            torch.einsum("bntd,d->bnt", tokens, query)
            / math.sqrt(tokens.shape[-1]),
            dim=-1,
        )
        return torch.einsum("bnt,bntd->bnd", weights, tokens)

    def _comfort_kinematics(self, pose_features: torch.Tensor) -> torch.Tensor:
        acceleration = pose_features[..., 6:8]
        acceleration_norm = torch.linalg.vector_norm(acceleration, dim=-1)
        yaw_rate = pose_features[..., 8].abs()
        previous_acceleration = torch.cat(
            (acceleration[..., :1, :], acceleration[..., :-1, :]), dim=-2
        )
        jerk_norm = torch.linalg.vector_norm(
            (acceleration - previous_acceleration) / self.pose_interval, dim=-1
        )
        return torch.stack(
            (
                acceleration_norm.mean(dim=-1),
                acceleration_norm.amax(dim=-1),
                yaw_rate.amax(dim=-1),
                jerk_norm.amax(dim=-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        proposals: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
        current_velocity: torch.Tensor,
        future_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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
        if current_velocity.shape != (batch_size, 2):
            raise ValueError(
                f"Expected current_velocity shape {(batch_size, 2)}, "
                f"got {tuple(current_velocity.shape)}."
            )

        pose_features = self._build_pose_features(proposals, current_velocity)
        tokens = self.pose_encoder(pose_features)
        tokens = tokens + self.temporal_embedding + ego_token[:, None, :, :]
        tokens = tokens.reshape(batch_size * proposal_num, num_poses, -1)
        tokens = self.temporal_encoder(tokens)
        proposal_tokens = tokens.reshape(batch_size, proposal_num, num_poses, -1)

        scene = scene_features[:, None].expand(-1, proposal_num, -1, -1)
        scene = scene.reshape(batch_size * proposal_num, scene_features.shape[1], -1)
        scene_tokens = tokens
        for layer in self.scene_layers:
            scene_tokens = layer(scene_tokens, scene)

        tokens = scene_tokens.reshape(batch_size, proposal_num, num_poses, -1)
        if future_latents is not None:
            if (
                future_latents.ndim != 4
                or future_latents.shape[0] != batch_size
                or future_latents.shape[-1] != self.d_model
            ):
                raise ValueError(
                    f"Expected future_latents shape (B, Tf, K, {self.d_model}), got "
                    f"{tuple(future_latents.shape)}."
                )
            if future_latents.shape[1] != num_poses:
                raise ValueError(
                    f"Expected {num_poses} future steps aligned at "
                    f"{self.pose_interval}s, got {future_latents.shape[1]}."
                )
            future_delta = self.future_attention(proposal_tokens, future_latents)
            future_gate = torch.sigmoid(
                self.future_gate(torch.cat((tokens, future_delta), dim=-1))
            )
            tokens = tokens + future_gate * future_delta
            tokens = self.post_fusion_temporal_encoder(
                tokens.reshape(batch_size * proposal_num, num_poses, -1)
            ).reshape(batch_size, proposal_num, num_poses, -1)
        tokens = self.output_norm(tokens)
        pred_clearance = self.clearance_head(tokens).squeeze(-1)

        nc_risk = self.nc_timestep_risk_head(tokens).squeeze(-1)
        nc_feature, _ = self._risk_aware_feature_pool(
            tokens, nc_risk, self.nc_temperature
        )
        ttc_feature, _ = self._risk_aware_feature_pool(
            tokens, nc_risk, self.ttc_temperature
        )
        nc_logit = self.nc_head(nc_feature).squeeze(-1)
        ttc_logit = self.ttc_head(ttc_feature).squeeze(-1)

        dac_features = self._metric_query_pool(tokens, self.dac_pool_query)
        ddc_features = self._metric_query_pool(tokens, self.ddc_pool_query)
        pred_logit = {
            "drivable_area_compliance": self.global_heads[
                "drivable_area_compliance"
            ](dac_features).squeeze(-1),
            "driving_direction_compliance": self.global_heads[
                "driving_direction_compliance"
            ](ddc_features).squeeze(-1),
        }
        progress_features = torch.cat((tokens.mean(dim=-2), tokens[..., -1, :]), dim=-1)
        token_mean = tokens.mean(dim=-2)
        peak_deviation = (tokens - token_mean.unsqueeze(-2)).abs().amax(dim=-2)
        comfort_features = self.comfort_summary(
            torch.cat((token_mean, peak_deviation), dim=-1)
        ) + self.comfort_kinematics_summary(
            self._comfort_kinematics(pose_features)
        )
        pred_logit["ego_progress"] = self.global_heads["ego_progress"](
            self.ego_progress_summary(progress_features)
        ).squeeze(-1)
        pred_logit["comfort"] = self.global_heads["comfort"](
            comfort_features
        ).squeeze(-1)
        pred_logit["no_at_fault_collisions"] = nc_logit
        pred_logit["time_to_collision_within_bound"] = ttc_logit
        return pred_logit, pred_clearance, nc_risk