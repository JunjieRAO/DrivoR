import torch
import torch.nn as nn


class TemporalFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.motion_encoder = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.history_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )
        self.output_projection = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        current_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        motion_embedding = self.motion_encoder(motion).unsqueeze(1)
        query = self.query_norm(current_tokens + motion_embedding)
        history = self.history_norm(history_tokens)
        temporal_features, _ = self.cross_attention(
            query, history, history, need_weights=False
        )
        temporal_features = temporal_features + self.ffn(
            self.ffn_norm(temporal_features)
        )
        return current_tokens + self.output_projection(temporal_features)