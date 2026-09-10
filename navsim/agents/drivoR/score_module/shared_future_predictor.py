import torch
import torch.nn as nn


class SharedFutureLatentPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_future_steps: int = 8,
        num_slots: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ffn: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_future_steps <= 0 or num_slots <= 0:
            raise ValueError("Future steps and latent slots must be positive.")

        self.d_model = d_model
        self.num_future_steps = num_future_steps
        self.num_slots = num_slots
        self.time_queries = nn.Parameter(
            torch.randn(num_future_steps, d_model) * 0.02
        )
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        if scene_features.ndim != 3 or scene_features.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected scene_features shape (B, S, {self.d_model}), got "
                f"{tuple(scene_features.shape)}."
            )

        batch_size = scene_features.shape[0]
        queries = self.time_queries[:, None, :] + self.slot_queries[None, :, :]
        queries = queries.reshape(1, self.num_future_steps * self.num_slots, -1)
        queries = queries.expand(batch_size, -1, -1)
        future_latents = self.decoder(queries, scene_features)
        future_latents = self.output_norm(future_latents)
        return future_latents.reshape(
            batch_size, self.num_future_steps, self.num_slots, self.d_model
        )


class FutureOccupancyDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_slots: int = 8,
        height: int = 128,
        width: int = 128,
        num_occupancy_classes: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Future occupancy dimensions must be divisible by 16.")
        self.d_model = d_model
        self.num_slots = num_slots
        self.height = height
        self.width = width
        self.seed_height = height // 16
        self.seed_width = width // 16
        self.seed_projection = nn.Sequential(
            nn.LayerNorm(num_slots * d_model),
            nn.Linear(
                num_slots * d_model,
                base_channels * self.seed_height * self.seed_width,
            ),
            nn.GELU(),
        )
        blocks = []
        channels = base_channels
        for _ in range(4):
            next_channels = max(channels // 2, 16)
            blocks.extend(
                (
                    nn.ConvTranspose2d(
                        channels, next_channels, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GroupNorm(4, next_channels),
                    nn.GELU(),
                )
            )
            channels = next_channels
        self.upsampler = nn.Sequential(*blocks)
        self.occupancy_head = nn.Conv2d(channels, num_occupancy_classes, 1)
        self.velocity_head = nn.Conv2d(channels, 2, 1)

    def forward(
        self, future_latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            future_latents.ndim != 4
            or future_latents.shape[2] != self.num_slots
            or future_latents.shape[3] != self.d_model
        ):
            raise ValueError(
                f"Expected future_latents shape (B, Tf, {self.num_slots}, "
                f"{self.d_model}), got {tuple(future_latents.shape)}."
            )
        batch_size, num_future_steps = future_latents.shape[:2]
        features = self.seed_projection(
            future_latents.reshape(batch_size * num_future_steps, -1)
        )
        features = features.reshape(
            batch_size * num_future_steps,
            -1,
            self.seed_height,
            self.seed_width,
        )
        features = self.upsampler(features)
        occupancy_logits = self.occupancy_head(features).reshape(
            batch_size, num_future_steps, -1, self.height, self.width
        )
        velocity = self.velocity_head(features).reshape(
            batch_size, num_future_steps, 2, self.height, self.width
        )
        return occupancy_logits, velocity