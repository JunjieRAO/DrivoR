import torch
import torch.nn as nn
import numpy as np
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from .layers.utils.mlp import MLP
from .score_module.spatial_sampler import LocalPatchSampler
from .timm_layers import (
    Mlp,
    DropPath,
    LayerScale,
)


class Attention(torch.nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.proj_drop = torch.nn.Dropout(proj_drop)

        self.attn = torch.nn.MultiheadAttention(dim, num_heads, 
                                                dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=True, device=None, dtype=None)


    def forward(self, q: torch.Tensor, 
                kv:Optional[torch.Tensor]=None) -> torch.Tensor:
        if kv is None:
            x = self.attn(query=q, key=q, value=q, need_weights=False)[0]
        else:
            x = self.attn(query=q, key=kv, value=kv, need_weights=False)[0]
        x = self.proj_drop(x)
        return x

class Block(torch.nn.Module):


    def __init__(self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            drop_path: float = 0.,
            init_values: float = 0.0,
            act_layer: Type[torch.nn.Module] = torch.nn.GELU,
            norm_layer: Type[torch.nn.Module] = torch.nn.LayerNorm,
            mlp_layer: Type[torch.nn.Module] = Mlp,):
        super().__init__()

        # self attention layer
        self.self_attn_norm = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            proj_drop=proj_drop,
        )
        self.self_attn_ls = LayerScale(dim, init_values=init_values) if (init_values > 0) else torch.nn.Identity()
        self.self_attn_drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

        # cross attention network
        self.cross_attn_norm_kv = norm_layer(dim)
        self.cross_attn_norm_q = norm_layer(dim)
        self.cross_attn = Attention(
            dim,
            num_heads=num_heads,
            proj_drop=proj_drop,
        )
        self.cross_attn_ls = LayerScale(dim, init_values=init_values) if (init_values > 0) else torch.nn.Identity()
        self.cross_attn_drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

        # create the FFN network
        self.mlp_norm = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            )
        self.mlp_ls = LayerScale(dim, init_values=init_values) if init_values else torch.nn.Identity()
        self.mlp_drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()



    def forward(self, 
                x: torch.Tensor,
                x_cross: torch.Tensor) -> torch.Tensor:

        x = x + self.self_attn_drop_path(
                        self.self_attn_ls(
                            self.self_attn(
                                self.self_attn_norm(x))))
        
        x = x + self.cross_attn_drop_path(
                self.cross_attn_ls(
                    self.cross_attn(
                        self.cross_attn_norm_q(x), 
                        self.cross_attn_norm_kv(x_cross))))

        x = x + self.mlp_drop_path(
                        self.mlp_ls(
                            self.mlp(
                                self.mlp_norm(x))))
        
        return x

class TransformerDecoder(torch.nn.Module):

    def __init__(self, proj_drop, drop_path, config):
        super().__init__()

        num_layers = config.ref_num
        d_model = config.tf_d_model

        _layers = []
        for i in range(num_layers):
            _layers.append(
                Block(
                    dim=d_model,
                    num_heads=config.refiner_num_heads if hasattr(config, "refiner_num_heads") else 1,
                    init_values= config.refiner_ls_values if hasattr(config, "refiner_ls_values") else 0.0,
                    proj_drop=proj_drop,
                    drop_path=drop_path
                )
            )
        self.layers = torch.nn.ModuleList(_layers)
        self.return_intermediate = True


    def forward(self, x, x_cross):
        
        intermediate = []
        for _, layer in enumerate(self.layers):
            x = layer(x, x_cross)
            if self.return_intermediate:
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return x


class TransformerDecoderScorer(torch.nn.Module):

    def __init__(self, num_layers, d_model, proj_drop, drop_path, config):
        super().__init__()

        _layers = []
        for i in range(num_layers):
            _layers.append(
                Block(
                    dim=d_model,
                    num_heads=config.refiner_num_heads if hasattr(config, "refiner_num_heads") else 1,
                    init_values= config.refiner_ls_values if hasattr(config, "refiner_ls_values") else 0.0,
                    proj_drop=proj_drop,
                    drop_path=drop_path
                )
            )
        self.layers = torch.nn.ModuleList(_layers)
        self.return_intermediate = False

        self.use_local_patch_branch = getattr(config, "use_local_patch_branch", False)
        self.local_fusion_alpha = getattr(config, "local_fusion_alpha", 1.0)

        if self.use_local_patch_branch:
            num_heads = getattr(config, "spatial_patch_num_heads", 4)
            num_poses = getattr(config, "num_poses", 8)
            sigma = getattr(config, "spatial_patch_sigma", 0.15)
            image_size = tuple(getattr(config, "image_size", (1148, 672)))

            self.local_patch_sampler = LocalPatchSampler(
                d_model=d_model,
                num_heads=num_heads,
                num_poses=num_poses,
                sigma=sigma,
                image_size=image_size,
            )
            self.local_fusion_proj = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.local_fusion_proj.weight)
            if self.local_fusion_proj.bias is not None:
                nn.init.zeros_(self.local_fusion_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        proposals: Optional[torch.Tensor] = None,
        patch_features: Optional[torch.Tensor] = None,
        cam_K: Optional[torch.Tensor] = None,
        world_2_cam: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        intermediate = []
        h_global = x
        for _, layer in enumerate(self.layers):
            h_global = layer(h_global, x_cross)
            if self.return_intermediate:
                intermediate.append(h_global)

        if self.return_intermediate:
            h_global_out = torch.stack(intermediate)
        else:
            h_global_out = h_global

        if (
            self.use_local_patch_branch
            and proposals is not None
            and patch_features is not None
            and cam_K is not None
            and world_2_cam is not None
        ):
            h_local = self.local_patch_sampler(x, proposals, patch_features, cam_K, world_2_cam)
            h_fused = h_global_out + self.local_fusion_alpha * self.local_fusion_proj(h_local)
            return h_fused

        return h_global_out



