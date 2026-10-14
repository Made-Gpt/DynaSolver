from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_

from models.Transolver import Transolver_block


class Model(nn.Module):
    """V1 preprocessed-feature model with two fluid-mask channels.

    The normal and displacement intentionally retain the historical V1
    semantics: ``static_fx`` is the next-input-frame tensor supplied by the dataset.
    """

    requires_surface_branch = False

    FEATURE_LAYOUT = {
        "query_xyz_normalized": (0, 3),
        "previous_state": (3, 7),
        "inflow_direction": (7, 10),
        "normalized_speed": (10, 11),
        "normalized_dt": (11, 12),
        "sdf_current_and_delta": (12, 14),
        "displacement_next_normal": (14, 17),
        "next_frame_normal": (17, 20),
        "nearest_boundary_velocity": (20, 23),
        "current_fluid_mask": (23, 24),
        "next_fluid_mask": (24, 25),
    }

    def __init__(self, args):
        super().__init__()
        if args.geotype != "unstructured":
            raise ValueError("DynamicTransolverV1PreMask25 requires unstructured points")
        self.__name__ = "DynamicTransolverV1PreMask25"
        self.args = args
        self.state_dim = int(args.state_dim)
        hidden = int(args.n_hidden)
        self.input_dim = 25
        self.preprocess = nn.Sequential(
            nn.Linear(25, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, hidden),
        )
        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads=args.n_heads,
                hidden_dim=hidden,
                dropout=args.dropout,
                act=args.act,
                mlp_ratio=args.mlp_ratio,
                last_layer=False,
                out_dim=self.state_dim,
                slice_num=args.slice_num,
                geotype=args.geotype,
                shapelist=args.shapelist,
            )
            for _ in range(int(args.n_layers))
        ])
        self.placeholder = nn.Parameter(torch.rand(hidden) / hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, self.state_dim)
        self.apply(self._init_weights)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _features(self, query_xyz, static_fx, previous_state, boundary_feat):
        if previous_state.ndim == 4:
            previous_state = previous_state[:, -1]
        if query_xyz.ndim != 3 or query_xyz.shape[-1] != 3:
            raise ValueError("query_xyz must be [B,N,3]")
        if static_fx.ndim != 3 or static_fx.shape[:2] != query_xyz.shape[:2] or static_fx.shape[-1] < 11:
            raise ValueError("static_fx must be [B,N,11]")
        if previous_state.shape[:2] != query_xyz.shape[:2] or previous_state.shape[-1] != self.state_dim:
            raise ValueError("previous_state has incompatible shape")
        if boundary_feat.ndim != 3 or boundary_feat.shape[:2] != query_xyz.shape[:2] or boundary_feat.shape[-1] < 8:
            raise ValueError("boundary_feat must be [B,N,8]")
        normal = static_fx[..., 4:7]
        displacement = boundary_feat[..., 0:1] * normal
        features = torch.cat((
            query_xyz,
            previous_state,
            static_fx[..., 7:11],
            boundary_feat[..., 7:8],
            boundary_feat[..., 0:2],
            displacement,
            normal,
            boundary_feat[..., 2:5],
            boundary_feat[..., 5:7],
        ), dim=-1)
        if features.shape[-1] != self.input_dim:
            raise RuntimeError(f"packed V1 mask25 feature width is {features.shape[-1]}, expected 25")
        return features

    def forward(self, query_xyz, static_fx, previous_state, boundary_feat, *unused, **kwargs):
        del unused, kwargs
        features = self._features(query_xyz, static_fx, previous_state, boundary_feat)
        hidden = self.preprocess(features) + self.placeholder[None, None, :]
        for block in self.blocks:
            if getattr(self.args, "checkpoint", False) and self.training:
                hidden = checkpoint.checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        previous = previous_state[:, -1] if previous_state.ndim == 4 else previous_state
        return previous + self.output_proj(self.output_norm(hidden))

