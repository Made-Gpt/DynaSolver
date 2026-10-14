from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_

from models.Transolver import Transolver_block
from layers.Physics_Attention import Physics_Attention_Irregular_Mesh


class _PointToSliceTokens(nn.Module):
    """Transolver-compatible point/slice projection with optional point masking."""

    def __init__(self, hidden: int, heads: int, slice_num: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Physics_Attention_Irregular_Mesh(
            hidden, heads=heads, dim_head=hidden // heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.heads = heads
        self.dim_head = hidden // heads
        self.slice_num = slice_num

    def tokenize(self, points: torch.Tensor, mask: torch.Tensor | None = None):
        weights = self.attn.compute_slice_weights(points)
        if mask is not None:
            if mask.shape != points.shape[:2]:
                raise ValueError(f"point mask must be [B,N], got {tuple(mask.shape)}")
            weights = weights * mask[:, None, :, None].to(weights.dtype)
            if torch.any(weights.sum(dim=(1, 2, 3)) <= 0):
                raise ValueError("each sample must contain at least one valid surface point")
        tokens = self.attn.slice_to_tokens(points, weights)
        tokens = tokens.permute(0, 2, 1, 3).reshape(points.shape[0], self.slice_num, -1)
        token_mass = weights.sum(dim=(1, 2))
        token_valid = token_mass > 1.0e-5
        return tokens, weights, token_valid

    def detokenize(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        b, g, c = tokens.shape
        if g != self.slice_num or c != self.heads * self.dim_head:
            raise ValueError("token shape does not match slice projector")
        head_tokens = tokens.reshape(b, g, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        return self.attn.deslice_to_out(head_tokens, weights)


class _SurfaceCrossBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: int, dropout: float, act: str):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(hidden)
        activation = nn.GELU() if act == "gelu" else nn.SiLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio), activation,
            nn.Linear(hidden * mlp_ratio, hidden), nn.Dropout(dropout),
        )

    def forward(self, volume_tokens, surface_tokens, surface_valid):
        key_padding_mask = ~surface_valid
        q = self.norm_q(volume_tokens)
        kv = self.norm_kv(surface_tokens)
        attended, _ = self.cross(q, kv, kv, key_padding_mask=key_padding_mask)
        volume_tokens = volume_tokens + attended
        volume_tokens = volume_tokens + self.ffn(self.norm_ffn(volume_tokens))
        return volume_tokens


class Model(nn.Module):
    """V1.1 fixed surface-token branch with volume self/self/cross groups."""

    requires_surface_branch = True

    def __init__(self, args):
        super().__init__()
        if args.geotype != "unstructured":
            raise ValueError("DynamicTransolverV1_1 requires unstructured points")
        self.__name__ = "DynamicTransolverV1_1"
        self.args = args
        self.state_dim = int(args.state_dim)
        self.hidden = int(getattr(args, "hidden_dim", args.n_hidden))
        self.heads = int(args.n_heads)
        self.slice_num = int(args.slice_num)
        self.groups = int(getattr(args, "interaction_groups", 3))
        if self.hidden % self.heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.volume_input = nn.Sequential(
            nn.Linear(23, 512), nn.GELU(), nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, self.hidden)
        )
        self.surface_input = nn.Sequential(
            nn.Linear(15, 256), nn.GELU(), nn.Linear(256, 512), nn.GELU(), nn.Linear(512, self.hidden)
        )
        self.volume_blocks = nn.ModuleList([
            Transolver_block(
                num_heads=self.heads, hidden_dim=self.hidden, dropout=float(args.dropout),
                act=args.act, mlp_ratio=int(args.mlp_ratio), last_layer=False,
                out_dim=self.state_dim, slice_num=self.slice_num,
                geotype="unstructured", shapelist=None,
            ) for _ in range(self.groups * 2)
        ])
        self.volume_projector = _PointToSliceTokens(self.hidden, self.heads, self.slice_num, float(args.dropout))
        self.surface_projector = _PointToSliceTokens(self.hidden, self.heads, self.slice_num, float(args.dropout))
        self.cross_blocks = nn.ModuleList([
            _SurfaceCrossBlock(self.hidden, self.heads, int(args.mlp_ratio), float(args.dropout), args.act)
            for _ in range(self.groups)
        ])
        self.placeholder = nn.Parameter(torch.rand(self.hidden) / self.hidden)
        self.output_norm = nn.LayerNorm(self.hidden)
        self.output_proj = nn.Linear(self.hidden, self.state_dim)
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

    @staticmethod
    def _volume_features(query_xyz, static_fx, previous_state, boundary_feat):
        if previous_state.ndim == 4:
            previous_state = previous_state[:, -1]
        normal = static_fx[..., 4:7]
        displacement = boundary_feat[..., 0:1] * normal
        return torch.cat((
            query_xyz, previous_state, static_fx[..., 7:11], boundary_feat[..., 7:8],
            boundary_feat[..., 0:2], displacement, normal, boundary_feat[..., 2:5],
        ), dim=-1), previous_state

    @staticmethod
    def _surface_features(surface_xyz, surface_normal, surface_state, static_fx, boundary_feat):
        inflow = static_fx[:, :1, 7:10].expand(-1, surface_xyz.shape[1], -1)
        speed = static_fx[:, :1, 10:11].expand(-1, surface_xyz.shape[1], -1)
        dt = boundary_feat[:, :1, 7:8].expand(-1, surface_xyz.shape[1], -1)
        return torch.cat((surface_xyz, surface_normal, surface_state, inflow, speed, dt), dim=-1)

    def forward(self, query_xyz, static_fx, previous_state, boundary_feat,
                surface_xyz_normalized, surface_normal_current,
                surface_state_current, surface_valid_mask):
        volume_features, previous = self._volume_features(query_xyz, static_fx, previous_state, boundary_feat)
        if surface_xyz_normalized.ndim != 3 or surface_xyz_normalized.shape[-1] != 3:
            raise ValueError("surface_xyz_normalized must be [B,Ns,3]")
        if surface_valid_mask.dtype != torch.bool:
            surface_valid_mask = surface_valid_mask.bool()
        surface_features = self._surface_features(
            surface_xyz_normalized, surface_normal_current, surface_state_current,
            static_fx, boundary_feat,
        )
        volume_hidden = self.volume_input(volume_features) + self.placeholder[None, None, :]
        surface_hidden = self.surface_input(surface_features)
        surface_tokens, _, surface_token_valid = self.surface_projector.tokenize(surface_hidden, surface_valid_mask)
        for group in range(self.groups):
            volume_hidden = self.volume_blocks[group * 2](volume_hidden)
            volume_hidden = self.volume_blocks[group * 2 + 1](volume_hidden)
            volume_tokens, volume_weights, _ = self.volume_projector.tokenize(volume_hidden)
            volume_tokens = self.cross_blocks[group](volume_tokens, surface_tokens, surface_token_valid)
            volume_hidden = volume_hidden + self.volume_projector.detokenize(volume_tokens, volume_weights)
        return previous + self.output_proj(self.output_norm(volume_hidden))
