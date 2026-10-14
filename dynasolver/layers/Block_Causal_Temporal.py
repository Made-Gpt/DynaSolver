from __future__ import annotations

import math

import torch
import torch.nn as nn

from layers.Temporal_RoPE import TemporalRoPE
from layers.flash_attn_utils import attention_qkv


class BlockCausalTemporalAttention(nn.Module):
    """Frame-level block-causal self-attention over Physics Tokens."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        use_rope: bool = True,
        use_flash_attn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.use_flash_attn = bool(use_flash_attn)
        self.dropout_p = float(dropout)

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rope = TemporalRoPE(self.head_dim) if use_rope else None
        # Identity at init: residual + 0, so spatial path is undisturbed before training.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_win: torch.Tensor) -> torch.Tensor:
        # z_win: [B, T, S, D]
        if z_win.ndim != 4:
            raise ValueError(f"z_win must be [B,T,S,D], got {tuple(z_win.shape)}")
        batch, frames, tokens, dim = z_win.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {dim}")

        residual = z_win
        x = self.norm(z_win).reshape(batch, frames * tokens, dim)

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = query.view(batch, frames * tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, frames * tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, frames * tokens, self.num_heads, self.head_dim).transpose(1, 2)

        frame_id = torch.arange(frames, device=z_win.device)
        time_pos = frame_id.repeat_interleave(tokens)
        if self.rope is not None:
            query = self.rope(query, time_pos)
            key = self.rope(key, time_pos)

        # Block-causal: tokens in frame i see all tokens in frames 0..i (full within-frame).
        # Not the same as token-level causal=True, so FlashAttn causal kernel cannot be used
        # directly; SDPA with a boolean mask still avoids a manual matmul+softmax graph.
        allow = time_pos[None, :] <= time_pos[:, None]  # [L, L]
        attn_mask = allow[None, None, :, :]  # [1,1,L,L] bool, True=keep

        if self.use_flash_attn:
            out = attention_qkv(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                causal=False,
                use_flash=True,
            )
        else:
            score = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            score = score.masked_fill(~attn_mask, float("-inf"))
            attn = torch.softmax(score, dim=-1)
            out = attn @ value

        out = out.transpose(1, 2).reshape(batch, frames, tokens, dim)
        out = self.out_proj(out)
        return residual + out
