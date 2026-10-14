from __future__ import annotations

import torch
import torch.nn as nn


def init_memory(
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    num_heads: int = 8,
    head_dim: int = 32,
) -> torch.Tensor:
    return torch.zeros(
        batch_size,
        num_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=dtype,
    )


class LongMemoryRead(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, z_short: torch.Tensor, memory_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # z_short: [B,S,D], memory_prev: [B,H,dh,dh]
        batch, tokens, dim = z_short.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {dim}")
        if memory_prev.shape != (batch, self.num_heads, self.head_dim, self.head_dim):
            raise ValueError(
                f"memory_prev shape {tuple(memory_prev.shape)} != "
                f"[{batch},{self.num_heads},{self.head_dim},{self.head_dim}]"
            )

        x = self.norm(z_short)
        query = self.q_proj(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        read = query @ memory_prev
        read = read.transpose(1, 2).reshape(batch, tokens, dim)
        read = self.out_proj(read)
        return z_short + read, read


class LongFlowMemoryUpdate(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.gamma_proj = nn.Linear(dim, dim)

    def forward(
        self, z_short: torch.Tensor, memory_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, dim = z_short.shape
        x = self.norm(z_short)
        key = self.k_proj(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        delta = (key.transpose(-2, -1) @ value) / float(tokens)

        gamma = torch.sigmoid(self.gamma_proj(x.mean(dim=1)))
        gamma = gamma.view(batch, self.num_heads, self.head_dim, 1)
        memory_new = gamma * memory_prev + delta
        return memory_new, delta, gamma
