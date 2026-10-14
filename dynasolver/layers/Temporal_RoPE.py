from __future__ import annotations

import torch
import torch.nn as nn


class TemporalRoPE(nn.Module):
    """Rotary position encoding applied to Q/K using frame-level time ids."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, time_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, L, dh]
            time_pos: [L] integer/float frame ids shared by tokens in a frame
        """
        if x.ndim != 4:
            raise ValueError(f"expected x shape [B,H,L,dh], got {tuple(x.shape)}")
        if time_pos.ndim != 1 or time_pos.shape[0] != x.shape[2]:
            raise ValueError(
                f"time_pos must have shape [L={x.shape[2]}], got {tuple(time_pos.shape)}"
            )
        dtype = x.dtype
        device = x.device
        t = time_pos.to(device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,d->ld", t, self.inv_freq)
        # [L, dh/2] -> cos/sin broadcast over B,H
        cos = torch.cos(freqs).to(dtype=dtype)[None, None, :, :]
        sin = torch.sin(freqs).to(dtype=dtype)[None, None, :, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.flatten(-2)
