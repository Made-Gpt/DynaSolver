"""Optional FlashAttention-2 / PyTorch SDPA helpers for slice and temporal attn."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore
except Exception:  # pragma: no cover
    _flash_attn_func = None


def flash_attn_available() -> bool:
    return _flash_attn_func is not None


def attention_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    use_flash: bool = True,
) -> torch.Tensor:
    """Multi-head attention.

    Args:
        q,k,v: [B, H, L, D]
        attn_mask: optional broadcastable mask; True/finite = keep.
            Bool mask uses True=keep. Float mask is additive (0 / -inf).
        causal: standard lower-triangular causal (token-level).
        use_flash: prefer flash_attn (no custom mask) or SDPA.

    Returns:
        [B, H, L, D]
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be [B,H,L,D], got {tuple(q.shape)}")

    # flash-attn wants [B, L, H, D]; only when no custom mask.
    if (
        use_flash
        and _flash_attn_func is not None
        and q.is_cuda
        and attn_mask is None
        and q.dtype in (torch.float16, torch.bfloat16)
    ):
        out = _flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=dropout_p if q.requires_grad else 0.0,
            causal=causal,
        )
        return out.transpose(1, 2)

    # PyTorch SDPA: often routes to FA2 / mem-efficient kernels.
    if use_flash and hasattr(F, "scaled_dot_product_attention"):
        sdpa_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # SDPA: True = allow when is float? Actually bool True means keep.
                sdpa_mask = attn_mask
            else:
                sdpa_mask = attn_mask
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            is_causal=causal and attn_mask is None,
        )

    # Manual fallback
    scale = q.shape[-1] ** -0.5
    score = (q @ k.transpose(-2, -1)) * scale
    if causal and attn_mask is None:
        L = q.shape[-2]
        causal_m = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        score = score.masked_fill(~causal_m[None, None], float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            score = score.masked_fill(~attn_mask, float("-inf"))
        else:
            score = score + attn_mask
    attn = torch.softmax(score, dim=-1)
    if dropout_p > 0.0 and q.requires_grad:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v
