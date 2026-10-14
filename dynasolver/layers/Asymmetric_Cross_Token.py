"""Token-level cross-attention between surface and volume Physics tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.flash_attn_utils import attention_qkv


class TokenCrossAttention(nn.Module):
    """Multi-head cross-attn: Q from one branch, KV from the other. Residual outside."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = float(dropout)
        self.use_flash_attn = bool(use_flash_attn)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_tokens: [B, Sq, D] or [B, T, Sq, D]
            kv_tokens:    [B, Sk, D] or [B, T, Sk, D]
        Returns:
            Same shape as query_tokens (cross output only; caller adds residual).
        """
        if query_tokens.shape[-1] != self.dim or kv_tokens.shape[-1] != self.dim:
            raise ValueError("token dim mismatch")
        if query_tokens.ndim == 4:
            batch, frames, sq, dim = query_tokens.shape
            sk = kv_tokens.shape[2]
            q = query_tokens.reshape(batch * frames, sq, dim)
            kv = kv_tokens.reshape(batch * frames, sk, dim)
            out = self._forward_bt(q, kv)
            return out.reshape(batch, frames, sq, dim)
        if query_tokens.ndim != 3 or kv_tokens.ndim != 3:
            raise ValueError(
                f"expected [B,S,D] or [B,T,S,D], got {tuple(query_tokens.shape)} / "
                f"{tuple(kv_tokens.shape)}"
            )
        return self._forward_bt(query_tokens, kv_tokens)

    def _forward_bt(self, query_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        batch, sq, _ = query_tokens.shape
        sk = kv_tokens.shape[1]
        q = self.q_proj(self.norm_q(query_tokens))
        kv = self.norm_kv(kv_tokens)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        q = q.view(batch, sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, sk, self.num_heads, self.head_dim).transpose(1, 2)
        out = attention_qkv(
            q,
            k,
            v,
            dropout_p=self.dropout_p,
            causal=False,
            use_flash=self.use_flash_attn,
        )
        out = out.transpose(1, 2).reshape(batch, sq, self.dim)
        return self.out_proj(out)


class AsymmetricSurfaceVolumeCross(nn.Module):
    """Frame-local asymmetric coupling: strong S→V, weak V→S with scale_vs.

    When ``share_cross_weights`` is True, both directions reuse one
    ``TokenCrossAttention`` (weight tying). Only one submodule is registered so
    ``state_dict`` / naive param sums do not double-count the tied weights
    (legacy dual-prefix checkpoints still load via ``_load_from_state_dict``).
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
        cross_vs_scale_init: float = 0.0,
        share_cross_weights: bool = False,
    ):
        super().__init__()
        self.share_cross_weights = bool(share_cross_weights)
        self.cross_sv = TokenCrossAttention(
            dim=dim, num_heads=num_heads, dropout=dropout, use_flash_attn=use_flash_attn
        )
        if self.share_cross_weights:
            # Python alias only — not a second nn.Module child.
            object.__setattr__(self, "cross_vs", self.cross_sv)
        else:
            self.cross_vs = TokenCrossAttention(
                dim=dim, num_heads=num_heads, dropout=dropout, use_flash_attn=use_flash_attn
            )
        self.scale_vs = nn.Parameter(torch.tensor(float(cross_vs_scale_init)))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Legacy share=True checkpoints wrote both cross_sv.* and cross_vs.*
        # for the same tied module; drop the duplicate prefix before load.
        if self.share_cross_weights:
            vs_prefix = prefix + "cross_vs."
            sv_prefix = prefix + "cross_sv."
            drop_keys = [k for k in list(state_dict.keys()) if k.startswith(vs_prefix)]
            for key in drop_keys:
                twin = sv_prefix + key[len(vs_prefix) :]
                if twin in state_dict and not torch.equal(state_dict[key], state_dict[twin]):
                    error_msgs.append(
                        f"shared-cross checkpoint mismatch: {key} != {twin}"
                    )
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self, z_s: torch.Tensor, z_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_s, z_v: [B,T,Ss,D] / [B,T,Sv,D] (or [B,S,D])
        Returns:
            Updated (z_s, z_v) with residuals applied.
        """
        # Strong: volume reads surface
        z_v = z_v + self.cross_sv(z_v, z_s)
        # Weak: surface reads volume (same module when share_cross_weights)
        z_s = z_s + self.scale_vs * self.cross_vs(z_s, z_v)
        return z_s, z_v
