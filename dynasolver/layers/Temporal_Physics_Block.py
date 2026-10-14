from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from einops import rearrange

from layers.Asymmetric_Cross_Token import AsymmetricSurfaceVolumeCross
from layers.Block_Causal_Temporal import BlockCausalTemporalAttention
from layers.Long_Flow_Memory import LongFlowMemoryUpdate, LongMemoryRead
from layers.Physics_Attention import Physics_Attention_Irregular_Mesh
from models.Transolver import MLP


_BRANCH_MODES = frozenset(
    {
        "unified",
        "surface_volume_asym",
        "surface_volume_latent",
        "surface_volume_per_latent",
    }
)


class HistSpatialCache:
    """Legacy cache API retained for callers; integrated temporal path is window-native
    and does not populate this store (prefer one-shot slice→spatial→causal→deslice).
    """

    def __init__(self, window_frames: int, n_layers: int) -> None:
        if window_frames < 1:
            raise ValueError(f"window_frames must be >= 1, got {window_frames}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.window_frames = int(window_frames)
        self.n_layers = int(n_layers)
        self.max_hist_frames = max(0, self.window_frames - 1)
        self.max_entries = max(1, self.n_layers * max(self.max_hist_frames, 1))
        self._store: dict[tuple[int, int], torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._store)

    def get_geopt(
        self,
        layer_idx: int,
        frame_idx: int,
        hidden: torch.Tensor,
        block_fn,
        *,
        detach: bool = True,
    ) -> torch.Tensor:
        if not (0 <= layer_idx < self.n_layers):
            raise IndexError(f"layer_idx={layer_idx} out of range [0, {self.n_layers})")
        if not (0 <= frame_idx < self.window_frames):
            raise IndexError(
                f"frame_idx={frame_idx} out of window range [0, {self.window_frames})"
            )
        key = (layer_idx, frame_idx)
        if key not in self._store:
            if len(self._store) >= self.max_entries:
                raise RuntimeError(
                    f"HistSpatialCache exceeded bound {self.max_entries} "
                    f"(window_frames={self.window_frames}, n_layers={self.n_layers})"
                )
            if detach:
                with torch.no_grad():
                    value = block_fn(hidden)
                self._store[key] = value.detach()
            else:
                self._store[key] = block_fn(hidden)
        return self._store[key]


class CrossStepHistCache:
    """Reuse post-spatial slice tokens by input-seq id across PF/AR steps.

    Stores ``(tokens [B,S,D], weights [H,N,S])`` after spatial residual, before causal.

    Safe when weights are frozen across the cached steps (eval AR, or train
    pushforward_last_only / no_grad intermediate PF steps). Not for ar_stream
    (per-step optimizer updates invalidate cached features).
    """

    def __init__(self) -> None:
        self._geopt: dict[tuple[int, int], torch.Tensor] = {}
        self._slice: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.geopt_hits = 0
        self.geopt_misses = 0
        self.slice_hits = 0
        self.slice_misses = 0

    def clear(self) -> None:
        self._geopt.clear()
        self._slice.clear()
        self.geopt_hits = 0
        self.geopt_misses = 0
        self.slice_hits = 0
        self.slice_misses = 0

    def get_geopt(self, layer_idx: int, seq_id: int) -> torch.Tensor | None:
        key = (int(layer_idx), int(seq_id))
        value = self._geopt.get(key)
        if value is None:
            self.geopt_misses += 1
            return None
        self.geopt_hits += 1
        return value

    def put_geopt(self, layer_idx: int, seq_id: int, value: torch.Tensor) -> None:
        self._geopt[(int(layer_idx), int(seq_id))] = value.detach()

    def get_slice(
        self, layer_idx: int, seq_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        key = (int(layer_idx), int(seq_id))
        value = self._slice.get(key)
        if value is None:
            self.slice_misses += 1
            return None
        self.slice_hits += 1
        return value

    def put_slice(
        self,
        layer_idx: int,
        seq_id: int,
        tokens: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        self._slice[(int(layer_idx), int(seq_id))] = (tokens.detach(), weights.detach())


class TemporalPhysicsBlock(nn.Module):
    """Figure-aligned Pre-LN block with integrated temporal Physics-Attention.

    Outer::
        LN → PhysicsAttn → + → LN → FFN → +

    PhysicsAttn (temporal enabled)::
        linear/slice → transolver token attn → block-causal → deslice → linear

    Spatial token path matches GeoPT ``Attn.forward`` (spa only, no ``z0+spa``);
    ``BlockCausalTemporalAttention`` supplies the temporal residual.

    ``spatial_only`` keeps the GeoPT ``Transolver_block`` path (full Attn.forward, no causal).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        slice_num: int = 32,
        dropout: float = 0.0,
        act: str = "gelu",
        mlp_ratio: int = 2,
        geotype: str = "unstructured",
        shapelist=None,
        short_scale_init: float = 1.0e-3,
        long_scale_init: float = 1.0e-3,
        use_rope: bool = True,
        use_long_memory: bool = True,
        use_short_window: bool = True,
        use_temporal_scale_gate: bool = False,
        slice_mode: str = "legacy",
        use_flash_attn: bool = False,
        branch_mode: str = "unified",
        cross_vs_scale_init: float = 0.0,
        share_cross_weights: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        branch_mode = str(branch_mode or "unified")
        if branch_mode not in _BRANCH_MODES:
            raise ValueError(f"branch_mode={branch_mode!r}; choose from {sorted(_BRANCH_MODES)}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.slice_num = slice_num
        self.head_dim = hidden_dim // num_heads
        self.use_long_memory = use_long_memory
        self.use_short_window = use_short_window
        self.use_temporal_scale_gate = bool(use_temporal_scale_gate)
        self.slice_mode = slice_mode
        self.use_flash_attn = bool(use_flash_attn)
        self.branch_mode = branch_mode
        self.use_sv_asym = branch_mode == "surface_volume_asym"
        self.use_sv_latent = branch_mode in (
            "surface_volume_latent",
            "surface_volume_per_latent",
        )
        self.share_cross_weights = bool(share_cross_weights)

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=self.head_dim,
            dropout=dropout,
            slice_num=slice_num,
            shapelist=shapelist,
            slice_mode=slice_mode,
            use_flash_attn=self.use_flash_attn,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )

        self.temporal_attn = (
            BlockCausalTemporalAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                use_rope=use_rope,
                use_flash_attn=self.use_flash_attn,
                dropout=dropout,
            )
            if use_short_window
            else None
        )
        self.sv_cross = (
            AsymmetricSurfaceVolumeCross(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_flash_attn=self.use_flash_attn,
                cross_vs_scale_init=cross_vs_scale_init,
                share_cross_weights=self.share_cross_weights,
            )
            if (self.use_sv_asym or self.use_sv_latent)
            else None
        )
        self.long_read = LongMemoryRead(dim=hidden_dim, num_heads=num_heads) if use_long_memory else None
        self.memory_update = (
            LongFlowMemoryUpdate(dim=hidden_dim, num_heads=num_heads) if use_long_memory else None
        )
        self.short_scale = nn.Parameter(torch.tensor(float(short_scale_init)))
        self.long_scale = nn.Parameter(torch.tensor(float(long_scale_init)))
        if not self.use_temporal_scale_gate:
            self.short_scale.requires_grad_(False)
            self.long_scale.requires_grad_(False)

    @property
    def spatial_only(self) -> bool:
        """True when both temporal modules are disabled (spatial-only ablation)."""
        return not self.use_short_window and not self.use_long_memory

    def _forward_geopt_block(self, hidden: torch.Tensor) -> torch.Tensor:
        """Match GeoPT ``Transolver_block`` point-level residuals on [B,N,D]."""
        fx = self.Attn(self.ln_1(hidden)) + hidden
        return self.mlp(self.ln_2(fx)) + fx

    def _pool_slice_tokens(self, point_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice-pool points → merged tokens ``[B,S,D]`` and weights ``[B,H,N,S]``."""
        attn = self.Attn
        slice_weights = attn.compute_slice_weights(point_hidden)
        slice_token = attn.slice_to_tokens(point_hidden, slice_weights)
        merged = rearrange(slice_token, "b h g d -> b g (h d)")
        return merged, slice_weights

    def _pool_slice_tokens_masked(
        self, point_hidden: torch.Tensor, point_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool using only points where ``point_mask`` is True (shared Attn weights)."""
        if point_mask.shape != point_hidden.shape[:2]:
            raise ValueError(
                f"point_mask must be [B,N], got {tuple(point_mask.shape)} vs "
                f"points {tuple(point_hidden.shape[:2])}"
            )
        attn = self.Attn
        slice_weights = attn.compute_slice_weights(point_hidden)  # B H N G
        mask_f = point_mask.to(dtype=slice_weights.dtype)[:, None, :, None]
        slice_weights = slice_weights * mask_f
        denom = slice_weights.sum(dim=2, keepdim=True).clamp_min(1.0e-5)
        slice_weights = slice_weights / denom
        empty = point_mask.sum(dim=1) <= 0
        if bool(empty.any()):
            raise RuntimeError(
                "surface/volume branch has empty point set for some batch items; "
                f"empty_batch_idx={torch.nonzero(empty, as_tuple=False).flatten().tolist()}"
            )
        slice_token = attn.slice_to_tokens(point_hidden, slice_weights)
        merged = rearrange(slice_token, "b h g d -> b g (h d)")
        return merged, slice_weights

    def _spatial_attend_merged(self, merged: torch.Tensor) -> torch.Tensor:
        """Transolver token attn on merged ``[B,S,D]`` (GeoPT Attn semantics: spa only).

        Do **not** add slice tokens ``z0`` before deslice: ``deslice(z0+spa)`` injects an
        extra linear bypass that breaks zero-causal ≡ GeoPT ``Transolver_block`` and
        hurt ep1 single-frame loss vs the old (working) casual_tf4.
        Causal keeps its own residual inside ``BlockCausalTemporalAttention``.
        """
        tok = rearrange(
            merged, "b g (h d) -> b h g d", h=self.num_heads, d=self.head_dim
        )
        tok = self.Attn.spatial_attend_tokens(tok)
        return rearrange(tok, "b h g d -> b g (h d)")

    # Back-compat alias for callers/tests that still use the old name.
    def _spatial_residual_on_merged(self, merged: torch.Tensor) -> torch.Tensor:
        return self._spatial_attend_merged(merged)

    def _deslice_merged(
        self, merged: torch.Tensor, slice_weights: torch.Tensor
    ) -> torch.Tensor:
        """Deslice merged ``[B,S,D]`` tokens with full ``to_out`` (bias + dropout)."""
        tok = rearrange(
            merged, "b g (h d) -> b h g d", h=self.num_heads, d=self.head_dim
        )
        return self.Attn.deslice_to_out(tok, slice_weights, use_bias=True, use_dropout=True)

    def _apply_chunk_detach(
        self, z_win: torch.Tensor, trainable_chunk: int | None
    ) -> torch.Tensor:
        if trainable_chunk is None:
            return z_win
        frames = z_win.shape[1]
        chunk = max(1, min(int(trainable_chunk), frames))
        trainable_start = frames - chunk
        if trainable_start <= 0:
            return z_win
        return torch.cat(
            [z_win[:, :trainable_start].detach(), z_win[:, trainable_start:]], dim=1
        )

    def _run_causal(self, z_for_attn: torch.Tensor) -> torch.Tensor:
        if self.temporal_attn is None:
            return z_for_attn
        attn_out = self.temporal_attn(z_for_attn)
        if self.use_temporal_scale_gate:
            return z_for_attn + self.short_scale * (attn_out - z_for_attn)
        return attn_out

    def _slice_spatial_window(
        self,
        x_n: torch.Tensor,
        *,
        layer_idx: int = 0,
        frame_ids: Sequence[int] | None = None,
        cross_step_cache: CrossStepHistCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LN'd points ``[B,T,N,D]`` → post-spatial-residual tokens + weights.

        Returns:
            z_win: ``[B,T,S,D]``
            weights_win: ``[B,T,H,N,S]``
        """
        batch, frames, num_points, dim = x_n.shape
        if cross_step_cache is None:
            flat = x_n.reshape(batch * frames, num_points, dim)
            merged, weights = self._pool_slice_tokens(flat)
            merged = self._spatial_attend_merged(merged)
            z_win = merged.reshape(batch, frames, self.slice_num, dim)
            weights_win = weights.reshape(
                batch, frames, self.num_heads, num_points, self.slice_num
            )
            return z_win, weights_win

        if frame_ids is not None and len(frame_ids) != frames:
            raise ValueError(
                f"frame_ids length {len(frame_ids)} != window frames {frames}"
            )
        token_frames: list[torch.Tensor] = []
        weight_frames: list[torch.Tensor] = []
        for frame_idx in range(frames):
            gid = int(frame_ids[frame_idx]) if frame_ids is not None else None
            if gid is not None:
                hit = cross_step_cache.get_slice(layer_idx, gid)
                if hit is not None:
                    tokens, weights = hit
                    token_frames.append(tokens)
                    weight_frames.append(weights)
                    continue
            merged, weights = self._pool_slice_tokens(x_n[:, frame_idx])
            merged = self._spatial_attend_merged(merged)
            if gid is not None:
                cross_step_cache.put_slice(layer_idx, gid, merged, weights)
            token_frames.append(merged)
            weight_frames.append(weights)
        return torch.stack(token_frames, dim=1), torch.stack(weight_frames, dim=1)

    def _ffn_residual_window(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, num_points, dim = x.shape
        flat = x.reshape(batch * frames, num_points, dim)
        return (flat + self.mlp(self.ln_2(flat))).reshape(batch, frames, num_points, dim)

    def forward(
        self,
        hidden_win: torch.Tensor,
        memory_prev: torch.Tensor | None,
        layer_idx: int = 0,
        hist_cache: HistSpatialCache | None = None,
        trainable_chunk: int | None = None,
        *,
        frame_ids: Sequence[int] | None = None,
        cross_step_cache: CrossStepHistCache | None = None,
        return_all_frames: bool = False,
        surface_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        del hist_cache  # integrated path is window-native; API kept for callers
        if hidden_win.ndim != 4:
            raise ValueError(f"hidden_win must be [B,T,N,D], got {tuple(hidden_win.shape)}")
        batch, frames, num_points, dim = hidden_win.shape
        if dim != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {dim}")

        if self.spatial_only:
            if return_all_frames:
                point_out = torch.stack(
                    [self._forward_geopt_block(hidden_win[:, t]) for t in range(frames)],
                    dim=1,
                )
            else:
                point_out = self._forward_geopt_block(hidden_win[:, -1])
            return {
                "point_out": point_out,
                "z_out": None,
                "z_short": None,
                "memory_new": memory_prev,
            }

        if self.use_sv_asym:
            return self._forward_sv_asym(
                hidden_win=hidden_win,
                memory_prev=memory_prev,
                trainable_chunk=trainable_chunk,
                return_all_frames=return_all_frames,
                surface_mask=surface_mask,
                cross_step_cache=cross_step_cache,
                layer_idx=layer_idx,
                frame_ids=frame_ids,
            )
        if self.use_sv_latent:
            raise RuntimeError(
                f"{self.branch_mode} is handled in Model._forward_window_sv_*; "
                "do not call TemporalPhysicsBlock.forward"
            )

        if cross_step_cache is not None and return_all_frames:
            raise RuntimeError("return_all_frames is incompatible with cross_step_cache")

        # Outer LN → integrated Physics-Attn
        x_n = self.ln_1(hidden_win.reshape(batch * frames, num_points, dim)).reshape(
            batch, frames, num_points, dim
        )
        z_win, weights_win = self._slice_spatial_window(
            x_n,
            layer_idx=layer_idx,
            frame_ids=frame_ids,
            cross_step_cache=cross_step_cache,
        )
        z_for_attn = self._apply_chunk_detach(z_win, trainable_chunk)
        z_temporal = self._run_causal(z_for_attn)
        z_short = z_temporal[:, -1]

        memory_new = memory_prev
        if self.use_long_memory:
            if memory_prev is None:
                raise ValueError("memory_prev is required when use_long_memory=True")
            if return_all_frames:
                raise RuntimeError(
                    "return_all_frames is incompatible with use_long_memory"
                )
            assert self.long_read is not None and self.memory_update is not None
            read_out, _ = self.long_read(z_short, memory_prev)
            if self.use_temporal_scale_gate:
                z_out = z_short + self.long_scale * (read_out - z_short)
            else:
                z_out = read_out
            memory_new, _, _ = self.memory_update(z_short, memory_prev)
            z_deslice = z_temporal.clone()
            z_deslice[:, -1] = z_out
        else:
            z_out = z_short
            z_deslice = z_temporal

        if return_all_frames:
            bt = batch * frames
            attn_pts = self._deslice_merged(
                z_deslice.reshape(bt, self.slice_num, dim),
                weights_win.reshape(bt, self.num_heads, num_points, self.slice_num),
            ).reshape(batch, frames, num_points, dim)
            x = hidden_win + attn_pts
            point_out = self._ffn_residual_window(x)
            return {
                "point_out": point_out,
                "z_out": z_temporal,
                "z_short": z_short,
                "memory_new": memory_new,
            }

        attn_pts = self._deslice_merged(z_deslice[:, -1], weights_win[:, -1])
        x = hidden_win[:, -1] + attn_pts
        point_out = x + self.mlp(self.ln_2(x))
        return {
            "point_out": point_out,
            "z_out": z_out,
            "z_short": z_short,
            "memory_new": memory_new,
        }

    def _expand_surface_mask(
        self, surface_mask: torch.Tensor, *, batch: int, frames: int, num_points: int
    ) -> torch.Tensor:
        if surface_mask.ndim == 2:
            if surface_mask.shape != (batch, num_points):
                raise ValueError(
                    f"surface_mask [B,N] expected {(batch, num_points)}, got {tuple(surface_mask.shape)}"
                )
            return surface_mask[:, None, :].expand(-1, frames, -1)
        if surface_mask.ndim == 3:
            if surface_mask.shape != (batch, frames, num_points):
                raise ValueError(
                    f"surface_mask [B,T,N] expected {(batch, frames, num_points)}, "
                    f"got {tuple(surface_mask.shape)}"
                )
            return surface_mask
        raise ValueError(f"surface_mask must be [B,N] or [B,T,N], got {tuple(surface_mask.shape)}")

    def sv_latent_slice(
        self, hidden_win: torch.Tensor, surface_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-shot LN + masked S/V slice (no spatial). Returns tokens + weights + surf."""
        batch, frames, num_points, dim = hidden_win.shape
        surf = self._expand_surface_mask(
            surface_mask, batch=batch, frames=frames, num_points=num_points
        ).to(dtype=torch.bool, device=hidden_win.device)
        vol = ~surf
        x_n = self.ln_1(hidden_win.reshape(batch * frames, num_points, dim)).reshape(
            batch, frames, num_points, dim
        )
        flat = x_n.reshape(batch * frames, num_points, dim)
        zs_flat, ws_flat = self._pool_slice_tokens_masked(flat, surf.reshape(batch * frames, num_points))
        zv_flat, wv_flat = self._pool_slice_tokens_masked(flat, vol.reshape(batch * frames, num_points))
        z_s = zs_flat.reshape(batch, frames, self.slice_num, dim)
        z_v = zv_flat.reshape(batch, frames, self.slice_num, dim)
        ws_win = ws_flat.reshape(batch, frames, self.num_heads, num_points, self.slice_num)
        wv_win = wv_flat.reshape(batch, frames, self.num_heads, num_points, self.slice_num)
        return z_s, z_v, ws_win, wv_win, surf

    def _spatial_attend_window(self, z_win: torch.Tensor) -> torch.Tensor:
        """Self-attn on ``[B,T,G,D]`` tokens (shared Attn; S/V call separately)."""
        batch, frames, g, dim = z_win.shape
        flat = z_win.reshape(batch * frames, g, dim)
        out = self._spatial_attend_merged(flat)
        return out.reshape(batch, frames, g, dim)

    def _ffn_residual_tokens(self, z_win: torch.Tensor) -> torch.Tensor:
        """LN→FFN residual on token window ``[B,T,G,D]``."""
        batch, frames, g, dim = z_win.shape
        flat = z_win.reshape(batch * frames, g, dim)
        return (flat + self.mlp(self.ln_2(flat))).reshape(batch, frames, g, dim)

    def sv_latent_token_layer(
        self,
        z_s: torch.Tensor,
        z_v: torch.Tensor,
        *,
        trainable_chunk: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token-space: Self → Cross → Causal → LN→FFN (S/V share Self/Cross weights)."""
        # Self (shared Physics Attn spa) + residual — needed for deep token stack.
        z_s = z_s + self._spatial_attend_window(z_s)
        z_v = z_v + self._spatial_attend_window(z_v)
        assert self.sv_cross is not None
        z_s_in = self._apply_chunk_detach(z_s, trainable_chunk)
        z_v_in = self._apply_chunk_detach(z_v, trainable_chunk)
        z_s, z_v = self.sv_cross(z_s_in, z_v_in)
        z_s = self._run_causal(z_s)
        z_v = self._run_causal(z_v)
        z_s = self._ffn_residual_tokens(z_s)
        z_v = self._ffn_residual_tokens(z_v)
        return z_s, z_v

    def sv_latent_deslice(
        self,
        z_s: torch.Tensor,
        z_v: torch.Tensor,
        ws_win: torch.Tensor,
        wv_win: torch.Tensor,
        surf: torch.Tensor,
        *,
        return_all_frames: bool,
    ) -> torch.Tensor:
        """One-shot deslice + mask-merge → point features ``[B,T,N,D]`` or ``[B,N,D]``."""
        batch, frames, _, dim = z_s.shape
        num_points = surf.shape[-1]
        if return_all_frames:
            bt = batch * frames
            pts_s = self._deslice_merged(
                z_s.reshape(bt, self.slice_num, dim),
                ws_win.reshape(bt, self.num_heads, num_points, self.slice_num),
            ).reshape(batch, frames, num_points, dim)
            pts_v = self._deslice_merged(
                z_v.reshape(bt, self.slice_num, dim),
                wv_win.reshape(bt, self.num_heads, num_points, self.slice_num),
            ).reshape(batch, frames, num_points, dim)
            return torch.where(surf.unsqueeze(-1), pts_s, pts_v)
        pts_s = self._deslice_merged(z_s[:, -1], ws_win[:, -1])
        pts_v = self._deslice_merged(z_v[:, -1], wv_win[:, -1])
        return torch.where(surf[:, -1].unsqueeze(-1), pts_s, pts_v)

    def _forward_sv_asym(
        self,
        *,
        hidden_win: torch.Tensor,
        memory_prev: torch.Tensor | None,
        trainable_chunk: int | None,
        return_all_frames: bool,
        surface_mask: torch.Tensor | None,
        cross_step_cache: CrossStepHistCache | None,
        layer_idx: int = 0,
        frame_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Integrated asym (same outer shell as unified tf4): LN → masked slice →
        spa-only → frame-local cross → causal → deslice → FFN.
        """
        del layer_idx, frame_ids
        if cross_step_cache is not None:
            raise RuntimeError("surface_volume_asym incompatible with cross_step_cache")
        if self.use_long_memory:
            raise RuntimeError("surface_volume_asym incompatible with use_long_memory")
        if surface_mask is None:
            raise RuntimeError("surface_volume_asym requires surface_mask [B,T,N] or [B,N]")

        batch, frames, num_points, dim = hidden_win.shape
        if surface_mask.ndim == 2:
            if surface_mask.shape != (batch, num_points):
                raise ValueError(
                    f"surface_mask [B,N] expected {(batch, num_points)}, got {tuple(surface_mask.shape)}"
                )
            surf = surface_mask[:, None, :].expand(-1, frames, -1)
        elif surface_mask.ndim == 3:
            if surface_mask.shape != (batch, frames, num_points):
                raise ValueError(
                    f"surface_mask [B,T,N] expected {(batch, frames, num_points)}, "
                    f"got {tuple(surface_mask.shape)}"
                )
            surf = surface_mask
        else:
            raise ValueError(f"surface_mask must be [B,N] or [B,T,N], got {tuple(surface_mask.shape)}")
        surf = surf.to(dtype=torch.bool, device=hidden_win.device)
        vol = ~surf

        x_n = self.ln_1(hidden_win.reshape(batch * frames, num_points, dim)).reshape(
            batch, frames, num_points, dim
        )
        flat = x_n.reshape(batch * frames, num_points, dim)
        surf_flat = surf.reshape(batch * frames, num_points)
        vol_flat = vol.reshape(batch * frames, num_points)
        zs_flat, ws_flat = self._pool_slice_tokens_masked(flat, surf_flat)
        zv_flat, wv_flat = self._pool_slice_tokens_masked(flat, vol_flat)
        zs_flat = self._spatial_attend_merged(zs_flat)
        zv_flat = self._spatial_attend_merged(zv_flat)
        z_s = zs_flat.reshape(batch, frames, self.slice_num, dim)
        z_v = zv_flat.reshape(batch, frames, self.slice_num, dim)
        ws_win = ws_flat.reshape(batch, frames, self.num_heads, num_points, self.slice_num)
        wv_win = wv_flat.reshape(batch, frames, self.num_heads, num_points, self.slice_num)

        z_s_in = self._apply_chunk_detach(z_s, trainable_chunk)
        z_v_in = self._apply_chunk_detach(z_v, trainable_chunk)
        assert self.sv_cross is not None
        z_s_x, z_v_x = self.sv_cross(z_s_in, z_v_in)
        z_s_t = self._run_causal(z_s_x)
        z_v_t = self._run_causal(z_v_x)

        if return_all_frames:
            bt = batch * frames
            pts_s = self._deslice_merged(
                z_s_t.reshape(bt, self.slice_num, dim),
                ws_win.reshape(bt, self.num_heads, num_points, self.slice_num),
            ).reshape(batch, frames, num_points, dim)
            pts_v = self._deslice_merged(
                z_v_t.reshape(bt, self.slice_num, dim),
                wv_win.reshape(bt, self.num_heads, num_points, self.slice_num),
            ).reshape(batch, frames, num_points, dim)
            attn_pts = torch.where(surf.unsqueeze(-1), pts_s, pts_v)
            x = hidden_win + attn_pts
            point_out = self._ffn_residual_window(x)
            return {
                "point_out": point_out,
                "z_out": z_s_t,
                "z_short": z_s_t[:, -1],
                "memory_new": memory_prev,
            }

        pts_s = self._deslice_merged(z_s_t[:, -1], ws_win[:, -1])
        pts_v = self._deslice_merged(z_v_t[:, -1], wv_win[:, -1])
        attn_pts = torch.where(surf[:, -1].unsqueeze(-1), pts_s, pts_v)
        x = hidden_win[:, -1] + attn_pts
        point_out = x + self.mlp(self.ln_2(x))
        return {
            "point_out": point_out,
            "z_out": z_s_t[:, -1],
            "z_short": z_s_t[:, -1],
            "memory_new": memory_prev,
        }
