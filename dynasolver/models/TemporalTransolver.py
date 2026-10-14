from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from layers.Long_Flow_Memory import init_memory
from layers.Temporal_Physics_Block import (
    CrossStepHistCache,
    HistSpatialCache,
    TemporalPhysicsBlock,
)
from models.Transolver import MLP

_VALID_LONG_PLACEMENTS = frozenset({"none", "per_layer", "mid_after_layer4", "last_layer"})
_ACT = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class PointResidualMLP(nn.Module):
    """Pre-LN residual MLP on point features ``[..., D]`` (dense decoder block)."""

    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: int = 2,
        act: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if act not in _ACT:
            raise ValueError(f"unsupported act {act!r}; choose from {sorted(_ACT)}")
        hidden = int(dim * mlp_ratio)
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = _ACT[act]()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + h


class PointDecoder(nn.Module):
    """Stack of Pre-LN residual MLPs after Deslice (dense ViT-style decode)."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        *,
        mlp_ratio: int = 2,
        act: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_layers < 0:
            raise ValueError(f"point_decoder_layers must be >= 0, got {n_layers}")
        self.blocks = nn.ModuleList(
            [
                PointResidualMLP(dim, mlp_ratio=mlp_ratio, act=act, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        # Identity at init: decoder starts as no-op, head still drives early Δ.
        for block in self.blocks:
            nn.init.zeros_(block.fc2.weight)
            nn.init.zeros_(block.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x



def resolve_long_memory_layers(
    n_layers: int,
    *,
    use_long_memory: bool,
    placement: str,
    mid_long_after_layer: int = 4,
) -> list[bool]:
    """Per-layer flags for LongMemoryRead/Update.

    Placements (1-based layer wording in configs):
      - none: no long memory
      - per_layer: every block
      - mid_after_layer4: only after 1-based layer ``mid_long_after_layer`` (default 4),
        i.e. between that layer and the next (layer_idx == mid_long_after_layer - 1)
      - last_layer: only the final block
    """
    placement = str(placement or ("per_layer" if use_long_memory else "none"))
    if placement not in _VALID_LONG_PLACEMENTS:
        raise ValueError(
            f"long_memory_placement={placement!r}; choose from {sorted(_VALID_LONG_PLACEMENTS)}"
        )
    if not use_long_memory or placement == "none":
        return [False] * n_layers
    if placement == "per_layer":
        return [True] * n_layers
    if placement == "last_layer":
        return [i == n_layers - 1 for i in range(n_layers)]
    # mid_after_layer4 (or mid_long_after_layer override)
    mid_1based = int(mid_long_after_layer)
    if mid_1based < 1 or mid_1based > n_layers:
        raise ValueError(
            f"mid_long_after_layer={mid_1based} out of range for n_layers={n_layers}"
        )
    mid_idx = mid_1based - 1
    return [i == mid_idx for i in range(n_layers)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Model(nn.Module):
    """
    Temporal Transolver for dynamic-boundary CFD.

    Spatial path reuses GeoPT ``preprocess`` + ``Transolver_block`` Physics Attention weights.
    Temporal path adds block-causal attention and recurrent linear memory on slice tokens.
    """

    def __init__(self, args: Namespace):
        super().__init__()
        if getattr(args, "geotype", "unstructured") != "unstructured":
            raise ValueError("TemporalTransolver currently supports only unstructured point sets")
        if int(args.space_dim) != 3 or int(args.fun_dim) != 11:
            raise ValueError("TemporalTransolver requires space_dim == 3 and fun_dim == 11")
        if int(args.state_dim) <= 0 or int(args.boundary_dim) <= 0:
            raise ValueError("state_dim and boundary_dim must be positive")

        self.__name__ = "TemporalTransolver"
        self.args = args
        self.state_dim = int(args.state_dim)
        self.boundary_dim = int(args.boundary_dim)
        self.hidden_dim = int(args.n_hidden)
        self.window_frames = int(getattr(args, "window_frames", 5))
        self.num_tokens = int(getattr(args, "slice_num", 32))
        self.num_heads = int(args.n_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.use_long_memory = bool(getattr(args, "use_long_memory", True))
        self.use_short_window = bool(getattr(args, "use_short_window", True))
        self.use_boundary_feat = bool(getattr(args, "use_boundary_feat", True))
        # If False: hist frames embed query+static(+boundary) only; last-frame state
        # still feeds state_proj (needed for residual head on state[:, -1]).
        self.use_hist_state_embed = bool(getattr(args, "use_hist_state_embed", True))
        self.branch_mode = str(getattr(args, "branch_mode", "unified"))
        self.cross_vs_scale_init = float(getattr(args, "cross_vs_scale_init", 0.0))
        self.share_cross_weights = bool(getattr(args, "share_cross_weights", False))
        self.point_decoder_layers = int(getattr(args, "point_decoder_layers", 0))
        if self.point_decoder_layers < 0:
            raise ValueError(
                f"point_decoder_layers must be >= 0, got {self.point_decoder_layers}"
            )
        if self.point_decoder_layers > 0 and self.branch_mode != "surface_volume_latent":
            raise ValueError(
                "point_decoder_layers > 0 requires branch_mode=surface_volume_latent "
                f"(got {self.branch_mode!r})"
            )
        self.long_memory_placement = str(
            getattr(
                args,
                "long_memory_placement",
                "per_layer" if self.use_long_memory else "none",
            )
        )
        self.mid_long_after_layer = int(getattr(args, "mid_long_after_layer", 4))
        stem_dim = self.hidden_dim * 2

        self.preprocess = MLP(
            args.fun_dim + args.space_dim,
            stem_dim,
            self.hidden_dim,
            n_layers=0,
            res=False,
            act=getattr(args, "act", "gelu"),
        )
        self.state_proj = nn.Linear(self.state_dim, stem_dim, bias=False)
        self.boundary_proj = nn.Linear(self.boundary_dim, stem_dim, bias=False)
        self.placeholder = nn.Parameter(torch.rand(self.hidden_dim) / self.hidden_dim)

        n_layers = int(args.n_layers)
        self.long_memory_layers = resolve_long_memory_layers(
            n_layers,
            use_long_memory=self.use_long_memory,
            placement=self.long_memory_placement,
            mid_long_after_layer=self.mid_long_after_layer,
        )
        # Model-level flag: any layer uses long → need memory tensor.
        self.use_long_memory = any(self.long_memory_layers)
        self.blocks = nn.ModuleList(
            [
                TemporalPhysicsBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    slice_num=self.num_tokens,
                    dropout=float(getattr(args, "dropout", 0.0)),
                    act=str(getattr(args, "act", "gelu")),
                    mlp_ratio=int(getattr(args, "mlp_ratio", 2)),
                    geotype=str(getattr(args, "geotype", "unstructured")),
                    shapelist=getattr(args, "shapelist", None),
                    short_scale_init=float(getattr(args, "short_scale_init", 1.0e-3)),
                    long_scale_init=float(getattr(args, "long_scale_init", 1.0e-3)),
                    use_rope=bool(getattr(args, "use_rope", True)),
                    use_long_memory=self.long_memory_layers[layer_idx],
                    use_short_window=self.use_short_window,
                    use_temporal_scale_gate=bool(getattr(args, "use_temporal_scale_gate", False)),
                    slice_mode=str(getattr(args, "slice_mode", "legacy")),
                    use_flash_attn=bool(getattr(args, "use_flash_attn", False)),
                    branch_mode=self.branch_mode,
                    cross_vs_scale_init=self.cross_vs_scale_init,
                    share_cross_weights=self.share_cross_weights,
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.state_dim)
        self.point_decoder: PointDecoder | None
        if self.point_decoder_layers > 0:
            self.point_decoder = PointDecoder(
                self.hidden_dim,
                self.point_decoder_layers,
                mlp_ratio=int(getattr(args, "mlp_ratio", 2)),
                act=str(getattr(args, "act", "gelu")),
                dropout=float(getattr(args, "dropout", 0.0)),
            )
        else:
            self.point_decoder = None

        self.apply(self._init_weights)
        nn.init.zeros_(self.state_proj.weight)
        nn.init.zeros_(self.boundary_proj.weight)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        # Re-apply identity init after apply() (trunc_normal would break no-op start).
        if self.point_decoder is not None:
            for block in self.point_decoder.blocks:
                nn.init.zeros_(block.fc2.weight)
                nn.init.zeros_(block.fc2.bias)
        # Re-zero temporal branch projections (apply() overwrote module-level zeros).
        for block in self.blocks:
            if block.temporal_attn is not None:
                nn.init.zeros_(block.temporal_attn.out_proj.weight)
                nn.init.zeros_(block.temporal_attn.out_proj.bias)
            if block.sv_cross is not None:
                nn.init.zeros_(block.sv_cross.cross_sv.out_proj.weight)
                nn.init.zeros_(block.sv_cross.cross_sv.out_proj.bias)
                if block.sv_cross.cross_vs is not block.sv_cross.cross_sv:
                    nn.init.zeros_(block.sv_cross.cross_vs.out_proj.weight)
                    nn.init.zeros_(block.sv_cross.cross_vs.out_proj.bias)
            if block.long_read is not None:
                nn.init.zeros_(block.long_read.out_proj.weight)
                nn.init.zeros_(block.long_read.out_proj.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.use_long_memory:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        return init_memory(
            batch_size,
            device,
            dtype,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

    def embed_frame(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        if query_xyz.shape[-1] != 3 or static_fx.shape[-1] != int(self.args.fun_dim):
            raise ValueError("query_xyz/static_fx channel mismatch")
        if state.shape[-1] != self.state_dim:
            raise ValueError("state channel mismatch")
        if self.use_boundary_feat:
            if boundary_feat.shape[-1] != self.boundary_dim:
                raise ValueError("boundary channel mismatch")
        base = torch.cat((query_xyz, static_fx), dim=-1)
        base_linear = self.preprocess.linear_pre[0]
        activation = self.preprocess.linear_pre[1]
        preactivation = base_linear(base)
        preactivation = preactivation + self.state_proj(state)
        if self.use_boundary_feat:
            preactivation = preactivation + self.boundary_proj(boundary_feat)
        else:
            del boundary_feat
        hidden = self.preprocess.linear_post(activation(preactivation))
        return hidden + self.placeholder[None, None, :]

    def forward_window(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        state: torch.Tensor,
        boundary_feat: torch.Tensor,
        memory_prev: torch.Tensor | None = None,
        trainable_chunk: int | None = None,
        *,
        frame_ids: Sequence[int] | None = None,
        cross_step_cache: CrossStepHistCache | None = None,
        parallel_causal_tf: bool = False,
        surface_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward a causal window of length ``T`` with ``1 <= T <= window_frames``.

        ``trainable_chunk``: Plan A=1 / Plan B=2 / None=legacy (all T frames trainable).
        History outside the chunk is cached under stop-grad (Transformer-XL style).

        ``cross_step_cache``: reuse GeoPT/slice by input seq id across PF/AR
        steps (including the previous current). Valid in eval, and in train when
        the caller keeps weights fixed across those steps (pushforward_last_only).

        ``parallel_causal_tf``: return per-position residuals
        ``predictions[B,T,N,4]`` with ``Ŝ_{k+1}=S_k+Δ(h_k)`` after block-causal attn.

        ``surface_mask``: required for ``branch_mode`` in
        ``{surface_volume_asym, surface_volume_latent}`` ([B,T,N] or [B,N]).
        """
        if query_xyz.ndim != 4:
            raise ValueError("window tensors must have shape [B,T,N,*]")
        batch, frames, _, _ = query_xyz.shape
        if frames < 1 or frames > self.window_frames:
            raise ValueError(
                f"expected 1 <= T <= window_frames={self.window_frames}, got {frames}"
            )
        if frame_ids is not None and len(frame_ids) != frames:
            raise ValueError(
                f"frame_ids length {len(frame_ids)} != window frames {frames}"
            )
        if parallel_causal_tf and cross_step_cache is not None:
            raise RuntimeError("parallel_causal_tf incompatible with cross_step_cache")
        if parallel_causal_tf and self.use_long_memory:
            raise RuntimeError("parallel_causal_tf incompatible with use_long_memory")
        needs_surface = self.branch_mode in (
            "surface_volume_asym",
            "surface_volume_latent",
            "surface_volume_per_latent",
        )
        if needs_surface and surface_mask is None:
            raise RuntimeError(f"{self.branch_mode} requires surface_mask")
        if needs_surface and cross_step_cache is not None:
            raise RuntimeError(f"{self.branch_mode} incompatible with cross_step_cache")

        if self.use_long_memory and memory_prev is None:
            memory_prev = self.init_memory(batch, query_xyz.device, query_xyz.dtype)

        hidden_frames = []
        for frame in range(frames):
            state_frame = state[:, frame]
            # Legacy no_hist zeros history CFD embeds; parallel TF keeps all GT states.
            if (
                (not parallel_causal_tf)
                and (not self.use_hist_state_embed)
                and frame < frames - 1
            ):
                state_frame = torch.zeros_like(state_frame)
            hidden_frames.append(
                self.embed_frame(
                    query_xyz[:, frame],
                    static_fx[:, frame],
                    state_frame,
                    boundary_feat[:, frame],
                )
            )
        hidden_win = torch.stack(hidden_frames, dim=1)

        if self.branch_mode == "surface_volume_latent":
            return self._forward_window_sv_latent(
                hidden_win=hidden_win,
                state=state,
                memory_prev=memory_prev,
                trainable_chunk=trainable_chunk,
                parallel_causal_tf=parallel_causal_tf,
                surface_mask=surface_mask,
            )
        if self.branch_mode == "surface_volume_per_latent":
            return self._forward_window_sv_per_latent(
                hidden_win=hidden_win,
                state=state,
                memory_prev=memory_prev,
                trainable_chunk=trainable_chunk,
                parallel_causal_tf=parallel_causal_tf,
                surface_mask=surface_mask,
            )

        memory = memory_prev
        point_hidden = None
        hist_cache = HistSpatialCache(window_frames=frames, n_layers=len(self.blocks))
        for layer_idx, block in enumerate(self.blocks):
            out = block(
                hidden_win,
                memory if self.use_long_memory else None,
                layer_idx=layer_idx,
                hist_cache=hist_cache,
                trainable_chunk=trainable_chunk,
                frame_ids=frame_ids,
                cross_step_cache=cross_step_cache,
                return_all_frames=parallel_causal_tf,
                surface_mask=surface_mask,
            )
            point_out = out["point_out"]
            if parallel_causal_tf:
                hidden_win = point_out
                point_hidden = point_out
            else:
                hidden_win = hidden_win.clone()
                hidden_win[:, -1] = point_out
                point_hidden = point_out
            if self.use_long_memory:
                # Blocks without long return memory_prev unchanged.
                memory = out["memory_new"]

        assert point_hidden is not None
        if parallel_causal_tf:
            n_points = point_hidden.shape[2]
            flat = point_hidden.reshape(batch * frames, n_points, -1)
            delta_flat = self.output_proj(self.output_norm(flat))
            delta = delta_flat.reshape(batch, frames, n_points, self.state_dim)
            predictions = state + delta
            return {
                "prediction": predictions[:, -1],
                "predictions": predictions,
                "delta": delta,
                "memory_new": memory if self.use_long_memory else memory_prev,
                "hidden": point_hidden,
            }

        delta = self.output_proj(self.output_norm(point_hidden))
        prediction = state[:, -1] + delta
        return {
            "prediction": prediction,
            "delta": delta,
            "memory_new": memory if self.use_long_memory else memory_prev,
            "hidden": point_hidden,
        }

    def _forward_window_sv_latent(
        self,
        *,
        hidden_win: torch.Tensor,
        state: torch.Tensor,
        memory_prev: torch.Tensor | None,
        trainable_chunk: int | None,
        parallel_causal_tf: bool,
        surface_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """One-shot Slice → L× token stack → one-shot Deslice → point MLP decoder → LN→Linear→Δ.

        Differs from ``surface_volume_asym`` by keeping Self/Cross/Causal/FFN entirely
        in token space between a single Slice and Deslice (no per-layer point roundtrip).
        Optional ``point_decoder`` (Pre-LN residual MLPs) refines point features after
        Deslice before the residual head — dense ViT-style decode.
        """
        if self.use_long_memory:
            raise RuntimeError("surface_volume_latent incompatible with use_long_memory")
        assert surface_mask is not None
        batch, frames, _, _ = hidden_win.shape
        # Slice with block0 (LN+pool); stack token layers with each block's weights.
        z_s, z_v, ws_win, wv_win, surf = self.blocks[0].sv_latent_slice(
            hidden_win, surface_mask
        )
        for block in self.blocks:
            z_s, z_v = block.sv_latent_token_layer(
                z_s, z_v, trainable_chunk=trainable_chunk
            )
        # Deslice with block0 weights (same Attn.deslice as the initial slice).
        point_hidden = self.blocks[0].sv_latent_deslice(
            z_s,
            z_v,
            ws_win,
            wv_win,
            surf,
            return_all_frames=parallel_causal_tf,
        )
        if parallel_causal_tf:
            # Optional residual from embed (stabilizes Δ); then dense point decode.
            point_hidden = hidden_win + point_hidden
            if self.point_decoder is not None:
                point_hidden = self.point_decoder(point_hidden)
            n_points = point_hidden.shape[2]
            flat = point_hidden.reshape(batch * frames, n_points, -1)
            delta_flat = self.output_proj(self.output_norm(flat))
            delta = delta_flat.reshape(batch, frames, n_points, self.state_dim)
            predictions = state + delta
            return {
                "prediction": predictions[:, -1],
                "predictions": predictions,
                "delta": delta,
                "memory_new": memory_prev,
                "hidden": point_hidden,
            }
        point_hidden = hidden_win[:, -1] + point_hidden
        if self.point_decoder is not None:
            point_hidden = self.point_decoder(point_hidden)
        delta = self.output_proj(self.output_norm(point_hidden))
        prediction = state[:, -1] + delta
        return {
            "prediction": prediction,
            "delta": delta,
            "memory_new": memory_prev,
            "hidden": point_hidden,
        }

    def _forward_window_sv_per_latent(
        self,
        *,
        hidden_win: torch.Tensor,
        state: torch.Tensor,
        memory_prev: torch.Tensor | None,
        trainable_chunk: int | None,
        parallel_causal_tf: bool,
        surface_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Per-layer Slice → token stack → Deslice (ablation of one-shot latent).

        Same S/V token ops as ``surface_volume_latent``, but each layer re-slices from
        points and deslices back (point residual), matching the cost structure of
        ``surface_volume_asym`` while keeping latent's token-space FFN order.
        """
        if self.use_long_memory:
            raise RuntimeError(
                "surface_volume_per_latent incompatible with use_long_memory"
            )
        assert surface_mask is not None
        batch, frames, _, _ = hidden_win.shape
        hidden = hidden_win
        for block in self.blocks:
            z_s, z_v, ws_win, wv_win, surf = block.sv_latent_slice(hidden, surface_mask)
            z_s, z_v = block.sv_latent_token_layer(
                z_s, z_v, trainable_chunk=trainable_chunk
            )
            point = block.sv_latent_deslice(
                z_s,
                z_v,
                ws_win,
                wv_win,
                surf,
                return_all_frames=True,
            )
            hidden = hidden + point
        point_hidden = hidden
        if parallel_causal_tf:
            n_points = point_hidden.shape[2]
            flat = point_hidden.reshape(batch * frames, n_points, -1)
            delta_flat = self.output_proj(self.output_norm(flat))
            delta = delta_flat.reshape(batch, frames, n_points, self.state_dim)
            predictions = state + delta
            return {
                "prediction": predictions[:, -1],
                "predictions": predictions,
                "delta": delta,
                "memory_new": memory_prev,
                "hidden": point_hidden,
            }
        last = point_hidden[:, -1]
        delta = self.output_proj(self.output_norm(last))
        prediction = state[:, -1] + delta
        return {
            "prediction": prediction,
            "delta": delta,
            "memory_new": memory_prev,
            "hidden": last,
        }

    def forward(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        state: torch.Tensor,
        boundary_feat: torch.Tensor,
        memory_prev: torch.Tensor | None = None,
        surface_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_window(
            query_xyz,
            static_fx,
            state,
            boundary_feat,
            memory_prev=memory_prev,
            surface_mask=surface_mask,
        )["prediction"]


def _geopt_spatial_block_prefix(layer_index: int) -> str:
    return f"blocks.{layer_index}."


def _legacy_static_head_keys(layer_index: int) -> set[str]:
    prefix = _geopt_spatial_block_prefix(layer_index)
    return {
        f"{prefix}ln_3.weight",
        f"{prefix}ln_3.bias",
        f"{prefix}mlp2.weight",
        f"{prefix}mlp2.bias",
    }


def _geopt_loadable_keys(model: Model) -> set[str]:
    keys: set[str] = {"placeholder"}
    keys.update(key for key in model.state_dict() if key.startswith("preprocess."))
    for layer_index in range(len(model.blocks)):
        prefix = _geopt_spatial_block_prefix(layer_index)
        for suffix in ("ln_1.", "Attn.", "ln_2.", "mlp."):
            keys.update(
                key
                for key in model.state_dict()
                if key.startswith(prefix + suffix)
            )
    return keys


def load_geopt_backbone(
    model: Model,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
) -> dict:
    """Transplant GeoPT preprocess + spatial Transolver block weights into TemporalTransolver."""
    checkpoint_path = Path(checkpoint_path)
    actual_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"GeoPT checkpoint SHA256 mismatch: {actual_sha256} != {expected_sha256}"
        )

    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(source, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in source.values()
    ):
        raise RuntimeError("GeoPT checkpoint must be a plain tensor state_dict")

    target = model.state_dict()
    loadable = _geopt_loadable_keys(model)
    excluded: set[str] = set()
    for layer_index in range(len(model.blocks)):
        excluded.update(_legacy_static_head_keys(layer_index))

    payload: dict[str, torch.Tensor] = {}
    for key in sorted(loadable):
        if key not in source:
            raise RuntimeError(f"GeoPT checkpoint is missing required key: {key}")
        if source[key].shape != target[key].shape:
            raise RuntimeError(
                f"GeoPT tensor shape mismatch for {key}: "
                f"{tuple(source[key].shape)} != {tuple(target[key].shape)}"
            )
        payload[key] = source[key]

    load_result = model.load_state_dict(payload, strict=False)
    expected_missing = set(target) - loadable
    if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected TemporalTransolver load result: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    loaded_state = model.state_dict()
    unequal = [
        key
        for key in sorted(loadable)
        if not torch.equal(loaded_state[key].cpu(), source[key].cpu())
    ]
    if unequal:
        raise RuntimeError(f"GeoPT tensors changed during load: {unequal}")

    return {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "source_tensor_count": len(source),
        "loaded_tensor_count": len(payload),
        "loaded_parameter_elements": sum(source[key].numel() for key in payload),
        "excluded_keys": sorted(excluded & set(source)),
        "new_keys": sorted(expected_missing),
    }
