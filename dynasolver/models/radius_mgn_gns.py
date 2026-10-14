"""Paper-faithful MeshGraphNet / GNS adaptations for Eulerian Temporal CFD.

Shared: radius graph on fixed point bank, Encode-Process-Decode with edge/node
latents and residuals, residual field update Ŷ = Y_t + Δ (task-aligned with
V0 / transolver_fs).

Differences (feature scheme A):
  * mgn_radius — single-frame state; boundary/type cues only in the encoder;
    15 message-passing steps, hidden=128 (MeshGraphNet defaults).
  * gns_radius — multi-frame state history; boundary features in every
    processor (GNS-style boundary conditioning); 10 steps, hidden=128
    (learning_to_simulate defaults).

These are task adaptations (Eulerian residual, radius on fixed xyz), not
particle solvers and not mesh-edge MGN.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


def estimate_connectivity_radius(
    points: np.ndarray | torch.Tensor, *, target_degree: int = 16
) -> float:
    """Median distance to the ``target_degree``-th neighbour (excludes self)."""
    if isinstance(points, torch.Tensor):
        xyz_t = points.detach().cpu().to(dtype=torch.float32)
        xyz = xyz_t.numpy()
    else:
        xyz = np.asarray(points, dtype=np.float32)
        xyz_t = torch.as_tensor(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {xyz.shape}")
    n = int(xyz.shape[0])
    if not (1 <= target_degree < n):
        raise ValueError(f"target_degree must be in [1,{n - 1}], got {target_degree}")
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(xyz.astype(np.float64)).query(
            xyz.astype(np.float64), k=target_degree + 1, workers=1
        )
        radius = float(np.median(distances[:, -1]))
    except Exception:
        # Fallback when SciPy/NumPy ABI mismatches (common on shared clusters).
        dist = torch.cdist(xyz_t, xyz_t)
        dist.fill_diagonal_(float("inf"))
        knn = torch.topk(dist, k=target_degree, largest=False).values
        radius = float(knn[:, -1].median().item())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"failed to estimate a positive radius, got {radius}")
    return radius


def radius_edges(
    points: np.ndarray | torch.Tensor, radius: float, *, max_degree: int = 64
) -> torch.Tensor:
    """Undirected radius graph as ``[2, E]`` (src, dst), no self-loops."""
    if isinstance(points, torch.Tensor):
        xyz_t = points.detach().cpu().to(dtype=torch.float32)
        xyz = xyz_t.numpy()
    else:
        xyz = np.asarray(points, dtype=np.float32)
        xyz_t = torch.as_tensor(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {xyz.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    n = int(xyz.shape[0])

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    used_scipy = False
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz.astype(np.float64))
        neighbours_list = tree.query_ball_point(xyz.astype(np.float64), r=radius, workers=1)
        used_scipy = True
        for i, nbrs in enumerate(neighbours_list):
            nbrs = [j for j in nbrs if j != i]
            if len(nbrs) > max_degree:
                dists = np.linalg.norm(xyz[nbrs] - xyz[i], axis=-1)
                keep = np.argpartition(dists, max_degree - 1)[:max_degree]
                nbrs = [nbrs[k] for k in keep.tolist()]
            if not nbrs:
                continue
            dst_chunks.append(np.full(len(nbrs), i, dtype=np.int64))
            src_chunks.append(np.asarray(nbrs, dtype=np.int64))
    except Exception:
        used_scipy = False

    if not used_scipy:
        dist = torch.cdist(xyz_t, xyz_t)
        mask = (dist <= radius) & (dist > 0)
        for i in range(n):
            nbrs = torch.nonzero(mask[i], as_tuple=False).flatten()
            if nbrs.numel() == 0:
                continue
            if nbrs.numel() > max_degree:
                d = dist[i, nbrs]
                keep = torch.topk(d, k=max_degree, largest=False).indices
                nbrs = nbrs[keep]
            src_chunks.append(nbrs.numpy().astype(np.int64))
            dst_chunks.append(np.full(nbrs.numel(), i, dtype=np.int64))

    if not src_chunks:
        raise ValueError(f"radius={radius} produced zero edges on {n} points")
    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)
    edge = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge)


def build_mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    *,
    num_hidden_layers: int = 2,
    layer_norm: bool = True,
) -> nn.Module:
    """Official-style MLP: ``num_hidden_layers`` hidden blocks then output."""
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_hidden_layers):
        layers.extend([nn.Linear(dim, hidden), nn.ReLU()])
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    mlp = nn.Sequential(*layers)
    if layer_norm:
        return nn.Sequential(mlp, nn.LayerNorm(out_dim))
    return mlp


class EncodeProcessDecode(nn.Module):
    """Shared EPD core: independent encoders, InteractionNetwork-style steps."""

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        latent_size: int = 128,
        mlp_hidden_size: int = 128,
        mlp_num_hidden_layers: int = 2,
        num_message_passing_steps: int = 10,
        output_size: int = 4,
        processor_extra_node_dim: int = 0,
    ) -> None:
        super().__init__()
        self.processor_extra_node_dim = int(processor_extra_node_dim)
        self.node_encoder = build_mlp(
            node_in_dim, mlp_hidden_size, latent_size, num_hidden_layers=mlp_num_hidden_layers
        )
        self.edge_encoder = build_mlp(
            edge_in_dim, mlp_hidden_size, latent_size, num_hidden_layers=mlp_num_hidden_layers
        )
        self.edge_processors = nn.ModuleList()
        self.node_processors = nn.ModuleList()
        for _ in range(num_message_passing_steps):
            self.edge_processors.append(
                build_mlp(
                    latent_size * 3,
                    mlp_hidden_size,
                    latent_size,
                    num_hidden_layers=mlp_num_hidden_layers,
                )
            )
            self.node_processors.append(
                build_mlp(
                    latent_size * 2 + self.processor_extra_node_dim,
                    mlp_hidden_size,
                    latent_size,
                    num_hidden_layers=mlp_num_hidden_layers,
                )
            )
        self.decoder = build_mlp(
            latent_size,
            mlp_hidden_size,
            output_size,
            num_hidden_layers=mlp_num_hidden_layers,
            layer_norm=False,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        processor_node_extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # node_features: [B,N,F], edge_features: [B,E,F_e] or [E,F_e]
        if edge_features.ndim == 2:
            edge_features = edge_features.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        src, dst = edge_index[0], edge_index[1]
        for edge_mlp, node_mlp in zip(self.edge_processors, self.node_processors):
            gathered = torch.cat([nodes[:, src], nodes[:, dst], edges], dim=-1)
            edge_update = edge_mlp(gathered)
            edges = edges + edge_update
            aggregate = nodes.new_zeros(nodes.shape)
            aggregate.index_add_(1, dst, edges)
            pieces = [nodes, aggregate]
            if self.processor_extra_node_dim > 0:
                if processor_node_extra is None:
                    raise ValueError("processor_node_extra required for this model")
                pieces.append(processor_node_extra)
            node_update = node_mlp(torch.cat(pieces, dim=-1))
            nodes = nodes + node_update
        return self.decoder(nodes)


class RadiusGraphSurrogate(nn.Module):
    """Eulerian radius-graph MGN / GNS with residual state update."""

    def __init__(
        self,
        points: np.ndarray | torch.Tensor,
        *,
        model: str = "mgn_radius",
        state_dim: int = 4,
        radius: float | None = None,
        target_degree: int = 16,
        max_degree: int = 64,
        latent_size: int = 128,
        mlp_hidden_size: int = 128,
        mlp_num_hidden_layers: int = 2,
        num_message_passing_steps: int | None = None,
        history: int | None = None,
    ) -> None:
        super().__init__()
        if model not in {"mgn_radius", "gns_radius"}:
            raise ValueError("model must be mgn_radius or gns_radius")
        self.model_name = model
        self.state_dim = int(state_dim)
        if history is None:
            history = 1 if model == "mgn_radius" else 5
        self.history = int(history)
        if model == "mgn_radius" and self.history != 1:
            raise ValueError("mgn_radius uses history=1")
        if self.history < 1:
            raise ValueError("history must be >= 1")
        if num_message_passing_steps is None:
            num_message_passing_steps = 15 if model == "mgn_radius" else 10

        xyz = torch.as_tensor(np.asarray(points), dtype=torch.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"points must be [N,3], got {tuple(xyz.shape)}")
        self.register_buffer("points", xyz, persistent=False)
        if radius is None:
            radius = estimate_connectivity_radius(xyz, target_degree=target_degree)
        self.radius = float(radius)
        edge_index = radius_edges(xyz, self.radius, max_degree=max_degree)
        self.register_buffer("edge_index", edge_index, persistent=False)
        src, dst = edge_index
        rel = xyz[src] - xyz[dst]
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        # GNS normalizes by radius; MGN uses raw relative mesh/world features.
        if model == "gns_radius":
            edge_attr = torch.cat([rel / self.radius, dist / self.radius], dim=-1)
        else:
            edge_attr = torch.cat([rel, dist], dim=-1)
        self.register_buffer("edge_attr", edge_attr, persistent=False)

        # MGN: Y_t + static_fx + fluid masks (type proxy). Boundary motion stays out of processors.
        # GNS: history*Y + static_fx + full boundary_feat; boundary also enters each processor.
        if model == "mgn_radius":
            node_in_dim = self.state_dim * self.history + 11 + 2
            processor_extra = 0
        else:
            node_in_dim = self.state_dim * self.history + 11 + 8
            processor_extra = 8

        self.epd = EncodeProcessDecode(
            node_in_dim=node_in_dim,
            edge_in_dim=int(edge_attr.shape[-1]),
            latent_size=latent_size,
            mlp_hidden_size=mlp_hidden_size,
            mlp_num_hidden_layers=mlp_num_hidden_layers,
            num_message_passing_steps=num_message_passing_steps,
            output_size=self.state_dim,
            processor_extra_node_dim=processor_extra,
        )

    def _pack_previous(self, previous_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(state_flat [B,N,H*C], y_t [B,N,C])``."""
        if previous_state.ndim == 3:
            # [B,N,C] — single frame; tile for history if needed.
            if self.history == 1:
                y_t = previous_state
                flat = previous_state
            else:
                y_t = previous_state
                flat = previous_state.unsqueeze(1).expand(
                    -1, self.history, -1, -1
                ).reshape(previous_state.shape[0], previous_state.shape[1], -1)
            return flat, y_t
        if previous_state.ndim != 4:
            raise ValueError(
                "previous_state must be [B,N,C] or [B,H,N,C], "
                f"got {tuple(previous_state.shape)}"
            )
        if previous_state.shape[1] != self.history:
            raise ValueError(
                f"expected history={self.history}, got {previous_state.shape[1]}"
            )
        if previous_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected state_dim={self.state_dim}, got {previous_state.shape[-1]}"
            )
        y_t = previous_state[:, -1]
        flat = previous_state.transpose(1, 2).reshape(
            previous_state.shape[0], previous_state.shape[2], -1
        )
        return flat, y_t

    def forward(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        del query_xyz  # graph geometry is fixed from construction-time points.
        if static_fx.shape[-1] != 11 or boundary_feat.shape[-1] != 8:
            raise ValueError("static_fx/boundary_feat must have dims 11/8")
        state_flat, y_t = self._pack_previous(previous_state)
        if self.model_name == "mgn_radius":
            # fluid_mask_current / fluid_mask_next — node-type proxy (boundary_feat[5], [6]).
            type_proxy = boundary_feat[..., 5:7]
            node_in = torch.cat([state_flat, static_fx, type_proxy], dim=-1)
            extra = None
        else:
            node_in = torch.cat([state_flat, static_fx, boundary_feat], dim=-1)
            extra = boundary_feat
        edge_index = self.edge_index.to(device=node_in.device)
        edge_attr = self.edge_attr.to(device=node_in.device)
        delta = self.epd(node_in, edge_index, edge_attr, processor_node_extra=extra)
        return y_t + delta


def build_radius_graph_model(
    points: np.ndarray | torch.Tensor, model: str = "mgn_radius", **kwargs
) -> RadiusGraphSurrogate:
    return RadiusGraphSurrogate(points, model=model, **kwargs)
