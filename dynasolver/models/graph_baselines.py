"""Small Eulerian graph surrogates (MGN-kNN / SDF-GNS).

Isolated local copy used for fair mesh-graph baselines in this release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn


def knn_edges(points: np.ndarray | torch.Tensor, k: int = 16) -> torch.Tensor:
    """Build bidirectional kNN edges as ``[2,E]`` (source, destination).

    scipy is used for 30k-point banks to avoid an O(N^2) distance matrix.  A
    torch cdist fallback is intentionally retained for small CPU smoke tests.
    """
    if isinstance(points, torch.Tensor):
        xyz = points.detach().cpu().numpy()
    else:
        xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {xyz.shape}")
    n = xyz.shape[0]
    if n < 2 or not (1 <= k < n):
        raise ValueError(f"k must be in [1,{n - 1}], got {k}")
    try:
        from scipy.spatial import cKDTree

        _, neighbours = cKDTree(xyz).query(xyz, k=k + 1, workers=1)
        neighbours = np.asarray(neighbours)[:, 1:]
    except Exception:
        distances = torch.cdist(torch.as_tensor(xyz), torch.as_tensor(xyz))
        neighbours = torch.topk(distances, k=k + 1, largest=False).indices[:, 1:].numpy()
    dst = np.repeat(np.arange(n, dtype=np.int64), k)
    src = neighbours.reshape(-1).astype(np.int64)
    # Adding the reverse direction makes aggregation independent of query order.
    edge = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    return torch.from_numpy(edge)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphProcessor(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, boundary_dim: int = 0) -> None:
        super().__init__()
        self.edge_mlp = MLP(hidden * 2 + edge_dim, hidden, hidden)
        self.node_mlp = MLP(hidden * 2 + boundary_dim, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, boundary: Optional[torch.Tensor] = None) -> torch.Tensor:
        src, dst = edge_index
        # [B,E,H], aggregate without torch_scatter/PyG.
        messages = self.edge_mlp(torch.cat([nodes[:, src], nodes[:, dst], edge_attr], dim=-1))
        aggregate = nodes.new_zeros(nodes.shape)
        aggregate.index_add_(1, dst, messages)
        degree = nodes.new_zeros((nodes.shape[1], 1))
        degree.index_add_(0, dst, nodes.new_ones((dst.numel(), 1)))
        aggregate = aggregate / degree.clamp_min(1.0).view(1, -1, 1)
        pieces = [nodes, aggregate]
        if boundary is not None:
            pieces.append(boundary)
        updated = self.node_mlp(torch.cat(pieces, dim=-1))
        return self.norm(nodes + updated)


class EulerianGraphSurrogate(nn.Module):
    """Shared implementation for ``mgn_knn`` and ``sdf_gns`` variants."""

    def __init__(self, points: np.ndarray | torch.Tensor, *, model: str = "mgn_knn",
                 k: int = 16, hidden: int = 96, processors: int = 3,
                 history: int = 1) -> None:
        super().__init__()
        if model not in {"mgn_knn", "sdf_gns"}:
            raise ValueError("model must be mgn_knn or sdf_gns")
        if model == "mgn_knn" and history != 1:
            raise ValueError("mgn_knn uses the one-frame input contract")
        self.model_name, self.history = model, int(history)
        xyz = torch.as_tensor(np.asarray(points), dtype=torch.float32)
        self.register_buffer("points", xyz, persistent=False)
        self.register_buffer("edge_index", knn_edges(xyz, k=k), persistent=False)
        src, dst = self.edge_index
        edge_attr = torch.cat([xyz[src] - xyz[dst],
                               torch.linalg.vector_norm(xyz[src] - xyz[dst], dim=-1, keepdim=True)], dim=-1)
        self.register_buffer("edge_attr", edge_attr, persistent=False)
        node_input = 4 * self.history + 11 + 8
        self.encoder = MLP(node_input, hidden, hidden)
        boundary_dim = 8 if model == "sdf_gns" else 0
        self.processors = nn.ModuleList(
            GraphProcessor(hidden, edge_attr.shape[-1], boundary_dim) for _ in range(processors)
        )
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4))

    def forward(self, query_xyz: torch.Tensor, static_fx: torch.Tensor,
                previous_state: torch.Tensor, boundary_feat: torch.Tensor) -> torch.Tensor:
        if previous_state.ndim == 3:
            previous_state = previous_state.unsqueeze(1)
        if previous_state.ndim != 4 or previous_state.shape[-1] != 4:
            raise ValueError("previous_state must have [B,H,N,4] or [B,N,4]")
        if previous_state.shape[1] != self.history:
            raise ValueError(f"expected history={self.history}, got {previous_state.shape[1]}")
        if static_fx.shape[-1] != 11 or boundary_feat.shape[-1] != 8:
            raise ValueError("static_fx/boundary_feat dimensions must be 11/8")
        state_flat = previous_state.transpose(1, 2).reshape(previous_state.shape[0], previous_state.shape[2], -1)
        node_input = torch.cat([state_flat, static_fx, boundary_feat], dim=-1)
        nodes = self.encoder(node_input)
        edge_index = self.edge_index.to(nodes.device)
        edge_attr = self.edge_attr.to(nodes.device).unsqueeze(0).expand(nodes.shape[0], -1, -1)
        for processor in self.processors:
            boundary = boundary_feat if self.model_name == "sdf_gns" else None
            nodes = processor(nodes, edge_index, edge_attr, boundary)
        delta = self.decoder(nodes)
        return previous_state[:, 0] + delta


def build_graph_model(points: np.ndarray | torch.Tensor, model: str = "mgn_knn", **kwargs) -> EulerianGraphSurrogate:
    return EulerianGraphSurrogate(points, model=model, **kwargs)
