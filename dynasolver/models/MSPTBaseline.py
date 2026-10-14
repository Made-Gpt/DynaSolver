"""Residual MSPT baseline (setting A): fx = [Y_t, static_fx].

Vendored official MSPT under ``vendor/mspt`` (isolated copy). Boundary is
accepted for trainer API compatibility but unused. Prediction is residual:
Ŷ_{t+1} = Y_t + MSPT(xyz, [Y_t, static_fx]).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_MSPT = REPO_ROOT / "vendor" / "mspt"
VENDOR_MSPT_LAYERS = VENDOR_MSPT / "benchmarks" / "Neural-Solver-Library-MSPT"


def _prepend(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def build_mspt_backbone(
    *,
    input_dim: int = 15,
    output_dim: int = 4,
    n_hidden: int = 256,
    n_layers: int = 5,
    n_heads: int = 8,
    supernodes: int = 32,
    dropout: float = 0.0,
) -> nn.Module:
    """Build official unstructured MSPT from the local vendor copy."""
    if not VENDOR_MSPT.is_dir():
        raise FileNotFoundError(f"missing isolated MSPT vendor at {VENDOR_MSPT}")
    _prepend(VENDOR_MSPT)
    _prepend(VENDOR_MSPT_LAYERS)
    module = importlib.import_module("MSPT")
    return module.MSPTUnstructured(
        pos_dim=3,
        fx_dim=input_dim,
        out_dim=output_dim,
        num_blocks=n_layers,
        n_hidden=n_hidden,
        num_heads=n_heads,
        V=supernodes,
        Q=1,
        dropout=dropout,
        chunking_mode="linear",
        unified_pos=False,
        Time_Input=False,
    )


class Model(nn.Module):
    """Residual MSPT with Transolver-matching volume inputs (no boundary)."""

    def __init__(self, args):
        super().__init__()
        space_dim = int(getattr(args, "space_dim", 3))
        fun_dim = int(getattr(args, "fun_dim", 11))
        state_dim = int(getattr(args, "state_dim", 4))
        if space_dim != 3 or fun_dim != 11:
            raise ValueError("MSPTBaseline requires space_dim == 3 and fun_dim == 11")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        self.__name__ = "MSPTBaseline"
        self.args = args
        self.state_dim = state_dim
        self.fun_dim = fun_dim
        self.feature_dim = state_dim + fun_dim  # setting A: [Y_t, static_fx]

        self.backbone = build_mspt_backbone(
            input_dim=self.feature_dim,
            output_dim=state_dim,
            n_hidden=int(getattr(args, "n_hidden", 256)),
            n_layers=int(getattr(args, "n_layers", 5)),
            n_heads=int(getattr(args, "n_heads", 8)),
            supernodes=int(getattr(args, "supernodes", 32)),
            dropout=float(getattr(args, "dropout", 0.0)),
        )
        # Zero-init residual head so step-0 ≈ persistence (matches ConditionOnly).
        last = self.backbone.blocks[-1]
        if hasattr(last, "mlp2"):
            nn.init.zeros_(last.mlp2.weight)
            nn.init.zeros_(last.mlp2.bias)

    def pack_features(
        self,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        if previous_state.ndim != 3 or static_fx.ndim != 3:
            raise ValueError("static_fx and previous_state must be [B, N, C]")
        if previous_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"previous_state last dim must be {self.state_dim}, got {previous_state.shape[-1]}"
            )
        if static_fx.shape[-1] != self.fun_dim:
            raise ValueError(
                f"static_fx last dim must be {self.fun_dim}, got {static_fx.shape[-1]}"
            )
        return torch.cat((previous_state, static_fx), dim=-1)

    def predict_delta(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        if query_xyz.ndim != 3 or query_xyz.shape[-1] != 3:
            raise ValueError(f"query_xyz must be [B,N,3], got {tuple(query_xyz.shape)}")
        features = self.pack_features(static_fx, previous_state)
        return self.backbone(query_xyz, features)

    def forward(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        del boundary_feat
        return previous_state + self.predict_delta(query_xyz, static_fx, previous_state)
