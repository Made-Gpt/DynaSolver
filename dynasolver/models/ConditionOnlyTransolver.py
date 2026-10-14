from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_

from models.Transolver import MLP, Transolver_block


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Model(nn.Module):
    """Residual Transolver with geometry + Y_t (no boundary).

    Stem matches V0 for the GeoPT path: preprocess(query_xyz, static_fx) plus a
    zero-init ``state_proj(Y_t)`` into the pre-activation. Boundary is accepted
    for API compatibility with the baseline trainer but is not used.
    Prediction is residual: Ŷ_{t+1} = Y_t + Δ(X^g, Y_t).
    """

    def __init__(self, args):
        super().__init__()
        if args.geotype != "unstructured":
            raise ValueError("ConditionOnlyTransolver supports only unstructured point sets")
        if args.space_dim != 3 or args.fun_dim != 11:
            raise ValueError("ConditionOnlyTransolver requires space_dim == 3 and fun_dim == 11")
        if int(args.state_dim) <= 0:
            raise ValueError("state_dim must be positive")

        self.__name__ = "ConditionOnlyTransolver"
        self.args = args
        self.state_dim = int(args.state_dim)
        hidden_dim = int(args.n_hidden)
        stem_dim = hidden_dim * 2

        self.preprocess = MLP(
            args.fun_dim + args.space_dim,
            stem_dim,
            hidden_dim,
            n_layers=0,
            res=False,
            act=args.act,
        )
        self.state_proj = nn.Linear(self.state_dim, stem_dim, bias=False)
        self.blocks = nn.ModuleList(
            [
                Transolver_block(
                    num_heads=args.n_heads,
                    hidden_dim=hidden_dim,
                    dropout=args.dropout,
                    act=args.act,
                    mlp_ratio=args.mlp_ratio,
                    out_dim=self.state_dim,
                    slice_num=args.slice_num,
                    last_layer=False,
                    geotype=args.geotype,
                    shapelist=args.shapelist,
                )
                for _ in range(args.n_layers)
            ]
        )
        self.placeholder = nn.Parameter(torch.rand(hidden_dim) / hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.state_dim)

        self.apply(self._init_weights)
        nn.init.zeros_(self.state_proj.weight)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def embed_inputs(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        if query_xyz.ndim != 3 or static_fx.ndim != 3 or previous_state.ndim != 3:
            raise ValueError("query_xyz, static_fx, previous_state must be [B, N, C]")
        if previous_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"previous_state last dim must be {self.state_dim}, got {previous_state.shape[-1]}"
            )
        base_input = torch.cat((query_xyz, static_fx), dim=-1)
        base_linear = self.preprocess.linear_pre[0]
        activation = self.preprocess.linear_pre[1]
        preactivation = base_linear(base_input) + self.state_proj(previous_state)
        hidden = self.preprocess.linear_post(activation(preactivation))
        return hidden + self.placeholder[None, None, :]

    def predict_delta(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> torch.Tensor:
        """Δ from geometry/condition + Y_t (boundary unused)."""
        hidden = self.embed_inputs(query_xyz, static_fx, previous_state)
        for block in self.blocks:
            if self.args.checkpoint and self.training:
                hidden = checkpoint.checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.output_proj(self.output_norm(hidden))

    def forward(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        del boundary_feat
        return previous_state + self.predict_delta(query_xyz, static_fx, previous_state)


def _backbone_keys(state: Mapping[str, torch.Tensor]) -> set[str]:
    return {
        key
        for key in state
        if key == "placeholder" or key.startswith("preprocess.") or key.startswith("blocks.")
    }


def _legacy_head_keys(model: Model) -> set[str]:
    last_block = len(model.blocks) - 1
    prefix = f"blocks.{last_block}"
    return {
        f"{prefix}.ln_3.weight",
        f"{prefix}.ln_3.bias",
        f"{prefix}.mlp2.weight",
        f"{prefix}.mlp2.bias",
    }


def load_geopt_backbone(
    model: Model,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
) -> dict:
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
    target_backbone_keys = _backbone_keys(target)
    legacy_head_keys = _legacy_head_keys(model)
    source_keys = set(source)
    source_backbone_keys = source_keys - legacy_head_keys

    if source_backbone_keys != target_backbone_keys:
        missing = sorted(target_backbone_keys - source_backbone_keys)
        unexpected = sorted(source_backbone_keys - target_backbone_keys)
        raise RuntimeError(
            f"GeoPT backbone key mismatch; missing={missing}, unexpected={unexpected}"
        )
    if not legacy_head_keys.issubset(source_keys):
        missing_head = sorted(legacy_head_keys - source_keys)
        raise RuntimeError(f"GeoPT legacy task head is incomplete: {missing_head}")

    for key in sorted(target_backbone_keys):
        if source[key].shape != target[key].shape:
            raise RuntimeError(
                f"GeoPT tensor shape mismatch for {key}: "
                f"{tuple(source[key].shape)} != {tuple(target[key].shape)}"
            )

    load_result = model.load_state_dict(
        {key: source[key] for key in target_backbone_keys},
        strict=False,
    )
    expected_missing = set(target) - target_backbone_keys
    if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected condition-only load result: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    return {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "loaded_tensor_count": len(target_backbone_keys),
        "new_keys": sorted(expected_missing),
    }
