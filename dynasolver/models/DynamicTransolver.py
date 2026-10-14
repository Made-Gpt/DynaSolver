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
    """V0 dynamic Transolver with an unchanged GeoPT spatial backbone."""

    def __init__(self, args):
        super().__init__()
        if args.geotype != "unstructured":
            raise ValueError("DynamicTransolver V0 currently supports only unstructured point sets")
        if args.space_dim != 3 or args.fun_dim != 11:
            raise ValueError("DynamicTransolver V0 requires space_dim == 3 and fun_dim == 11")
        if int(args.state_dim) <= 0:
            raise ValueError("state_dim must be positive")
        if int(args.boundary_dim) <= 0:
            raise ValueError("boundary_dim must be positive")

        self.__name__ = "DynamicTransolverV0"
        self.args = args
        self.state_dim = int(args.state_dim)
        self.boundary_dim = int(args.boundary_dim)
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
        self.boundary_proj = nn.Linear(self.boundary_dim, stem_dim, bias=False)

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
        nn.init.zeros_(self.boundary_proj.weight)
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

    def _validate_inputs(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> None:
        if query_xyz.ndim != 3:
            raise ValueError("query_xyz must have shape [B, N, 3]")
        batch_points = query_xyz.shape[:2]
        expected = {
            "query_xyz": 3,
            "static_fx": int(self.args.fun_dim),
            "previous_state": self.state_dim,
            "boundary_feat": self.boundary_dim,
        }
        tensors = {
            "query_xyz": query_xyz,
            "static_fx": static_fx,
            "previous_state": previous_state,
            "boundary_feat": boundary_feat,
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 3 or tensor.shape[:2] != batch_points or tensor.shape[-1] != expected[name]:
                raise ValueError(
                    f"{name} must have shape [B, N, {expected[name]}] with matching B and N; "
                    f"got {tuple(tensor.shape)}"
                )

    def embed_inputs(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(query_xyz, static_fx, previous_state, boundary_feat)
        base_input = torch.cat((query_xyz, static_fx), dim=-1)

        base_linear = self.preprocess.linear_pre[0]
        activation = self.preprocess.linear_pre[1]
        preactivation = base_linear(base_input)
        preactivation = preactivation + self.state_proj(previous_state)
        preactivation = preactivation + self.boundary_proj(boundary_feat)
        hidden = self.preprocess.linear_post(activation(preactivation))
        return hidden + self.placeholder[None, None, :]

    def encode(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embed_inputs(query_xyz, static_fx, previous_state, boundary_feat)
        for block in self.blocks:
            if self.args.checkpoint and self.training:
                hidden = checkpoint.checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return hidden

    def predict_delta(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode(query_xyz, static_fx, previous_state, boundary_feat)
        return self.output_proj(self.output_norm(hidden))

    def forward(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        previous_state: torch.Tensor,
        boundary_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one step; previous_state must already be aligned to query_xyz."""
        return previous_state + self.predict_delta(query_xyz, static_fx, previous_state, boundary_feat)


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
    """Strictly transplant the GeoPT spatial backbone and reject schema drift."""
    checkpoint_path = Path(checkpoint_path)
    actual_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"GeoPT checkpoint SHA256 mismatch: {actual_sha256} != {expected_sha256}"
        )

    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(source, Mapping) or not all(isinstance(value, torch.Tensor) for value in source.values()):
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
                f"GeoPT tensor shape mismatch for {key}: {tuple(source[key].shape)} != {tuple(target[key].shape)}"
            )

    load_result = model.load_state_dict(
        {key: source[key] for key in target_backbone_keys},
        strict=False,
    )
    expected_missing = set(target) - target_backbone_keys
    if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected dynamic-model load result: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    loaded_state = model.state_dict()
    unequal = [key for key in sorted(target_backbone_keys) if not torch.equal(loaded_state[key].cpu(), source[key].cpu())]
    if unequal:
        raise RuntimeError(f"GeoPT tensors changed during load: {unequal}")

    return {
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "source_tensor_count": len(source),
        "loaded_tensor_count": len(target_backbone_keys),
        "loaded_parameter_elements": sum(source[key].numel() for key in target_backbone_keys),
        "excluded_keys": sorted(legacy_head_keys),
        "new_keys": sorted(expected_missing),
    }
