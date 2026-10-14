"""Numerical equivalence: legacy vs fast slice/deslice (same weights, FP32)."""

from __future__ import annotations

import copy
import unittest

import torch

from layers.Physics_Attention import Physics_Attention_Irregular_Mesh
from layers.Temporal_Physics_Block import TemporalPhysicsBlock


def _err_report(y_legacy: torch.Tensor, y_fast: torch.Tensor) -> dict[str, float]:
    diff = (y_legacy - y_fast).abs()
    denom = y_legacy.abs().amax().clamp_min(1.0e-12)
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "relative_error": float((diff.max() / denom).item()),
    }


def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(copy.deepcopy(src.state_dict()))


class FastSliceEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.dim = 256
        self.heads = 8
        self.dim_head = 32
        self.slice_num = 32
        self.batch = 2
        self.n_points = 128

    def _make_pair(self) -> tuple[Physics_Attention_Irregular_Mesh, Physics_Attention_Irregular_Mesh]:
        legacy = Physics_Attention_Irregular_Mesh(
            self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=0.0,
            slice_num=self.slice_num,
            slice_mode="legacy",
        ).eval()
        fast = Physics_Attention_Irregular_Mesh(
            self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=0.0,
            slice_num=self.slice_num,
            slice_mode="fast",
        ).eval()
        _copy_weights(legacy, fast)
        return legacy, fast

    def test_physics_attention_forward_fp32(self) -> None:
        legacy, fast = self._make_pair()
        x = torch.randn(self.batch, self.n_points, self.dim, dtype=torch.float32)
        with torch.no_grad():
            y_legacy = legacy(x)
            y_fast = fast(x)
        report = _err_report(y_legacy, y_fast)
        print("\n[Physics_Attention forward]", report)
        self.assertLess(report["max_abs_error"], 1.0e-5, msg=report)
        self.assertLess(report["relative_error"], 1.0e-5, msg=report)

    def test_slice_tokens_only_fp32(self) -> None:
        legacy, fast = self._make_pair()
        x = torch.randn(self.batch, self.n_points, self.dim, dtype=torch.float32)
        with torch.no_grad():
            w = legacy.compute_slice_weights(x)
            t_legacy = legacy.slice_to_tokens(x, w)
            t_fast = fast.slice_to_tokens(x, w)
        report = _err_report(t_legacy, t_fast)
        print("\n[slice_to_tokens]", report)
        self.assertLess(report["max_abs_error"], 1.0e-5, msg=report)

    def test_deslice_only_fp32(self) -> None:
        legacy, fast = self._make_pair()
        tokens = torch.randn(
            self.batch, self.heads, self.slice_num, self.dim_head, dtype=torch.float32
        )
        weights = torch.softmax(
            torch.randn(self.batch, self.heads, self.n_points, self.slice_num), dim=-1
        )
        with torch.no_grad():
            y_legacy = legacy.deslice_to_out(tokens, weights, use_bias=True, use_dropout=False)
            y_fast = fast.deslice_to_out(tokens, weights, use_bias=True, use_dropout=False)
        report = _err_report(y_legacy, y_fast)
        print("\n[deslice_to_out]", report)
        self.assertLess(report["max_abs_error"], 1.0e-5, msg=report)

    def test_deslice_delta_no_bias_fp32(self) -> None:
        legacy, fast = self._make_pair()
        tokens = torch.randn(
            self.batch, self.heads, self.slice_num, self.dim_head, dtype=torch.float32
        )
        weights = torch.softmax(
            torch.randn(self.batch, self.heads, self.n_points, self.slice_num), dim=-1
        )
        with torch.no_grad():
            y_legacy = legacy.deslice_to_out(tokens, weights, use_bias=False, use_dropout=False)
            y_fast = fast.deslice_to_out(tokens, weights, use_bias=False, use_dropout=False)
        report = _err_report(y_legacy, y_fast)
        print("\n[deslice_to_out no bias]", report)
        self.assertLess(report["max_abs_error"], 1.0e-5, msg=report)

    def test_temporal_pool_and_deslice_fp32(self) -> None:
        legacy_block = TemporalPhysicsBlock(
            hidden_dim=self.dim,
            num_heads=self.heads,
            slice_num=self.slice_num,
            dropout=0.0,
            use_long_memory=False,
            use_short_window=False,
            slice_mode="legacy",
        ).eval()
        fast_block = TemporalPhysicsBlock(
            hidden_dim=self.dim,
            num_heads=self.heads,
            slice_num=self.slice_num,
            dropout=0.0,
            use_long_memory=False,
            use_short_window=False,
            slice_mode="fast",
        ).eval()
        _copy_weights(legacy_block, fast_block)

        x = torch.randn(self.batch, self.n_points, self.dim, dtype=torch.float32)
        with torch.no_grad():
            tok_l, w_l = legacy_block._pool_slice_tokens(x)
            tok_f, w_f = fast_block._pool_slice_tokens(x)
            d_l = legacy_block._deslice_merged(tok_l, w_l)
            d_f = fast_block._deslice_merged(tok_f, w_f)

        w_report = _err_report(w_l, w_f)
        tok_report = _err_report(tok_l, tok_f)
        d_report = _err_report(d_l, d_f)
        print("\n[temporal weights]", w_report)
        print("[temporal tokens]", tok_report)
        print("[temporal deslice merged]", d_report)
        self.assertLess(w_report["max_abs_error"], 1.0e-6, msg=w_report)
        self.assertLess(tok_report["max_abs_error"], 1.0e-5, msg=tok_report)
        self.assertLess(d_report["max_abs_error"], 1.0e-5, msg=d_report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
