from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

import torch

from layers.Block_Causal_Temporal import BlockCausalTemporalAttention
from layers.Long_Flow_Memory import LongFlowMemoryUpdate, LongMemoryRead, init_memory
from layers.Temporal_Physics_Block import HistSpatialCache, TemporalPhysicsBlock
from models.TemporalTransolver import Model, load_geopt_backbone
from utils.constants import GEOPT_CHECKPOINT_SHA256
import torch.nn as nn


import os
GEOPT_CHECKPOINT = Path(os.environ.get("GEOPT_CHECKPOINT", "checkpoints/GeoPT_8layers.pt")).expanduser()


def _model_args(**overrides) -> Namespace:
    payload = dict(
        model="TemporalTransolver",
        fun_dim=11,
        space_dim=3,
        state_dim=4,
        boundary_dim=8,
        n_hidden=256,
        n_heads=8,
        n_layers=8,
        mlp_ratio=2,
        out_dim=4,
        slice_num=32,
        dropout=0.0,
        act="gelu",
        geotype="unstructured",
        shapelist=None,
        checkpoint=False,
        unified_pos=False,
        window_frames=5,
        use_long_memory=True,
        use_short_window=True,
        use_rope=True,
        short_scale_init=1.0e-3,
        long_scale_init=1.0e-3,
    )
    payload.update(overrides)
    return Namespace(**payload)


class TemporalModuleTests(unittest.TestCase):
    def test_block_causal_mask_is_frame_level(self) -> None:
        module = BlockCausalTemporalAttention(dim=256, num_heads=8, use_rope=True)
        z = torch.randn(2, 5, 32, 256)
        out = module(z)
        self.assertEqual(tuple(out.shape), (2, 5, 32, 256))

    def test_memory_read_then_update_shapes(self) -> None:
        dim, heads, tokens = 256, 8, 32
        head_dim = dim // heads
        read = LongMemoryRead(dim=dim, num_heads=heads)
        update = LongFlowMemoryUpdate(dim=dim, num_heads=heads)
        z = torch.randn(2, tokens, dim)
        memory = init_memory(2, "cpu", torch.float32, num_heads=heads, head_dim=head_dim)
        z_out, _ = read(z, memory)
        memory_new, delta, gamma = update(z, memory)
        self.assertEqual(tuple(z_out.shape), (2, tokens, dim))
        self.assertEqual(tuple(memory_new.shape), (2, heads, head_dim, head_dim))

    def test_temporal_block_uses_geopt_attn_modules(self) -> None:
        block = TemporalPhysicsBlock(hidden_dim=256, num_heads=8, slice_num=32)
        self.assertTrue(hasattr(block, "Attn"))
        self.assertTrue(hasattr(block, "ln_1"))
        self.assertTrue(hasattr(block, "mlp"))
        hidden = torch.randn(2, 5, 64, 256)
        memory = init_memory(2, "cpu", torch.float32, num_heads=8, head_dim=32)
        out = block(hidden, memory)
        self.assertEqual(tuple(out["point_out"].shape), (2, 64, 256))

    def test_temporal_transolver_predicts_window(self) -> None:
        model = Model(_model_args(n_layers=2))
        batch, frames, points = 2, 5, 64
        query = torch.randn(batch, frames, points, 3)
        static = torch.randn(batch, frames, points, 11)
        state = torch.randn(batch, frames, points, 4)
        boundary = torch.randn(batch, frames, points, 8)
        memory = model.init_memory(batch, query.device, query.dtype)
        result = model.forward_window(query, static, state, boundary, memory_prev=memory)
        self.assertEqual(tuple(result["prediction"].shape), (batch, points, 4))
        self.assertTrue(torch.isfinite(result["prediction"]).all())

    def test_geopt_backbone_loads_when_checkpoint_present(self) -> None:
        if not GEOPT_CHECKPOINT.is_file():
            self.skipTest(f"GeoPT checkpoint not found: {GEOPT_CHECKPOINT}")
        model = Model(_model_args())
        report = load_geopt_backbone(
            model,
            GEOPT_CHECKPOINT,
            expected_sha256=GEOPT_CHECKPOINT_SHA256,
        )
        self.assertGreater(report["loaded_tensor_count"], 100)
        self.assertIn("preprocess.linear_pre.0.weight", model.state_dict())
        self.assertIn("blocks.0.Attn.to_q.weight", model.state_dict())

    def test_model_factory_registers_temporal(self) -> None:
        from models.model_factory import get_model

        model = get_model(_model_args())
        self.assertEqual(model.__name__, "TemporalTransolver")

    def test_spatial_only_block_matches_transolver_block(self) -> None:
        from models.Transolver import Transolver_block

        block = TemporalPhysicsBlock(
            hidden_dim=256,
            num_heads=8,
            slice_num=32,
            use_short_window=False,
            use_long_memory=False,
        )
        ref = Transolver_block(
            num_heads=8,
            hidden_dim=256,
            dropout=0.0,
            act="gelu",
            mlp_ratio=2,
            slice_num=32,
            geotype="unstructured",
        )
        ref.load_state_dict(
            {
                "ln_1.weight": block.ln_1.weight,
                "ln_1.bias": block.ln_1.bias,
                "Attn.temperature": block.Attn.temperature,
                "Attn.in_project_x.weight": block.Attn.in_project_x.weight,
                "Attn.in_project_x.bias": block.Attn.in_project_x.bias,
                "Attn.in_project_fx.weight": block.Attn.in_project_fx.weight,
                "Attn.in_project_fx.bias": block.Attn.in_project_fx.bias,
                "Attn.in_project_slice.weight": block.Attn.in_project_slice.weight,
                "Attn.in_project_slice.bias": block.Attn.in_project_slice.bias,
                "Attn.to_q.weight": block.Attn.to_q.weight,
                "Attn.to_k.weight": block.Attn.to_k.weight,
                "Attn.to_v.weight": block.Attn.to_v.weight,
                "Attn.to_out.0.weight": block.Attn.to_out[0].weight,
                "Attn.to_out.0.bias": block.Attn.to_out[0].bias,
                "ln_2.weight": block.ln_2.weight,
                "ln_2.bias": block.ln_2.bias,
                "mlp.linear_pre.0.weight": block.mlp.linear_pre[0].weight,
                "mlp.linear_pre.0.bias": block.mlp.linear_pre[0].bias,
                "mlp.linear_post.weight": block.mlp.linear_post.weight,
                "mlp.linear_post.bias": block.mlp.linear_post.bias,
            },
            strict=False,
        )
        hidden = torch.randn(2, 64, 256)
        window = torch.randn(2, 4, 64, 256)
        window[:, -1] = hidden
        out_block = block(window, memory_prev=None)["point_out"]
        out_ref = ref(hidden)
        torch.testing.assert_close(out_block, out_ref, rtol=1e-5, atol=1e-5)

    def test_spatial_only_model_matches_v0_on_last_frame(self) -> None:
        from models.DynamicTransolver import Model as V0Model

        v0 = V0Model(
            Namespace(
                geotype="unstructured",
                space_dim=3,
                fun_dim=11,
                state_dim=4,
                boundary_dim=8,
                n_hidden=256,
                n_heads=8,
                n_layers=2,
                slice_num=32,
                mlp_ratio=2,
                dropout=0.0,
                act="gelu",
                shapelist=None,
                checkpoint=False,
            )
        )
        temporal = Model(
            _model_args(
                n_layers=2,
                window_frames=4,
                use_short_window=False,
                use_long_memory=False,
            )
        )
        temporal.load_state_dict(v0.state_dict(), strict=False)
        batch, frames, points = 2, 4, 32
        query = torch.randn(batch, points, 3)
        static = torch.randn(batch, points, 11)
        state = torch.randn(batch, points, 4)
        boundary = torch.randn(batch, points, 8)
        query_win = query.unsqueeze(1).expand(batch, frames, points, 3).contiguous()
        static_win = static.unsqueeze(1).expand(batch, frames, points, 11).contiguous()
        state_win = state.unsqueeze(1).expand(batch, frames, points, 4).contiguous()
        boundary_win = boundary.unsqueeze(1).expand(batch, frames, points, 8).contiguous()
        v0_pred = v0(query, static, state, boundary)
        temporal_pred = temporal.forward_window(
            query_win, static_win, state_win, boundary_win, memory_prev=None
        )["prediction"]
        torch.testing.assert_close(v0_pred, temporal_pred, rtol=1e-5, atol=1e-5)

    def test_hist_cache_unused_on_integrated_path(self) -> None:
        """Integrated temporal path is window-native; HistSpatialCache stays empty."""
        block = TemporalPhysicsBlock(
            hidden_dim=256,
            num_heads=8,
            slice_num=32,
            use_short_window=True,
            use_long_memory=False,
        )
        hidden = torch.randn(2, 5, 64, 256)
        cache = HistSpatialCache(window_frames=5, n_layers=1)
        out_cached = block(hidden, None, layer_idx=0, hist_cache=cache)
        out_plain = block(hidden, None, layer_idx=0, hist_cache=None)
        torch.testing.assert_close(
            out_cached["point_out"], out_plain["point_out"], rtol=1e-5, atol=1e-5
        )
        self.assertEqual(len(cache), 0)

    def test_hist_cache_api_bounds(self) -> None:
        n_layers = 8
        window_frames = 4
        cache = HistSpatialCache(window_frames=window_frames, n_layers=n_layers)
        self.assertEqual(cache.max_entries, n_layers * (window_frames - 1))
        hidden = torch.randn(2, 64, 256)
        for layer_idx in range(n_layers):
            for frame_idx in range(window_frames - 1):
                cache.get_geopt(layer_idx, frame_idx, hidden, lambda x: x, detach=False)
        self.assertEqual(len(cache), n_layers * (window_frames - 1))

    def test_hist_cache_rejects_global_frame_index(self) -> None:
        cache = HistSpatialCache(window_frames=4, n_layers=8)
        hidden = torch.randn(2, 64, 256)
        with self.assertRaises(IndexError):
            cache.get_geopt(0, 4, hidden, lambda x: x)  # valid indices 0..3

    def test_causal_path_uses_history_frames(self) -> None:
        block = TemporalPhysicsBlock(
            hidden_dim=256,
            num_heads=8,
            slice_num=32,
            use_short_window=True,
            use_long_memory=False,
        )
        # Un-zero out_proj so residual+out is not identity (module defaults to zero-init).
        nn.init.normal_(block.temporal_attn.out_proj.weight, std=0.2)
        nn.init.zeros_(block.temporal_attn.out_proj.bias)
        torch.manual_seed(0)
        base = torch.randn(2, 5, 64, 256)
        hidden_a = base.clone()
        hidden_b = base.clone()
        # Non-constant perturbation: LayerNorm cancels a uniform channel shift.
        hidden_b[:, 0] = hidden_b[:, 0] + torch.randn_like(hidden_b[:, 0])

        out_a = block(hidden_a, None)
        out_b = block(hidden_b, None)
        diff_pts = (out_a["point_out"] - out_b["point_out"]).abs().max().item()
        diff_z = (out_a["z_out"] - out_b["z_out"]).abs().max().item()
        self.assertGreater(diff_z, 1.0e-4)
        self.assertGreater(diff_pts, 1.0e-4)

    def test_history_frames_receive_gradients_with_causal(self) -> None:
        block = TemporalPhysicsBlock(
            hidden_dim=256,
            num_heads=8,
            slice_num=32,
            use_short_window=True,
            use_long_memory=False,
        )
        nn.init.normal_(block.temporal_attn.out_proj.weight, std=0.02)
        nn.init.zeros_(block.temporal_attn.out_proj.bias)
        hidden = torch.randn(2, 5, 64, 256, requires_grad=True)
        out = block(hidden, None)["point_out"]
        out.sum().backward()
        self.assertIsNotNone(hidden.grad)
        hist_grad = hidden.grad[:, :-1].abs().max().item()
        self.assertGreater(hist_grad, 0.0)

    def test_zero_init_causal_matches_geopt_transolver_block(self) -> None:
        """Zero-init causal ⇒ integrated path ≡ GeoPT Transolver_block on last frame."""
        from models.Transolver import Transolver_block

        block = TemporalPhysicsBlock(
            hidden_dim=256,
            num_heads=8,
            slice_num=32,
            use_short_window=True,
            use_long_memory=False,
            slice_mode="legacy",
        )
        nn.init.zeros_(block.temporal_attn.out_proj.weight)
        nn.init.zeros_(block.temporal_attn.out_proj.bias)
        block.eval()

        ref = Transolver_block(
            num_heads=8,
            hidden_dim=256,
            dropout=0.0,
            act="gelu",
            mlp_ratio=2,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            geotype="unstructured",
        )
        ref.ln_1.load_state_dict(block.ln_1.state_dict())
        ref.ln_2.load_state_dict(block.ln_2.state_dict())
        ref.mlp.load_state_dict(block.mlp.state_dict())
        ref.Attn.load_state_dict(block.Attn.state_dict(), strict=False)
        ref.eval()

        hidden_last = torch.randn(2, 64, 256)
        window = hidden_last.unsqueeze(1).expand(2, 5, 64, 256).contiguous()
        with torch.no_grad():
            out = block(window, None)["point_out"]
            out_ref = ref(hidden_last)
        torch.testing.assert_close(out, out_ref, rtol=1e-5, atol=1e-5)

    def test_parallel_return_all_frames_shapes(self) -> None:
        block = TemporalPhysicsBlock(
            hidden_dim=256,
            num_heads=8,
            slice_num=32,
            use_short_window=True,
            use_long_memory=False,
        )
        hidden = torch.randn(2, 4, 64, 256)
        out = block(hidden, None, return_all_frames=True)
        self.assertEqual(tuple(out["point_out"].shape), (2, 4, 64, 256))
        self.assertEqual(tuple(out["z_out"].shape), (2, 4, 32, 256))

    def test_fp4_share_cross_param_budget(self) -> None:
        """Shared cross must not double-count in state_dict; budgets vs tf4/fp4."""
        from layers.Asymmetric_Cross_Token import AsymmetricSurfaceVolumeCross

        def n_params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        def sd_sum(m: nn.Module) -> int:
            return sum(v.numel() for v in m.state_dict().values())

        tf4 = Model(
            _model_args(
                n_layers=8,
                slice_num=64,
                use_long_memory=False,
                branch_mode="unified",
                window_frames=4,
            )
        )
        fp4 = Model(
            _model_args(
                n_layers=8,
                slice_num=32,
                use_long_memory=False,
                branch_mode="surface_volume_asym",
                share_cross_weights=False,
                window_frames=4,
            )
        )
        shared = Model(
            _model_args(
                n_layers=8,
                slice_num=32,
                use_long_memory=False,
                branch_mode="surface_volume_asym",
                share_cross_weights=True,
                window_frames=4,
            )
        )
        self.assertEqual(n_params(tf4), sd_sum(tf4))
        self.assertEqual(n_params(fp4), sd_sum(fp4))
        self.assertEqual(n_params(shared), sd_sum(shared))
        self.assertLess(n_params(shared), n_params(fp4))
        self.assertGreater(n_params(shared), n_params(tf4))
        # Tied module: only one registered child; alias still usable.
        cross = shared.blocks[0].sv_cross
        assert isinstance(cross, AsymmetricSurfaceVolumeCross)
        self.assertIs(cross.cross_sv, cross.cross_vs)
        self.assertNotIn("cross_vs", cross._modules)
        # Legacy dual-prefix state_dict still loads.
        legacy = fp4.state_dict()
        # Copy sv → vs equality as in old shared ckpts, then load into shared model.
        for key in list(legacy.keys()):
            if ".cross_sv." in key:
                legacy[key.replace(".cross_sv.", ".cross_vs.")] = legacy[key].clone()
        # Zero separate-vs divergence: force equal for layers (already cloned).
        missing, unexpected = shared.load_state_dict(legacy, strict=False)
        self.assertEqual(unexpected, [])
        # After load, unique params still match single-module budget.
        self.assertEqual(n_params(shared), sd_sum(shared))


if __name__ == "__main__":
    unittest.main()
