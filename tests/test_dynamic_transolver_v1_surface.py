from __future__ import annotations

import unittest
from argparse import Namespace

import torch

from models.DynamicTransolverV1Pre import Model as V1Pre
from models.DynamicTransolverV1_1 import Model as V11


def args(**overrides):
    values = dict(
        geotype="unstructured", space_dim=3, fun_dim=11, state_dim=4,
        boundary_dim=8, n_hidden=32, hidden_dim=32, n_heads=4,
        n_layers=2, slice_num=4, mlp_ratio=2, dropout=0.0,
        act="gelu", shapelist=None, checkpoint=False,
        input_feature_dim=23, interaction_groups=1,
    )
    values.update(overrides)
    return Namespace(**values)


def inputs(batch=2, points=13, surface=19):
    query = torch.randn(batch, points, 3)
    static = torch.randn(batch, points, 11)
    static[..., 7:10] = torch.nn.functional.normalize(static[..., 7:10], dim=-1)
    static[..., 10] = static[..., 10].abs()
    previous = torch.randn(batch, points, 4)
    boundary = torch.randn(batch, points, 8)
    surface_xyz = torch.randn(batch, surface, 3)
    surface_normal = torch.nn.functional.normalize(torch.randn(batch, surface, 3), dim=-1)
    surface_state = torch.randn(batch, surface, 4)
    surface_mask = torch.ones(batch, surface, dtype=torch.bool)
    surface_mask[:, -2:] = False
    return query, static, previous, boundary, surface_xyz, surface_normal, surface_state, surface_mask


class V1SurfaceTests(unittest.TestCase):
    def test_v1_pre_feature_contract_and_persistence(self):
        model = V1Pre(args()).eval()
        query, static, previous, boundary, *_ = inputs()
        features = model._features(query, static, previous, boundary)
        self.assertEqual(tuple(features.shape), (2, 13, 23))
        self.assertTrue(torch.equal(model(query, static, previous, boundary), previous))

    def test_v11_shapes_persistence_and_cross_gradients(self):
        model = V11(args()).train()
        query, static, previous, boundary, sx, sn, ss, sm = inputs()
        output = model(query, static, previous, boundary, sx, sn, ss, sm)
        self.assertEqual(tuple(output.shape), tuple(previous.shape))
        self.assertTrue(torch.equal(output, previous))
        torch.nn.init.normal_(model.output_proj.weight, std=0.01)
        loss = model(query, static, previous, boundary, sx, sn, ss, sm).square().mean()
        loss.backward()
        self.assertIsNotNone(model.cross_blocks[0].cross.in_proj_weight.grad)
        self.assertTrue(torch.isfinite(model.cross_blocks[0].cross.in_proj_weight.grad).all())

    def test_surface_mask_does_not_change_fixed_valid_surface_tokens(self):
        model = V11(args()).eval()
        query, static, previous, boundary, sx, sn, ss, sm = inputs(batch=1)
        with torch.no_grad():
            hidden = model.surface_input(model._surface_features(sx, sn, ss, static, boundary))
            tokens_a, _, valid_a = model.surface_projector.tokenize(hidden, sm)
            sx[:, -2:] = 1000.0
            sn[:, -2:] = 0.0
            ss[:, -2:] = 1000.0
            hidden_b = model.surface_input(model._surface_features(sx, sn, ss, static, boundary))
            tokens_b, _, valid_b = model.surface_projector.tokenize(hidden_b, sm)
        self.assertTrue(torch.allclose(tokens_a, tokens_b, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(valid_a, valid_b))


if __name__ == "__main__":
    unittest.main()
