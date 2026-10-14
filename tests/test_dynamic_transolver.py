from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from models.DynamicTransolver import Model as DynamicModel
from models.DynamicTransolver import load_geopt_backbone, sha256_file
from models.Transolver import Model as LegacyModel
from models.model_factory import get_model


def model_args(**overrides) -> Namespace:
    values = {
        "fun_dim": 11,
        "space_dim": 3,
        "state_dim": 6,
        "boundary_dim": 8,
        "n_hidden": 32,
        "n_heads": 4,
        "n_layers": 2,
        "mlp_ratio": 2,
        "out_dim": 9,
        "slice_num": 4,
        "dropout": 0.0,
        "act": "gelu",
        "geotype": "unstructured",
        "shapelist": None,
        "checkpoint": False,
        "unified_pos": False,
    }
    values.update(overrides)
    return Namespace(**values)


def inputs(batch: int = 2, points: int = 11):
    return (
        torch.randn(batch, points, 3),
        torch.randn(batch, points, 11),
        torch.randn(batch, points, 6),
        torch.randn(batch, points, 8),
    )


class DynamicTransolverTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_initialized_model_starts_from_persistence(self):
        model = DynamicModel(model_args()).eval()
        query_xyz, static_fx, previous_state, boundary_feat = inputs()
        prediction = model(query_xyz, static_fx, previous_state, boundary_feat)
        self.assertTrue(torch.equal(prediction, previous_state))

    def test_zero_adapters_preserve_legacy_stem_embedding(self):
        model = DynamicModel(model_args()).eval()
        query_xyz, static_fx, previous_state, boundary_feat = inputs()
        dynamic_embedding = model.embed_inputs(query_xyz, static_fx, previous_state, boundary_feat)
        legacy_embedding = model.preprocess(torch.cat((query_xyz, static_fx), dim=-1))
        legacy_embedding = legacy_embedding + model.placeholder[None, None, :]
        self.assertTrue(torch.equal(dynamic_embedding, legacy_embedding))

    def test_output_is_permutation_equivariant(self):
        model = DynamicModel(model_args()).eval()
        torch.nn.init.normal_(model.output_proj.weight, std=0.01)
        query_xyz, static_fx, previous_state, boundary_feat = inputs(batch=1, points=13)
        permutation = torch.randperm(query_xyz.shape[1])

        direct = model(query_xyz, static_fx, previous_state, boundary_feat)
        permuted = model(
            query_xyz[:, permutation],
            static_fx[:, permutation],
            previous_state[:, permutation],
            boundary_feat[:, permutation],
        )
        self.assertTrue(torch.allclose(permuted, direct[:, permutation], atol=1e-6, rtol=1e-5))

    def test_dynamic_paths_receive_gradients_after_head_is_active(self):
        model = DynamicModel(model_args()).train()
        torch.nn.init.normal_(model.output_proj.weight, std=0.01)
        query_xyz, static_fx, previous_state, boundary_feat = inputs(batch=1)
        target = torch.randn_like(previous_state)
        loss = torch.mean((model(query_xyz, static_fx, previous_state, boundary_feat) - target) ** 2)
        loss.backward()
        self.assertGreater(float(model.state_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.boundary_proj.weight.grad.abs().sum()), 0.0)

    def test_checkpoint_loader_transplants_exact_backbone(self):
        legacy = LegacyModel(model_args())
        dynamic = DynamicModel(model_args())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(legacy.state_dict(), path)
            digest = sha256_file(path)
            manifest = load_geopt_backbone(dynamic, path, expected_sha256=digest)

        source = legacy.state_dict()
        target = dynamic.state_dict()
        for key in source:
            if "ln_3" not in key and "mlp2" not in key:
                self.assertTrue(torch.equal(target[key], source[key]), key)
        self.assertEqual(manifest["loaded_tensor_count"], len(source) - 4)
        self.assertEqual(manifest["source_tensor_count"], len(source))

    def test_checkpoint_loader_rejects_wrong_digest(self):
        legacy = LegacyModel(model_args())
        dynamic = DynamicModel(model_args())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(legacy.state_dict(), path)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                load_geopt_backbone(dynamic, path, expected_sha256="0" * 64)

    def test_input_shape_mismatch_is_rejected(self):
        model = DynamicModel(model_args())
        query_xyz, static_fx, previous_state, boundary_feat = inputs()
        with self.assertRaisesRegex(ValueError, "boundary_feat"):
            model(query_xyz, static_fx, previous_state, boundary_feat[..., :-1])

    def test_constructor_requires_exact_static_feature_contract(self):
        with self.assertRaisesRegex(ValueError, "space_dim == 3 and fun_dim == 11"):
            DynamicModel(model_args(space_dim=4, fun_dim=10))

    def test_constructor_requires_positive_dynamic_dimensions(self):
        for name in ("state_dim", "boundary_dim"):
            for value in (0, -1):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"{name} must be positive"):
                        DynamicModel(model_args(**{name: value}))

    def test_model_factory_constructs_dynamic_transolver(self):
        args = model_args(model="DynamicTransolver")
        model = get_model(args)
        self.assertIsInstance(model, DynamicModel)


if __name__ == "__main__":
    unittest.main()
