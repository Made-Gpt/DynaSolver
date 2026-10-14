from __future__ import annotations

import unittest
from argparse import Namespace

import torch

from models.ConditionOnlyTransolver import Model


def model_args(**overrides) -> Namespace:
    values = {
        "fun_dim": 11,
        "space_dim": 3,
        "state_dim": 4,
        "n_hidden": 32,
        "n_heads": 4,
        "n_layers": 2,
        "mlp_ratio": 2,
        "out_dim": 4,
        "slice_num": 4,
        "dropout": 0.0,
        "act": "gelu",
        "geotype": "unstructured",
        "shapelist": None,
        "checkpoint": False,
    }
    values.update(overrides)
    return Namespace(**values)


class ConditionOnlyTransolverTests(unittest.TestCase):
    def test_delta_depends_on_previous_state(self):
        """Backbone Δ uses Y_t via state_proj once head is unlocked."""
        torch.manual_seed(0)
        model = Model(model_args()).eval()
        # Un-zero state_proj + output_proj so Δ must change with Y_t.
        with torch.no_grad():
            model.state_proj.weight.normal_(0.0, 0.05)
            model.output_proj.weight.normal_(0.0, 0.05)
            model.output_proj.bias.normal_(0.0, 0.01)
        query = torch.randn(2, 9, 3)
        static = torch.randn(2, 9, 11)
        prev_a = torch.randn(2, 9, 4)
        prev_b = torch.randn(2, 9, 4)
        boundary = torch.randn(2, 9, 8)
        with torch.no_grad():
            d_a = model.predict_delta(query, static, prev_a)
            d_b = model.predict_delta(query, static, prev_b)
            out_a = model(query, static, prev_a, boundary)
            out_b = model(query, static, prev_b, boundary)
        self.assertFalse(torch.allclose(d_a, d_b))
        self.assertTrue(torch.allclose(out_a, prev_a + d_a))
        self.assertTrue(torch.allclose(out_b, prev_b + d_b))

    def test_zero_head_and_state_proj_starts_at_persistence(self):
        """Zero state_proj + output_proj → Ŷ = Y_t."""
        model = Model(model_args()).eval()
        query = torch.randn(1, 5, 3)
        static = torch.randn(1, 5, 11)
        prev = torch.randn(1, 5, 4)
        out = model(query, static, prev, torch.zeros(1, 5, 8))
        self.assertTrue(torch.allclose(out, prev))


if __name__ == "__main__":
    unittest.main()
