#!/usr/bin/env python3
"""Smoke: MSPT residual setting A matches Transolver input contract."""

from __future__ import annotations

import unittest
from argparse import Namespace

import torch

from models.MSPTBaseline import Model


class MSPTBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = Namespace(
            space_dim=3,
            fun_dim=11,
            state_dim=4,
            n_hidden=64,
            n_heads=4,
            n_layers=2,
            supernodes=8,
            dropout=0.0,
            act="gelu",
            geotype="unstructured",
        )
        self.model = Model(self.args)

    def test_feature_dim_setting_a(self) -> None:
        self.assertEqual(self.model.feature_dim, 15)

    def test_forward_residual_near_persistence_at_init(self) -> None:
        b, n = 2, 64
        xyz = torch.randn(b, n, 3)
        static_fx = torch.randn(b, n, 11)
        y_t = torch.randn(b, n, 4)
        boundary = torch.randn(b, n, 8)
        pred = self.model(xyz, static_fx, y_t, boundary)
        self.assertEqual(tuple(pred.shape), (b, n, 4))
        # Zero-init head → pred ≈ Y_t at step 0.
        self.assertTrue(torch.allclose(pred, y_t, atol=1e-5, rtol=1e-5))

    def test_backward(self) -> None:
        b, n = 1, 32
        xyz = torch.randn(b, n, 3, requires_grad=True)
        static_fx = torch.randn(b, n, 11)
        y_t = torch.randn(b, n, 4)
        boundary = torch.zeros(b, n, 8)
        pred = self.model(xyz, static_fx, y_t, boundary)
        pred.sum().backward()
        self.assertIsNotNone(xyz.grad)


if __name__ == "__main__":
    unittest.main()
