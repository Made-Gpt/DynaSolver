"""Smoke tests for radius MGN / GNS surrogates."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from models.radius_mgn_gns import (
    build_radius_graph_model,
    estimate_connectivity_radius,
    radius_edges,
)


class RadiusMgnGnsTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.points = rng.normal(size=(64, 3)).astype(np.float32)

    def test_radius_graph_nonempty(self) -> None:
        radius = estimate_connectivity_radius(self.points, target_degree=8)
        edges = radius_edges(self.points, radius, max_degree=16)
        self.assertEqual(edges.ndim, 2)
        self.assertEqual(edges.shape[0], 2)
        self.assertGreater(edges.shape[1], 0)

    def test_mgn_residual_shapes(self) -> None:
        model = build_radius_graph_model(
            self.points, model="mgn_radius", state_dim=4, target_degree=8, max_degree=16
        )
        b, n = 2, self.points.shape[0]
        out = model(
            torch.randn(b, n, 3),
            torch.randn(b, n, 11),
            torch.randn(b, n, 4),
            torch.randn(b, n, 8),
        )
        self.assertEqual(tuple(out.shape), (b, n, 4))

    def test_gns_history_shapes(self) -> None:
        model = build_radius_graph_model(
            self.points,
            model="gns_radius",
            state_dim=4,
            history=5,
            target_degree=8,
            max_degree=16,
        )
        b, n = 2, self.points.shape[0]
        out = model(
            torch.randn(b, n, 3),
            torch.randn(b, n, 11),
            torch.randn(b, 5, n, 4),
            torch.randn(b, n, 8),
        )
        self.assertEqual(tuple(out.shape), (b, n, 4))
        self.assertEqual(model.history, 5)
        self.assertEqual(len(model.epd.edge_processors), 10)
        mgn = build_radius_graph_model(
            self.points, model="mgn_radius", state_dim=4, target_degree=8, max_degree=16
        )
        self.assertEqual(len(mgn.epd.edge_processors), 15)


if __name__ == "__main__":
    unittest.main()
