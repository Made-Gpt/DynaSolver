from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_temporal_cfd.py"
    spec = importlib.util.spec_from_file_location("train_temporal_cfd", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight


def test_build_and_load_training_checkpoint(tmp_path: Path) -> None:
    train = _load_train_module()
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    normalizer_dict = {
        "state_dim": 4,
        "u_ref_m_per_s": 1.0,
        "l_ref_m": 1.0,
        "p_farfield_kinematic": 0.0,
        "positive_floor": 1.0e-12,
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0],
    }
    normalizer = SimpleNamespace(to_dict=lambda: dict(normalizer_dict))
    history = [{"epoch": 1, "train_loss": 0.2, "validation_loss": 0.1, "epoch_seconds": 1.0}]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    run_manifest = {
        "manifest": str(manifest_path.resolve()),
        "seed": 17,
        "points": 3200,
    }
    payload = train.build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        normalizer=normalizer,
        run_manifest=run_manifest,
        epoch=1,
        best_validation_loss=0.1,
        best_epoch=1,
        history=history,
    )
    payload["best_selection_metric"] = "validation_rollout_relative_l2"
    payload["best_selection_value"] = 0.32
    payload["selection_metric"] = "validation_rollout_relative_l2"
    # Simulate the resume bug scenario: val-loss is tiny vs rollout selection.
    history_roll = [
        {
            "epoch": 1,
            "train_loss": 0.2,
            "validation_loss": 0.007,
            "validation_rollout_relative_l2": 0.32,
            "selection_metric": "validation_rollout_relative_l2",
            "selection_value": 0.32,
            "epoch_seconds": 1.0,
        },
        {
            "epoch": 2,
            "train_loss": 0.18,
            "validation_loss": 0.0065,
            "validation_rollout_relative_l2": 0.30,
            "selection_metric": "validation_rollout_relative_l2",
            "selection_value": 0.30,
            "epoch_seconds": 1.0,
        },
    ]
    payload["history"] = history_roll
    payload["best_validation_loss"] = 0.007
    payload["best_epoch"] = 1
    ckpt_path = tmp_path / "checkpoint_last.pt"
    train.save_training_checkpoint(ckpt_path, payload)

    model2 = DummyModel()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1.0e-3)
    (tmp_path / "history.json").write_text(json.dumps(history_roll), encoding="utf-8")

    loaded_normalizer, loaded_history, best, best_epoch, start_epoch = train.load_training_resume(
        ckpt_path,
        model=model2,
        optimizer=optimizer2,
        device=torch.device("cpu"),
        manifest=manifest_path,
        seed=17,
        points=3200,
    )
    assert start_epoch == 3
    assert best_epoch == 2
    assert abs(best - 0.30) < 1.0e-9
    assert loaded_history == history_roll
    assert loaded_normalizer.to_dict()["mean"] == [0.0, 0.0, 0.0, 0.0]
    assert loaded_normalizer.to_dict()["std"] == [1.0, 1.0, 1.0, 1.0]


def test_validate_resume_compatibility_rejects_manifest_mismatch(tmp_path: Path) -> None:
    train = _load_train_module()
    payload = {"run_manifest": {"manifest": "/tmp/a.json", "seed": 17, "points": 3200}}
    try:
        train.validate_resume_compatibility(
            payload,
            manifest=Path("/tmp/b.json"),
            seed=17,
            points=3200,
        )
    except ValueError as exc:
        assert "manifest" in str(exc)
    else:
        raise AssertionError("expected ValueError")
