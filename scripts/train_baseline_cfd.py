#!/usr/bin/env python3
"""Train aligned one-step baselines: V0 / Transolver / MGN / GNS on Temporal contract.

Uses the same G1/H1 n3200 manifest, loss, and budget as Temporal Transolver.
Legacy kNN graph baselines (mgn_knn / sdf_gns) remain available; preferred
paper-faithful radius EPD baselines are mgn_radius / gns_radius.
"""
from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import random
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "dynasolver"
for _p in (REPO_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_provider.dynamic_cfd import (
    DynamicCFDOneStepDataset,
    DynamicStateNormalizer,
    load_trajectory_manifest,
    resolve_point_indices,
    validate_trajectory_hdf5,
)
from dynamic_cfd.training import rollout_trajectory, run_one_step_epoch
from models.ConditionOnlyTransolver import Model as ConditionOnlyTransolverModel
from models.ConditionOnlyTransolver import load_geopt_backbone as load_transolver_geopt_backbone
from models.DynamicTransolver import Model as DynamicTransolverModel
from models.DynamicTransolverV1Pre import Model as DynamicTransolverV1PreModel
from models.DynamicTransolverV1PreMask25 import Model as DynamicTransolverV1PreMask25Model
from models.DynamicTransolverV1_1 import Model as DynamicTransolverV1_1Model
from models.DynamicTransolver import load_geopt_backbone
from models.MSPTBaseline import Model as MSPTBaselineModel
from models.graph_baselines import build_graph_model
from models.radius_mgn_gns import build_radius_graph_model
from utils.constants import GEOPT_CHECKPOINT_SHA256
from utils.paths import GEOPT_CHECKPOINT


GRAPH_LEGACY = {"mgn_knn", "sdf_gns"}
GRAPH_RADIUS = {"mgn_radius", "gns_radius"}
GRAPH_MODELS = GRAPH_LEGACY | GRAPH_RADIUS


class PersistenceModel(torch.nn.Module):
    def forward(self, query_xyz, static_fx, previous_state, boundary_feat):
        del query_xyz, static_fx, boundary_feat
        if previous_state.ndim == 4:
            return previous_state[:, -1]
        return previous_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument(
        "--model",
        choices=(
            "v0",
            "v1_pre",
            "v1_pre_surface",
            "v1_pre_fs_rerun",
            "v1_1_fs",
            "transolver",
            "mspt",
            "mgn_knn",
            "sdf_gns",
            "mgn_radius",
            "gns_radius",
        ),
        help="override config model.name",
    )
    parser.add_argument(
        "--initialization",
        choices=("from_scratch", "geopt_pretrained"),
    )
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--pretrained-sha256", default=GEOPT_CHECKPOINT_SHA256)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--points", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-unhashed-data", action="store_true")
    parser.add_argument(
        "--normalizer-path",
        type=Path,
        help="frozen train-only normalizer JSON; if set, skip fit_from_manifest",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("validation_loss", "validation_rollout_relative_l2"),
        help="default from yaml (aligned Temporal uses validation_loss)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from output-dir/checkpoint_last.pt (model, optimizer, history, best)",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    return _expand(payload)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    fields = list(history[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_model_name(cfg_model: dict, override: str | None) -> str:
    if override is not None:
        return override
    raw = str(cfg_model.get("name", "v0")).lower()
    if raw in {"v0", "dynamictransolver", "dynamictransolverv0"}:
        return "v0"
    if raw in {"v1_pre", "dynamictransolverv1pre", "v1_pre_fs", "v1_pre_surface", "v1_pre_fs_rerun", "dynamictransolverv1premask25"}:
        if raw in {"v1_pre_fs_rerun", "dynamictransolverv1premask25"}:
            return "v1_pre_fs_rerun"
        return "v1_pre_surface" if raw == "v1_pre_surface" else "v1_pre"
    if raw in {"v1_1_fs", "dynamictransolverv1_1", "dynamictransolverv11"}:
        return "v1_1_fs"
    if raw in {"mgn_knn", "mgn"}:
        return "mgn_knn"
    if raw in {"sdf_gns", "sdf-gns"}:
        return "sdf_gns"
    if raw in {"mgn_radius", "mgn-radius"}:
        return "mgn_radius"
    if raw in {"gns_radius", "gns-radius", "gns"}:
        return "gns_radius"
    if raw in {"transolver", "conditiononlytransolver", "condition_only_transolver"}:
        return "transolver"
    if raw in {"mspt", "msptbaseline", "mspt_baseline"}:
        return "mspt"
    raise ValueError(
        f"unsupported model.name={raw!r}; "
        "use v0|v1_pre|v1_pre_surface|v1_1_fs|transolver|mspt|mgn_knn|sdf_gns|mgn_radius|gns_radius"
    )


def load_graph_points(
    manifest_path: Path,
    *,
    points: int,
    subset_seed: int,
) -> np.ndarray:
    """Load fixed Eulerian coordinates for graph construction."""
    manifest = load_trajectory_manifest(
        manifest_path,
        require_production_eligible=False,
        require_hashes=False,
        verify_hashes=False,
    )
    record = manifest.records[0]
    shape = validate_trajectory_hdf5(record.hdf5_path, validate_values=False)
    indices = resolve_point_indices(
        shape.points,
        points,
        seed=subset_seed,
        point_bank_id=record.point_bank_id,
    )
    with h5py.File(record.hdf5_path, "r") as handle:
        xyz = np.asarray(handle["points_world_m"], dtype=np.float32)[indices]
    if xyz.shape != (len(indices), 3):
        raise ValueError(f"unexpected points shape {xyz.shape}")
    return np.ascontiguousarray(xyz)


def _base_model_args(cfg_model: dict, state_dim: int) -> Namespace:
    return Namespace(
        fun_dim=int(cfg_model.get("fun_dim", 11)),
        space_dim=int(cfg_model.get("space_dim", 3)),
        state_dim=state_dim,
        boundary_dim=int(cfg_model.get("boundary_dim", 8)),
        n_hidden=int(cfg_model.get("n_hidden", 256)),
        n_heads=int(cfg_model.get("n_heads", 8)),
        n_layers=int(cfg_model.get("n_layers", 8)),
        mlp_ratio=int(cfg_model.get("mlp_ratio", 2)),
        out_dim=state_dim,
        slice_num=int(cfg_model.get("slice_num", 32)),
        dropout=float(cfg_model.get("dropout", 0.0)),
        act=str(cfg_model.get("act", "gelu")),
        geotype=str(cfg_model.get("geotype", "unstructured")),
        shapelist=None,
        checkpoint=False,
        unified_pos=False,
    )


def build_v0_model(cfg_model: dict, state_dim: int) -> DynamicTransolverModel:
    return DynamicTransolverModel(_base_model_args(cfg_model, state_dim))


def build_v1_pre_model(cfg_model: dict, state_dim: int) -> DynamicTransolverV1PreModel:
    args = _base_model_args(cfg_model, state_dim)
    args.input_feature_dim = int(cfg_model.get("input_feature_dim", 23))
    return DynamicTransolverV1PreModel(args)


def build_v1_pre_mask25_model(cfg_model: dict, state_dim: int) -> DynamicTransolverV1PreMask25Model:
    args = _base_model_args(cfg_model, state_dim)
    args.input_feature_dim = 25
    return DynamicTransolverV1PreMask25Model(args)


def build_v1_1_model(cfg_model: dict, state_dim: int) -> DynamicTransolverV1_1Model:
    args = _base_model_args(cfg_model, state_dim)
    args.hidden_dim = int(cfg_model.get("hidden_dim", cfg_model.get("n_hidden", 256)))
    args.interaction_groups = int(cfg_model.get("interaction_groups", 3))
    return DynamicTransolverV1_1Model(args)


def build_transolver_model(cfg_model: dict, state_dim: int) -> ConditionOnlyTransolverModel:
    return ConditionOnlyTransolverModel(_base_model_args(cfg_model, state_dim))


def build_mspt_model(cfg_model: dict, state_dim: int) -> MSPTBaselineModel:
    args = _base_model_args(cfg_model, state_dim)
    args.n_layers = int(cfg_model.get("n_layers", 5))
    args.supernodes = int(cfg_model.get("supernodes", 32))
    return MSPTBaselineModel(args)


def build_model(
    model_name: str,
    cfg_model: dict,
    *,
    state_dim: int,
    graph_points: np.ndarray | None,
) -> torch.nn.Module:
    if model_name == "v0":
        return build_v0_model(cfg_model, state_dim)
    if model_name in {"v1_pre", "v1_pre_surface"}:
        return build_v1_pre_model(cfg_model, state_dim)
    if model_name == "v1_pre_fs_rerun":
        return build_v1_pre_mask25_model(cfg_model, state_dim)
    if model_name == "v1_1_fs":
        return build_v1_1_model(cfg_model, state_dim)
    if model_name == "transolver":
        return build_transolver_model(cfg_model, state_dim)
    if model_name == "mspt":
        return build_mspt_model(cfg_model, state_dim)
    if graph_points is None:
        raise ValueError("graph baselines require graph_points")
    if model_name in GRAPH_RADIUS:
        kwargs: dict[str, Any] = {
            "state_dim": state_dim,
            "target_degree": int(cfg_model.get("target_degree", 16)),
            "max_degree": int(cfg_model.get("max_degree", 64)),
            "latent_size": int(cfg_model.get("latent_size", 128)),
            "mlp_hidden_size": int(cfg_model.get("mlp_hidden_size", 128)),
            "mlp_num_hidden_layers": int(cfg_model.get("mlp_num_hidden_layers", 2)),
        }
        if cfg_model.get("radius") is not None:
            kwargs["radius"] = float(cfg_model["radius"])
        if cfg_model.get("history") is not None:
            kwargs["history"] = int(cfg_model["history"])
        if cfg_model.get("processors") is not None:
            kwargs["num_message_passing_steps"] = int(cfg_model["processors"])
        elif cfg_model.get("num_message_passing_steps") is not None:
            kwargs["num_message_passing_steps"] = int(cfg_model["num_message_passing_steps"])
        return build_radius_graph_model(graph_points, model=model_name, **kwargs)
    return build_graph_model(
        graph_points,
        model=model_name,
        k=int(cfg_model.get("k", 16)),
        hidden=int(cfg_model.get("hidden", 96)),
        processors=int(cfg_model.get("processors", 3)),
        history=int(cfg_model.get("history", 1)),
    )


def rollout_split(model, dataset, device, max_steps: int | None) -> dict:
    trajectories = [
        rollout_trajectory(model, dataset, record.trajectory_id, device, max_steps=max_steps)
        for record in dataset.records
    ]
    return {
        "trajectories": trajectories,
        "relative_l2_mean": float(np.mean([row["relative_l2_mean"] for row in trajectories])),
        "relative_l2_final_mean": float(
            np.mean([row["relative_l2_final"] for row in trajectories])
        ),
        "channel_rmse_mean": np.mean(
            [row["channel_rmse"] for row in trajectories], axis=0
        ).tolist(),
        "channel_relative_l2_mean": np.mean(
            [row["channel_relative_l2"] for row in trajectories], axis=0
        ).tolist(),
    }


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, train_cfg: dict, device: str):
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": bool(train_cfg.get("pin_memory", False)) and device == "cuda",
    }
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = True
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, **kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config.resolve())
    paths = cfg.get("paths", {})
    train_cfg = dict(cfg.get("training", {}))
    model_cfg = dict(cfg.get("model", {}))

    model_name = resolve_model_name(model_cfg, args.model)
    initialization = args.initialization or str(train_cfg.get("initialization", "from_scratch"))
    if model_name in GRAPH_MODELS and initialization != "from_scratch":
        raise ValueError(f"{model_name} only supports from_scratch initialization")

    state_history = 1
    if model_name == "gns_radius":
        state_history = int(model_cfg.get("history", 5))
    elif model_name in GRAPH_LEGACY:
        state_history = int(model_cfg.get("history", 1))
    elif model_name == "mgn_radius":
        state_history = 1

    manifest = Path(args.manifest or paths["manifest"]).resolve()
    seed = int(args.seed if args.seed is not None else train_cfg.get("seeds", [17])[0])
    points = int(args.points if args.points is not None else train_cfg.get("points", 3200))
    subset_seed = int(train_cfg.get("subset_seed", 20260803))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 200))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 4))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1.0e-4))
    patience = int(train_cfg.get("early_stop_patience", 40))
    early_stop_min_epoch = int(train_cfg.get("early_stop_min_epoch", 0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-5))
    selection_metric_name = args.selection_metric or str(
        train_cfg.get("selection_metric", "validation_loss")
    )
    validation_rollout_steps = int(train_cfg.get("validation_rollout_steps", 100))
    if early_stop_min_epoch < 0:
        raise ValueError("early_stop_min_epoch must be >= 0")
    if early_stop_min_epoch > epochs:
        raise ValueError("early_stop_min_epoch must not exceed epochs")

    runs_root = Path(paths.get("runs_root", REPO_ROOT / "runs")).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{stamp}_{initialization}_n{points}"
    output_dir = Path(args.output_dir or (runs_root / run_id / f"seed_{seed}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seed_everything(seed)

    raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    l_ref = float(raw_manifest.get("dataset_metadata", {}).get("l_ref_m", 1.0))
    sample_speed = next(
        (
            float(entry["speed_m_per_s"])
            for entry in raw_manifest.get("trajectories", [])
            if entry.get("speed_m_per_s") is not None
        ),
        1.0,
    )
    traj_manifest = load_trajectory_manifest(
        manifest,
        require_production_eligible=True,
        require_hashes=not args.allow_unhashed_data,
        verify_hashes=False,
    )
    shapes = [validate_trajectory_hdf5(record.hdf5_path, validate_values=False) for record in traj_manifest.records]
    state_dims = {shape.state_dim for shape in shapes}
    if len(state_dims) != 1:
        raise ValueError(f"all trajectories must share one state_dim, got {state_dims}")
    state_dim = int(state_dims.pop())

    normalizer_path = None
    if args.normalizer_path is not None:
        normalizer_path = Path(args.normalizer_path).resolve()
    elif paths.get("normalizer"):
        normalizer_path = Path(paths["normalizer"]).resolve()

    if normalizer_path is not None:
        normalizer = DynamicStateNormalizer.from_dict(
            json.loads(normalizer_path.read_text(encoding="utf-8"))
        )
        if int(normalizer.state_dim) != state_dim:
            raise ValueError(
                f"frozen normalizer state_dim={normalizer.state_dim} != data state_dim={state_dim}"
            )
        normalizer_record = {
            "source": "frozen",
            "path": str(normalizer_path),
            "sha256": sha256_file(normalizer_path),
        }
    else:
        normalizer = DynamicStateNormalizer.fit_from_manifest(
            traj_manifest,
            u_ref_m_per_s=float(sample_speed),
            l_ref_m=l_ref,
            p_farfield_kinematic=0.0,
            subset_points=points,
            subset_seed=subset_seed,
        )
        normalizer_record = {
            "source": "fit_from_manifest",
            "u_ref_m_per_s": float(sample_speed),
            "l_ref_m": l_ref,
            "subset_points": points,
            "subset_seed": subset_seed,
        }
    write_json(output_dir / "normalizer.json", normalizer.to_dict())
    write_json(output_dir / "normalizer_provenance.json", normalizer_record)

    datasets = {
        split: DynamicCFDOneStepDataset(
            traj_manifest,
            split,
            normalizer,
            subset_points=points,
            subset_seed=subset_seed,
            history=state_history,
            include_surface=model_name in {"v1_pre_surface", "v1_1_fs"},
        )
        for split in ("train", "validation", "test")
    }
    train_loader = make_loader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        train_cfg=train_cfg,
        device=args.device,
    )
    validation_loader = make_loader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        train_cfg=train_cfg,
        device=args.device,
    )

    graph_points = None
    if model_name in GRAPH_MODELS:
        graph_points = load_graph_points(manifest, points=points, subset_seed=subset_seed)
        np.save(output_dir / "graph_points.npy", graph_points)

    model = build_model(model_name, model_cfg, state_dim=state_dim, graph_points=graph_points)
    checkpoint_load = None
    if initialization == "geopt_pretrained":
        pretrained = Path(
            args.pretrained_checkpoint
            or paths.get("geopt_checkpoint")
            or GEOPT_CHECKPOINT
        ).resolve()
        loader = (
            load_transolver_geopt_backbone
            if model_name == "transolver"
            else load_geopt_backbone
        )
        checkpoint_load = loader(model, pretrained, expected_sha256=args.pretrained_sha256)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "model_name": model_name,
        "experiment_id": cfg.get("experiment_id"),
        "config": str(args.config.resolve()),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "initialization": initialization,
        "seed": seed,
        "points": points,
        "subset_seed": subset_seed,
        "batch_size": batch_size,
        "lr": lr,
        "epochs_requested": epochs,
        "selection_metric": selection_metric_name,
        "validation_rollout_steps": validation_rollout_steps,
        "device": str(device),
        "state_dim": state_dim,
        "model": model_cfg,
        "feature_contract": {
            "volume_input_dim": 25 if model_name == "v1_pre_fs_rerun" else (23 if model_name in {"v1_pre", "v1_pre_surface", "v1_1_fs"} else None),
            "volume_feature_layout": [
                "query_xyz_normalized", "previous_state", "inflow_direction",
                "normalized_speed", "normalized_dt", "sdf_current_and_delta",
                "displacement_using_next_frame_normal", "next_frame_normal",
                "nearest_boundary_velocity", "current_fluid_mask", "next_fluid_mask",
            ] if model_name == "v1_pre_fs_rerun" else [],
            "surface_input_dim": 15 if model_name == "v1_1_fs" else None,
            "surface_v_definition": "surface_state_raw[...,0:3] current surface-adjacent fluid velocity" if model_name == "v1_1_fs" else None,
            "surface_features": ["surface_xyz_normalized", "surface_normal_current", "surface_velocity_encoded", "surface_pressure_encoded", "inflow_direction", "normalized_speed", "normalized_dt"] if model_name == "v1_1_fs" else [],
            "surface_points": int(cfg.get("comparison", {}).get("surface_points", 0)) if model_name in {"v1_pre_surface", "v1_1_fs"} else 0,
            "surface_tokens_fixed": bool(model_cfg.get("surface_tokens_fixed", True)) if model_name == "v1_1_fs" else False,
            "interaction_pattern": str(model_cfg.get("interaction_pattern", "")) if model_name == "v1_1_fs" else "",
        },
        "checkpoint_load": checkpoint_load,
        "normalizer": normalizer_record,
        "split_counts": {split: len(datasets[split]) for split in datasets},
        "started_at_utc": utc_now(),
        "status": "running",
        "output_dir": str(output_dir),
        "log_file": str(args.log_file.resolve()) if args.log_file else None,
        "command": sys.argv,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "status": "run_start",
                "model": model_name,
                "initialization": initialization,
                "output_dir": str(output_dir),
                "manifest": str(manifest),
                "points": points,
                "seed": seed,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    history: list[dict] = []
    best_metric = float("inf")
    best_epoch = 0
    cumulative_seconds = 0.0
    start_epoch = 1

    if args.resume:
        resume_path = output_dir / "checkpoint_last.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume requested but missing {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        history = list(resume_payload.get("history") or [])
        if not history:
            raise ValueError(f"resume checkpoint has empty history: {resume_path}")
        best_epoch = int(resume_payload.get("epoch", history[-1]["epoch"]))
        selection_key = str(resume_payload.get("selection_metric", selection_metric_name))
        if selection_key != selection_metric_name:
            raise ValueError(
                f"resume selection_metric={selection_key!r} != {selection_metric_name!r}"
            )
        best_rows = [
            row for row in history if int(row["epoch"]) <= best_epoch
        ]
        best_metric = float(
            min(float(row[selection_metric_name]) for row in best_rows)
        )
        best_epoch = min(
            int(row["epoch"])
            for row in best_rows
            if float(row[selection_metric_name]) == best_metric
        )
        cumulative_seconds = float(history[-1].get("cumulative_seconds", 0.0))
        start_epoch = int(history[-1]["epoch"]) + 1
        run_manifest["status"] = "resumed"
        run_manifest["resumed_from_epoch"] = int(history[-1]["epoch"])
        run_manifest["resume_checkpoint"] = str(resume_path)
        write_json(output_dir / "run_manifest.json", run_manifest)
        print(
            json.dumps(
                {
                    "status": "resume_start",
                    "start_epoch": start_epoch,
                    "best_epoch": best_epoch,
                    "best_selection_value": best_metric,
                    "history_epochs": len(history),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for epoch in range(start_epoch, epochs + 1):
        synchronize(device)
        started = time.perf_counter()
        train_metrics = run_one_step_epoch(
            model, train_loader, device, optimizer, max_grad_norm=max_grad_norm
        )
        validation_metrics = run_one_step_epoch(model, validation_loader, device)
        validation_rollout = rollout_split(
            model,
            datasets["validation"],
            device,
            max_steps=validation_rollout_steps,
        )
        synchronize(device)
        epoch_seconds = time.perf_counter() - started
        cumulative_seconds += epoch_seconds
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_persistence_loss": train_metrics["persistence_loss"],
            "validation_loss": validation_metrics["loss"],
            "validation_persistence_loss": validation_metrics["persistence_loss"],
            "validation_rollout_relative_l2": validation_rollout["relative_l2_mean"],
            "epoch_seconds": epoch_seconds,
            "cumulative_seconds": cumulative_seconds,
        }
        if not all(np.isfinite(float(v)) for v in row.values()):
            raise RuntimeError(f"non-finite metric at epoch {epoch}: {row}")
        history.append(row)
        write_json(output_dir / "history.json", history)
        write_history_csv(output_dir / "epoch_history.csv", history)
        print(json.dumps(row, sort_keys=True), flush=True)

        selection_metric = float(row[selection_metric_name])
        is_best = selection_metric < best_metric
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.to_dict(),
            "run_manifest": run_manifest,
            "epoch": epoch,
            "validation_loss": row["validation_loss"],
            "validation_rollout_relative_l2": row["validation_rollout_relative_l2"],
            "selection_metric": selection_metric_name,
            "selection_value": selection_metric,
            "history": history,
            "model_name": model_name,
        }
        torch.save(payload, output_dir / "checkpoint_last.pt")
        if is_best:
            best_metric = selection_metric
            best_epoch = epoch
            torch.save(payload, output_dir / "checkpoint_best.pt")
        if (
            epoch >= early_stop_min_epoch
            and epoch - best_epoch >= patience
        ):
            print(
                json.dumps(
                    {
                        "status": "early_stopped",
                        "best_epoch": best_epoch,
                        "patience": patience,
                        "early_stop_min_epoch": early_stop_min_epoch,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break

    checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loader = make_loader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        train_cfg=train_cfg,
        device=args.device,
    )
    test_one_step = run_one_step_epoch(model, test_loader, device)
    test_rollout = rollout_split(model, datasets["test"], device, max_steps=None)
    persistence_rollout = rollout_split(
        PersistenceModel().to(device), datasets["test"], device, max_steps=None
    )
    summary = {
        "status": "completed",
        "model_name": model_name,
        "initialization": initialization,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_selection_metric": selection_metric_name,
        "best_selection_value": best_metric,
        "best_validation_loss": min(row["validation_loss"] for row in history),
        "test_one_step": test_one_step,
        "test_full_cycle_rollout": {
            k: v for k, v in test_rollout.items() if k != "trajectories"
        },
        "persistence_full_cycle_rollout": {
            k: v for k, v in persistence_rollout.items() if k != "trajectories"
        },
        "finished_at_utc": utc_now(),
    }
    # keep per-traj compact metrics
    summary["test_full_cycle_rollout"]["trajectories"] = [
        {
            "trajectory_id": t["trajectory_id"],
            "relative_l2_mean": t["relative_l2_mean"],
            "relative_l2_final": t["relative_l2_final"],
            "mse_mean": t["mse_mean"],
            "steps": t["steps"],
        }
        for t in test_rollout["trajectories"]
    ]
    summary["persistence_full_cycle_rollout"]["trajectories"] = [
        {
            "trajectory_id": t["trajectory_id"],
            "relative_l2_mean": t["relative_l2_mean"],
            "relative_l2_final": t["relative_l2_final"],
            "mse_mean": t["mse_mean"],
            "steps": t["steps"],
        }
        for t in persistence_rollout["trajectories"]
    ]
    write_json(output_dir / "summary.json", summary)
    run_manifest["status"] = "completed"
    run_manifest["finished_at_utc"] = utc_now()
    write_json(output_dir / "run_manifest.json", run_manifest)

    for dataset in datasets.values():
        dataset.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
