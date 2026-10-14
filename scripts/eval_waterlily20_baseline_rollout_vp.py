#!/usr/bin/env python3
"""AR rollout V/P relative-L2 for external waterlily20 baselines (best_roll ckpt).

Writes the same schema as ``eval_waterlily20_rollout_vp.py`` so collect/plot
can merge ours + baselines.

Requires an external baseline workspace via ``DYNASOLVER_BASELINE_ROOT``
(or ``--baseline-root``) that provides ``scripts/evaluate_waterlily20_baselines.py``.

Example:
  export DYNASOLVER_BASELINE_ROOT=/path/to/baseline_workspace
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:./dynasolver \
    python -u scripts/eval_waterlily20_baseline_rollout_vp.py \
      --arm mspt --run-dir .../final_baselines/mspt \
      --manifest .../manifest.json \
      --normalizer .../normalizer.json \
      --checkpoint checkpoint_best_rollout.pt \
      --output-dir .../baselines/mspt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "dynasolver"
for _p in (REPO_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _resolve_baseline_root(cli: str | None) -> Path:
    raw = cli or os.environ.get("DYNASOLVER_BASELINE_ROOT")
    if not raw:
        raise SystemExit(
            "Set DYNASOLVER_BASELINE_ROOT or pass --baseline-root to the external "
            "baseline workspace (must contain scripts/evaluate_waterlily20_baselines.py)."
        )
    root = Path(os.path.expandvars(raw)).expanduser().resolve()
    if not (root / "scripts" / "evaluate_waterlily20_baselines.py").is_file():
        raise SystemExit(f"missing evaluate_waterlily20_baselines.py under {root}")
    return root


import importlib.util


def _load_eval_mod(baseline_root: Path):
    path = baseline_root / "scripts" / "evaluate_waterlily20_baselines.py"
    spec = importlib.util.spec_from_file_location("eval_wl20", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _import_multibaseline(baseline_root: Path):
    """Import external multibaseline package after putting its src on sys.path."""
    for p in (
        baseline_root / "src",
        baseline_root / "scripts",
        baseline_root / "vendor" / "neuraloperator",
    ):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from multibaseline.losses import weighted_relative_l2
    from multibaseline.normalization import StateNormalizer
    from multibaseline.persistent_graph_dataset import PersistentGraphDataset
    from multibaseline.rollout import (
        _seed_history,
        _stack_history,
        resolve_history_frames,
    )

    return {
        "weighted_relative_l2": weighted_relative_l2,
        "StateNormalizer": StateNormalizer,
        "PersistentGraphDataset": PersistentGraphDataset,
        "_seed_history": _seed_history,
        "_stack_history": _stack_history,
        "resolve_history_frames": resolve_history_frames,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--normalizer", type=Path, required=True)
    p.add_argument("--checkpoint", default="checkpoint_best_rollout.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--baseline-root",
        type=str,
        default=None,
        help="External baseline workspace (or set DYNASOLVER_BASELINE_ROOT).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-test-trajs", type=int, default=0, help="0 = all")
    p.add_argument("--horizon", type=int, default=0, help="0 = full traj")
    p.add_argument("--decimals", type=int, default=6)
    p.add_argument("--neighbors", type=int, default=16)
    p.add_argument(
        "--robots",
        type=str,
        default="",
        help="comma-separated robot prefixes to keep (e.g. h1_2,h1,solo,a1). "
        "Empty = all test trajs. Match is on trajectory_id (h1_2 before h1).",
    )
    p.add_argument(
        "--traj-ids",
        type=str,
        default="",
        help="comma-separated trajectory_ids to keep (exact match). Empty = all.",
    )
    p.add_argument(
        "--save-fields",
        action="store_true",
        help="dump per-traj NPZ under output-dir/fields/ (pred/gt [T,N,4], xyz, masks)",
    )
    return p.parse_args()


def _robot_of(tid: str) -> str:
    tid = str(tid)
    if tid.startswith("h1_2"):
        return "h1_2"
    if tid.startswith("h1_"):
        return "h1"
    if tid.startswith("solo"):
        return "solo"
    if tid.startswith("a1_"):
        return "a1"
    return tid.split("_", 1)[0]


def _rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    channel_slice: slice | None = None,
) -> float:
    if channel_slice is not None:
        pred = pred[..., channel_slice]
        target = target[..., channel_slice]
    return float(
        weighted_relative_l2(
            pred.unsqueeze(0),
            target.unsqueeze(0),
            mask.unsqueeze(0),
            weight.unsqueeze(0),
        ).mean().item()
    )


@torch.no_grad()
def rollout_vp_baseline(
    model,
    dataset,
    run_sample,
    device: torch.device,
    *,
    max_trajs: int = 0,
    horizon: int = 0,
    robots: set[str] | None = None,
    traj_ids_keep: set[str] | None = None,
    save_fields: bool = False,
    fields_dir: Path | None = None,
) -> dict[str, Any]:
    history_frames = resolve_history_frames(model, dataset)
    groups: dict[str, list[tuple[int, int]]] = {}
    for index, (record, time_index) in enumerate(dataset.samples):
        groups.setdefault(record.trajectory_id, []).append((int(time_index), index))

    traj_ids = sorted(groups)
    if robots:
        traj_ids = [tid for tid in traj_ids if _robot_of(tid) in robots]
        print(f"robot filter={sorted(robots)} → {len(traj_ids)} trajs", flush=True)
    if traj_ids_keep:
        traj_ids = [tid for tid in traj_ids if tid in traj_ids_keep]
        print(f"traj-ids filter={sorted(traj_ids_keep)} → {len(traj_ids)} trajs", flush=True)
    if max_trajs > 0:
        traj_ids = traj_ids[:max_trajs]

    per_traj: list[dict[str, Any]] = []
    vel_curves: list[list[float]] = []
    p_curves: list[list[float]] = []
    all_curves: list[list[float]] = []

    if model is not None:
        model.eval()
    for ti, tid in enumerate(traj_ids):
        items = sorted(groups[tid])
        if horizon > 0:
            items = items[:horizon]
        print(f"  AR {ti+1}/{len(traj_ids)} {tid} steps={len(items)}", flush=True)
        window: list[torch.Tensor] = []
        all_list: list[float] = []
        vel_list: list[float] = []
        p_list: list[float] = []
        # Build [T=n_steps+1, N, 4]: frame 0 = GT seed; frames 1.. = AR preds.
        pred_frames: list[np.ndarray] = []
        gt_frames: list[np.ndarray] = []
        query_xyz = None
        valid_frames: list[np.ndarray] = []
        for step, (_time_index, sample_index) in enumerate(items):
            sample = dataset[sample_index]
            if step == 0:
                window = _seed_history(sample["previous_state"])
                if save_fields:
                    prev0 = sample["previous_state"]
                    if prev0.ndim == 3:
                        prev0 = prev0[-1]
                    seed = prev0.detach().cpu().numpy().astype(np.float32)
                    pred_frames.append(seed.copy())
                    gt_frames.append(seed.copy())
                    query_xyz = sample["xyz"].detach().cpu().numpy().astype(np.float32)
                    valid_frames.append(np.ones((seed.shape[0],), dtype=np.bool_))
            else:
                sample = dict(sample)
                stacked = _stack_history(window, history_frames)
                sample["previous_state"] = stacked if stacked is not None else window[-1]
            prediction = run_sample(model, sample)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            if prediction.ndim == 3:
                prediction = prediction[0]
            target = sample["target_state"].to(device)
            mask = sample["target_mask"].to(device)
            weight = sample["weight"].to(device)
            pred = prediction if prediction.device == device else prediction.to(device)
            all_list.append(_rel_l2(pred, target, mask, weight))
            vel_list.append(_rel_l2(pred, target, mask, weight, slice(0, 3)))
            p_list.append(_rel_l2(pred, target, mask, weight, slice(3, 4)))
            window.append(pred.detach().cpu())
            if len(window) > history_frames:
                del window[0]
            if save_fields:
                pred_frames.append(pred.detach().cpu().numpy().astype(np.float32))
                gt_frames.append(target.detach().cpu().numpy().astype(np.float32))
                valid_frames.append(mask.detach().cpu().numpy().astype(np.bool_))
        vel_curves.append(vel_list)
        p_curves.append(p_list)
        all_curves.append(all_list)
        traj_entry: dict[str, Any] = {
            "trajectory_id": tid,
            "robot": _robot_of(tid),
            "steps": len(all_list),
            "relative_l2_all_mean": float(np.mean(all_list)),
            "relative_l2_velocity_mean": float(np.mean(vel_list)),
            "relative_l2_pressure_mean": float(np.mean(p_list)),
            "per_step_relative_l2_velocity": vel_list,
            "per_step_relative_l2_pressure": p_list,
            "per_step_relative_l2_all": all_list,
        }
        if save_fields and fields_dir is not None and pred_frames:
            safe_id = str(tid).replace("/", "_")
            npz_path = fields_dir / f"{safe_id}.npz"
            pred_state = np.stack(pred_frames, axis=0)
            gt_state = np.stack(gt_frames, axis=0)
            valid = np.stack(valid_frames, axis=0)
            # fluid ≈ valid for baseline dumps (no separate fluid channel here)
            np.savez_compressed(
                npz_path,
                pred_state=pred_state,
                gt_state=gt_state,
                query_xyz=query_xyz,
                valid=valid,
                fluid=valid,
                base_weight=np.ones((pred_state.shape[1],), dtype=np.float32),
                keyframe_timeline=np.arange(pred_state.shape[0], dtype=np.int32),
                per_step_relative_l2_all=np.asarray(all_list, dtype=np.float64),
                per_step_relative_l2_velocity=np.asarray(vel_list, dtype=np.float64),
                per_step_relative_l2_pressure=np.asarray(p_list, dtype=np.float64),
                relative_l2_all_mean=np.float64(traj_entry["relative_l2_all_mean"]),
                relative_l2_velocity_mean=np.float64(
                    traj_entry["relative_l2_velocity_mean"]
                ),
                relative_l2_pressure_mean=np.float64(
                    traj_entry["relative_l2_pressure_mean"]
                ),
            )
            traj_entry["fields_npz"] = str(npz_path)
        per_traj.append(traj_entry)
    return {
        "per_traj": per_traj,
        "vel_curves": vel_curves,
        "p_curves": p_curves,
        "all_curves": all_curves,
    }


def _mean_pad(curves: list[list[float]]) -> list[float]:
    if not curves:
        return []
    n = max(len(c) for c in curves)
    out = []
    for i in range(n):
        vals = [c[i] for c in curves if i < len(c)]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = _resolve_baseline_root(args.baseline_root)
    mb = _import_multibaseline(baseline_root)
    globals().update(mb)
    StateNormalizer = mb["StateNormalizer"]
    PersistentGraphDataset = mb["PersistentGraphDataset"]

    eval_mod = _load_eval_mod(baseline_root)
    import re

    # strip version suffix for arm key (gino_v2 -> gino), match eval_finished_arms.sh
    arm_key = re.sub(r"_(v[0-9]+|legacy[0-9]*)$", "", args.arm)

    normalizer = StateNormalizer.from_json(args.normalizer)
    history_frames = 1
    run_manifest_path = args.run_dir / "run_manifest.json"
    if run_manifest_path.is_file():
        history_frames = max(
            int(json.loads(run_manifest_path.read_text()).get("history_frames", 1) or 1),
            1,
        )
    traj_ids_keep: set[str] | None = None
    if str(args.traj_ids).strip():
        traj_ids_keep = {x.strip() for x in str(args.traj_ids).split(",") if x.strip()}

    # When targeting specific traj-ids, include train+test (failure wind sweeps
    # may need train-split cases like a1_u2_yaw_p0 / a1_u8_yaw_p0).
    splits = ("test",)
    if traj_ids_keep:
        splits = ("train", "test")
    datasets = [
        PersistentGraphDataset(
            args.manifest,
            sp,
            normalizer,
            drop_boundary=True,
            state_history=history_frames,
        )
        for sp in splits
    ]
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        class _ConcatDS:
            def __init__(self, parts):
                self.parts = parts
                self.samples = []
                self._index = []  # (part_i, local_i)
                for pi, p in enumerate(parts):
                    for li in range(len(p.samples)):
                        self.samples.append(p.samples[li])
                        self._index.append((pi, li))
                # expose attrs used by resolve_history_frames
                self.state_history = parts[0].state_history

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, i):
                pi, li = self._index[i]
                return self.parts[pi][li]

        ds = _ConcatDS(datasets)
        print(f"traj-ids mode: loaded splits={list(splits)} n_samples={len(ds)}", flush=True)

    model, run_sample, model_meta = eval_mod.build(
        arm_key,
        args.run_dir,
        args.manifest,
        device,
        args.neighbors,
        args.checkpoint,
        bsms_hierarchy=baseline_root / "metadata/graphs/waterlily18_grid18_n11664_bsms_k12_d2.npz",
    )
    print(
        f"arm={args.arm} key={arm_key} ckpt={args.checkpoint} "
        f"ep={model_meta.get('checkpoint_epoch')} hist={history_frames}",
        flush=True,
    )

    robots: set[str] | None = None
    if str(args.robots).strip():
        robots = {x.strip() for x in str(args.robots).split(",") if x.strip()}
    fields_dir: Path | None = None
    if args.save_fields:
        fields_dir = out_dir / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)
        print(f"save_fields -> {fields_dir}", flush=True)

    rolled = rollout_vp_baseline(
        model,
        ds,
        run_sample,
        device,
        max_trajs=args.max_test_trajs,
        horizon=args.horizon,
        robots=robots,
        traj_ids_keep=traj_ids_keep,
        save_fields=bool(args.save_fields),
        fields_dir=fields_dir,
    )
    per_traj = rolled["per_traj"]
    vel_mean_per_frame = _mean_pad(rolled["vel_curves"])
    p_mean_per_frame = _mean_pad(rolled["p_curves"])
    all_mean_per_frame = _mean_pad(rolled["all_curves"])

    summary = {
        "label": args.arm,
        "run_dir": str(args.run_dir),
        "manifest": str(args.manifest),
        "checkpoint": args.checkpoint,
        "checkpoint_meta": model_meta,
        "n_trajs": len(per_traj),
        "n_steps": len(vel_mean_per_frame),
        "robots": sorted(robots) if robots else None,
        "save_fields": bool(args.save_fields),
        "fields_dir": str(fields_dir) if fields_dir is not None else None,
        "average": {
            "relative_l2_all": float(np.mean([t["relative_l2_all_mean"] for t in per_traj])),
            "relative_l2_velocity": float(
                np.mean([t["relative_l2_velocity_mean"] for t in per_traj])
            ),
            "relative_l2_pressure": float(
                np.mean([t["relative_l2_pressure_mean"] for t in per_traj])
            ),
        },
        "per_frame": {
            "relative_l2_all": all_mean_per_frame,
            "relative_l2_velocity": vel_mean_per_frame,
            "relative_l2_pressure": p_mean_per_frame,
        },
        "trajectories": per_traj,
    }
    dest = out_dir / "rollout_vp.json"
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if fields_dir is not None:
        index_path = fields_dir / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "label": summary["label"],
                    "n_trajs": summary["n_trajs"],
                    "robots": summary["robots"],
                    "average": summary["average"],
                    "trajectories": [
                        {
                            "trajectory_id": t["trajectory_id"],
                            "robot": t.get("robot"),
                            "fields_npz": t.get("fields_npz"),
                            "relative_l2_all_mean": t["relative_l2_all_mean"],
                            "relative_l2_velocity_mean": t["relative_l2_velocity_mean"],
                            "relative_l2_pressure_mean": t["relative_l2_pressure_mean"],
                        }
                        for t in per_traj
                    ],
                    "note": (
                        "pred_state/gt_state normalizer space [T,N,4]; "
                        "T=n_steps+1 with frame0=GT seed, frames1..=AR preds."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {index_path}", flush=True)

    nd = int(args.decimals)
    avg = summary["average"]
    print("\n=== average ===", flush=True)
    print(f"relative_l2_all      = {avg['relative_l2_all']:.{nd}f}", flush=True)
    print(f"relative_l2_velocity = {avg['relative_l2_velocity']:.{nd}f}", flush=True)
    print(f"relative_l2_pressure = {avg['relative_l2_pressure']:.{nd}f}", flush=True)
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
