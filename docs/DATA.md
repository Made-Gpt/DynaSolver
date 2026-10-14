# Data layout

DynaSolver does not ship HDF5 trajectories or checkpoints in this repository.
A public dataset release is planned; until then, point the code at a local copy.

Set:

| Variable | Purpose |
|----------|---------|
| `DYNASOLVER_DATA` | Root for manifests, runs, optional checkpoints |
| `GEOPT_CHECKPOINT` | Optional GeoPT backbone weights (`.pt`) |
| `DYNASOLVER_BASELINE_ROOT` | Optional external baseline workspace for `eval_waterlily20_baseline_rollout_vp.py` |
| `PYTHON_BIN` | Interpreter override (defaults to `python`) |

Suggested tree:

```
$DYNASOLVER_DATA/
  manifests/
    waterlily20_grid18x18x36_n11664_f30_kinematic_temporal_manifest.json
    normalizer.json
  runs/
  checkpoints/
    GeoPT_8layers.pt          # or set GEOPT_CHECKPOINT elsewhere
  # HDF5 trajectories referenced by the manifest
```

Manifest JSON entries should point at trajectory HDF5 files on your machine.
Example config paths use placeholders under `$DYNASOLVER_DATA`; override any
path with CLI flags (`--manifest`, `--output-dir`, …).

Normalizer JSON (kinematic claim) is typically stored next to the packed
dataset; pass it via config `paths.normalizer` or the training CLI.

## Reproducing paper numbers

1. Install deps (`pip install -e ".[dev]"` + `source scripts/env.sh`).
2. Obtain the WaterLily20 kinematic pack and place it under `$DYNASOLVER_DATA`
   as above (see [kinematic_claim_dataset.md](kinematic_claim_dataset.md)).
3. Train with the flagship config in the README (`from_scratch` needs no GeoPT).
4. Evaluate with `scripts/eval_waterlily20_rollout_vp.py`.

Without the full dataset you can still validate the install via `pytest -q`
(synthetic HDF5 fixtures).
