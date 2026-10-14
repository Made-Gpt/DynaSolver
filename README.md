# DynaSolver

**A neural operator for autoregressive 4D flow prediction with dynamic boundaries.**

<p align="center">
  <a href="https://made-gpt.github.io/DynaSolver-ProjectPage/"><img src="https://img.shields.io/badge/Project-Page-0A66C2?style=for-the-badge" alt="Project Page"></a>
  <a href="https://github.com/Made-Gpt/DynaSolver"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <a href="https://made-gpt.github.io/DynaSolver-ProjectPage/"><b>🌐 Project Page</b></a>
  ·
  <a href="https://made-gpt.github.io/DynaSolver-ProjectPage/static/paper/DynaSolver.pdf"><b>📄 Paper</b></a>
  ·
  <a href="docs/DATA.md"><b>📦 Data layout</b></a>
</p>

In configs and checkpoints the architecture is registered as `TemporalTransolver`;
`DynaSolver` is an equivalent alias.

## Highlights

- Long-horizon **autoregressive** prediction on unstructured CFD point sets
- Native support for **dynamic boundaries** (moving robots / kinematic scenes)
- Causal temporal blocks on top of a Transolver-style spatial backbone
- Ready-to-run train / rollout-eval / paper-plot scripts

## Setup

Requires **Python ≥ 3.10** and a recent PyTorch (`≥ 2.2`).

```bash
git clone https://github.com/Made-Gpt/DynaSolver.git
cd DynaSolver

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Required for flat imports used by training scripts
source scripts/env.sh
# equivalent: export PYTHONPATH="$PWD:$PWD/dynasolver${PYTHONPATH:+:$PYTHONPATH}"
```

Training / eval entrypoints under `scripts/` also prepend the package path themselves,
so `python scripts/train_temporal_cfd.py …` works after `source scripts/env.sh`.

### Optional extras

| Extra | When you need it |
|-------|------------------|
| `GEOPT_CHECKPOINT` | Optional GeoPT spatial backbone init (`--initialization geopt_pretrained`) |
| `flash-attn`, `torch-geometric` | MSPTUn / MSPTStr vendor paths or flash-attention speedups (not required for the flagship kinematic run) |

```bash
export DYNASOLVER_DATA=/path/to/your/data_root
export GEOPT_CHECKPOINT=/path/to/GeoPT_8layers.pt   # optional
```

See [docs/DATA.md](docs/DATA.md) for the expected directory layout and
[docs/kinematic_claim_dataset.md](docs/kinematic_claim_dataset.md) for the
WaterLily20 kinematic claim split used in the paper.

> **Data release status.** Trajectory HDF5 packs and pretrained weights are **not**
> bundled in this repository yet. Once the public dataset is available we will
> update `docs/DATA.md` with download links. Unit tests use synthetic HDF5 and
> do not require the full dataset.

## Quick smoke test (no external data)

```bash
source scripts/env.sh
pytest -q
```

## Train (flagship kinematic WaterLily20)

```bash
export DYNASOLVER_DATA=/path/to/your/data_root
# optional: export GEOPT_CHECKPOINT=/path/to/GeoPT_8layers.pt

python -u scripts/train_temporal_cfd.py \
  --config configs/casual_dual_grad_latent_dec_s16_rope_waterlily20_kinematic_grid18.yaml \
  --manifest "$DYNASOLVER_DATA/manifests/waterlily20_grid18x18x36_n11664_f30_kinematic_temporal_manifest.json" \
  --normalizer-path "$DYNASOLVER_DATA/manifests/normalizer.json" \
  --output-dir "$DYNASOLVER_DATA/runs/dynasolver_s16_kinematic/seed_17" \
  --initialization from_scratch
```

Baselines (MSPT / Transolver / …):

```bash
python -u scripts/train_baseline_cfd.py --config configs/baseline_mspt_grid18.yaml
```

## Evaluate autoregressive rollout

```bash
python -u scripts/eval_waterlily20_rollout_vp.py \
  --run-dir "$DYNASOLVER_DATA/runs/.../seed_17" \
  --checkpoint checkpoint_best_rollout.pt \
  --manifest "$DYNASOLVER_DATA/manifests/waterlily20_grid18x18x36_n11664_f30_kinematic_temporal_manifest.json" \
  --output-dir "$DYNASOLVER_DATA/runs/.../rollout_vp" \
  --label dynasolver
```

External third-party baseline runners (if used) must be provided separately via
`DYNASOLVER_BASELINE_ROOT`; see `scripts/eval_waterlily20_baseline_rollout_vp.py`.

## Paper curve plots

```bash
python -u scripts/plot_paper_rollout_all_relL2.py --root /path/to/methods_dir --out figures/paper/k6
```

## Repository layout

```
configs/          # flagship + baseline YAML configs
docs/             # data layout and claim-split notes
dynasolver/       # models, layers, data loaders, training helpers
scripts/          # train / eval / plot entrypoints
tests/            # unit tests (synthetic data)
figures/paper/    # example paper figures
```

## Citation

If you use this code, please cite:

```bibtex
@article{qian2026dynasolver,
  title   = {DynaSolver: A Neural Operator for Autoregressive 4D Flow Prediction with Dynamic Boundaries},
  author  = {Qian, Rui and Dai, Shibo and Lin, Mengjun and Hou, Yuk-Kit and Zhang, Saining and Chen, Junhao and Zhao, Hao and Li, Chaojian},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://made-gpt.github.io/DynaSolver-ProjectPage/}
}
```

## Acknowledgments

Spatial blocks build on the [Transolver](https://github.com/thuml/Transolver) / GeoPT lineage.
The MSPT baseline vendors [MSPT](https://github.com/pedrocurvo/mspt) (Apache-2.0);
see [NOTICE](NOTICE) for third-party attribution.

## License

MIT — Copyright (c) 2026 Rui Qian et al.
Vendored MSPT retains its upstream license under `dynasolver/vendor/mspt/LICENSE`.
