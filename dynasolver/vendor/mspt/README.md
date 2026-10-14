# MSPT

<p align="center">
  <strong>MSPT: Efficient Large-Scale Physical Modeling via Parallelized Multi-Scale Attention</strong>
</p>

<p align="center">
  <img src="docs/static/images/figure1.png" alt="MSPT Parallelized Multi-Scale Attention overview" width="560">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2026-4b44ce.svg" alt="CVPR 2026">
  <a href="https://arxiv.org/abs/2512.01738"><img src="https://img.shields.io/badge/arXiv-2512.01738-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.alphaxiv.org/abs/2512.01738"><img src="https://img.shields.io/badge/AlphaXiv-2512.01738-1f6feb.svg" alt="AlphaXiv"></a>
  <a href="https://huggingface.co/collections/pedrocurvo/mspt"><img src="https://img.shields.io/badge/HuggingFace-Checkpoints-f9c74f" alt="HF"></a>
  <img src="https://img.shields.io/github/license/pedrocurvo/mspt" alt="license">
  <img src="https://img.shields.io/badge/git-submodule-enabled-blue" alt="submodule-enabled">
</p>

MSPT is a scalable neural PDE solver based on **Parallelized Multi-Scale Attention (PMSA)**. It combines local patch attention with global supernode communication, enabling strong accuracy and efficient scaling on PDE and CFD benchmarks.

## Abstract
A key scalability challenge in neural solvers for industrial-scale physics simulations is efficiently capturing both fine-grained local interactions and long-range global dependencies across millions of spatial elements. We introduce the Multi-Scale Patch Transformer (MSPT), an architecture that combines local point attention within patches with global attention to coarse patch-level representations. To partition the input domain into spatially coherent patches, we employ ball trees, which handle irregular geometries efficiently. This dual-scale design enables MSPT to scale to millions of points on a single GPU. We validate MSPT on standard PDE benchmarks (elasticity, plasticity, fluid dynamics, porous flow) and large-scale aerodynamic datasets (ShapeNet-Car, Ahmed-ML), achieving state-of-the-art accuracy with substantially lower memory footprint and computational cost.

## Links
- Venue: CVPR 2026
- Paper: `https://arxiv.org/abs/2512.01738`
- AlphaXiv: `https://www.alphaxiv.org/abs/2512.01738`
- Project website: `./docs` (GitHub Pages content)
- Checkpoints collection: `https://huggingface.co/collections/pedrocurvo/mspt`

## Repository Layout
- `benchmarks/00-PDE/`: PDE benchmark harness for elasticity, plasticity, Navier-Stokes, Darcy, airfoil, and pipe experiments.
- `benchmarks/00-PDE/mspt/`: standalone MSPT PDE scripts and SLURM launchers.
- `benchmarks/Neural-Solver-Library-MSPT/`: general Neural Solver benchmark framework, included as a git submodule.
- `docs/`: project website source (GitHub Pages)

## Models
- `MSPT.py`: combined implementation containing both structured and unstructured MSPT variants.
- `MSPTStr.py`: structured MSPT variant (separate file).
- `MSPTUn.py`: unstructured MSPT variant (separate file).
- `benchmarks/MSPT.py`: benchmark-local structured MSPT implementation used by `benchmarks/00-PDE/mspt`.
- `benchmarks/MSPTUn.py`: benchmark-local unstructured MSPT implementation used by the elasticity PDE benchmark.

> [!WARNING]
> `MSPT.py` is the reference file used in **Neural-Solver-Library-MSPT**, and it contains both structured and unstructured models in one place.  
> `MSPTStr.py` and `MSPTUn.py` are the same model family split into separate files, with minor implementation variations.

## Setup
This repository uses a git submodule.

```bash
git clone --recurse-submodules https://github.com/pedrocurvo/mspt.git
cd mspt
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

Install dependencies:

```bash
cd benchmarks/Neural-Solver-Library-MSPT
pip install -r requirements.txt
```

## PDE Benchmarks
For PDE experiments, use `benchmarks/00-PDE`. The dataset-specific MSPT scripts live in:

```bash
cd benchmarks/00-PDE/mspt
```

SLURM launchers are in:

```bash
benchmarks/00-PDE/mspt/scripts/
```

Example (MSPT on Plasticity):

```bash
cd benchmarks/00-PDE/mspt
sbatch scripts/plas.sh
```

Other PDE launchers:

```bash
sbatch scripts/airfoil.sh
sbatch scripts/darcy.sh
sbatch scripts/elas.sh
sbatch scripts/ns.sh
sbatch scripts/pipe.sh
```

Available PDE datasets:
- `airfoil`
- `darcy`
- `elasticity`
- `ns`
- `pipe`
- `plasticity`

Evaluation uses the same dataset script with `--eval 1` and the same `--save_name` used during training. For example:

```bash
cd benchmarks/00-PDE/mspt
python plas.py \
  --model MSPT_Structured_Mesh_2D \
  --n-hidden 128 \
  --n-heads 8 \
  --n-layers 4 \
  --batch-size 8 \
  --unified_pos 0 \
  --ref 8 \
  --V 64 \
  --Q 1 \
  --eval 1 \
  --save_name MSPTPlasV32
```

PDE data, SLURM outputs, W&B runs, and checkpoints are ignored by git.

## Neural Solver Framework
The Neural Solver submodule remains available for the broader unified benchmark framework:

```bash
cd benchmarks/Neural-Solver-Library-MSPT
bash ./scripts/StandardBench/plasticity/MSPT.sh
```

Use this path when comparing many solver families through the shared `run.py` pipeline.

## Notes
- PDE training/evaluation entrypoints live in `benchmarks/00-PDE/mspt/`.
- The general Neural Solver pipeline lives in `benchmarks/Neural-Solver-Library-MSPT/`.
- The website in `docs/` is aligned with project figures and benchmark results.

## Citation
```bibtex
@inproceedings{curvo2026mspt,
  title={MSPT: Efficient Large-Scale Physical Modeling via Parallelized Multi-Scale Attention},
  author={Curvo, Pedro M. P. and van de Meent, Jan-Willem and Zhdanov, Maksim},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
  eprint={2512.01738},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2512.01738}
}
```
