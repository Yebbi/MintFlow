# MintFlow: Physical System Modeling

## Environment Setup

Python 3.10--3.12 is supported; Python 3.11 is used for the reported runs. The locked environment installs PyTorch's CUDA 12.1 wheels and therefore requires an NVIDIA driver compatible with CUDA 12.1. The A100 GPUs on NERSC Perlmutter are the reference hardware.

The recommended installation uses `uv`:

```bash
uv sync --group dev
bash scripts/step_0_environment_setup.sh
```

## Data Generation and Pretraining

Heat training trajectories are generated analytically on demand from the parameter ranges in `configs/heat.yml`; there is no persisted Heat training file. RD1D uses finite-volume solver outputs in HDF5. Generate its 6,400-train and 256-test trajectory banks with:

```bash
bash scripts/step_1_generate_datasets.sh --root datasets/data --seed 42 --nproc 64
```

Pretrain one or both flow-matching models:

```bash
bash scripts/step_2_pretrain.sh \
  --datasets diffusion,rd1d --device cuda --wandb-mode disabled
```

## Reproducing the Benchmarks

```bash
# Heat: mass + paired IC + structural periodic BC
uv run python scripts/run_heat_mass_12way_benchmark.py \
  --constraint-scenario conservation_ic_bc \
  --num-samples 1000 --reference-samples 1000

# RD1D: global balance + paired IC + Neumann BC
uv run python scripts/run_rd_global_balance_12way_benchmark.py \
  --constraint-scenario conservation_ic_bc \
  --num-samples 1000 --reference-samples 1000
```