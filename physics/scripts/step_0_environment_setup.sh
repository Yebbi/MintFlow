#!/usr/bin/env bash
# Synchronize and validate the production uv environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${UV_CACHE_DIR:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    export UV_CACHE_DIR="$SCRATCH/.cache/uv"
  else
    export UV_CACHE_DIR="$REPO_ROOT/.uv-cache"
  fi
fi
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$UV_CACHE_DIR"

uv sync --group dev
uv run python - <<'PY'
import h5py
import gpytorch
import matplotlib
import neuralop
import numpy
import scienceplots
import torch
import torchdiffeq
import tqdm

from datasets import get_dataset
from models import get_flow_model
from sampling.methods import run_vanilla_sampling

print(f"python environment OK; torch={torch.__version__}; cuda={torch.cuda.is_available()}")
PY

for required in \
  configs/heat.yml configs/rd1d.yml \
  scripts/step_1_generate_datasets.sh scripts/step_2_pretrain.sh \
  scripts/step_3_sampling_comparison.sh \
  scripts/run_heat_mass_12way_benchmark.py \
  scripts/run_rd_global_balance_12way_benchmark.py; do
  [[ -f "$required" ]] || { echo "Missing production file: $required" >&2; exit 1; }
done

echo "Production environment validation passed."
