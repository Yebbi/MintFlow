#!/usr/bin/env bash
# Generate the only persisted production dataset: RD1D train/test fields.
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

ROOT="datasets/data"
SEED=42
NPROC=64
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--root PATH] [--seed N] [--nproc N]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

uv run python -m datasets.generate_all_datasets \
  --root "$ROOT" --seed "$SEED" --nproc "$NPROC"
