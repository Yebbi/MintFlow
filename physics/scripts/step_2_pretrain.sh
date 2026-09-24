#!/usr/bin/env bash
# Pretrain the Heat and/or RD1D FFM model through the uv environment.
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

DATASETS="diffusion,rd1d"
DEVICE="cuda"
WANDB_MODE_ARG="disabled"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE_ARG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--datasets diffusion,rd1d] [--device cuda] [--wandb-mode MODE] [--dry-run]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export WANDB_MODE="$WANDB_MODE_ARG"
IFS=',' read -ra REQUESTED <<< "$DATASETS"
for dataset in "${REQUESTED[@]}"; do
  case "$dataset" in
    diffusion) config="configs/heat.yml" ;;
    rd1d) config="configs/rd1d.yml" ;;
    *) echo "Unsupported production dataset: $dataset" >&2; exit 2 ;;
  esac
  command=(uv run python scripts/training/main.py "$config" --device "$DEVICE" --savename "$dataset")
  printf '%q ' "${command[@]}"; printf '\n'
  [[ "$DRY_RUN" == true ]] || "${command[@]}"
done
