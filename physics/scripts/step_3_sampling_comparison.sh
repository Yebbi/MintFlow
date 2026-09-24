#!/usr/bin/env bash
# Run the production 14-method benchmark for Heat, RD1D, or both.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# NERSC home uses a filesystem where uv's advisory cache lock can fail with
# error 524. Keep the cache on $SCRATCH (or locally outside Git elsewhere).
if [[ -z "${UV_CACHE_DIR:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    export UV_CACHE_DIR="$SCRATCH/.cache/uv"
  else
    export UV_CACHE_DIR="$REPO_ROOT/.uv-cache"
  fi
fi
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$UV_CACHE_DIR"

EQUATION="both"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --equation) EQUATION="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--equation heat|rd1d|both] [benchmark options]"
      exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

case "$EQUATION" in
  heat) uv run python scripts/run_heat_mass_12way_benchmark.py "${EXTRA[@]}" ;;
  rd1d) uv run python scripts/run_rd_global_balance_12way_benchmark.py "${EXTRA[@]}" ;;
  both)
    uv run python scripts/run_heat_mass_12way_benchmark.py "${EXTRA[@]}"
    uv run python scripts/run_rd_global_balance_12way_benchmark.py "${EXTRA[@]}"
    ;;
  *) echo "--equation must be heat, rd1d, or both" >&2; exit 2 ;;
esac
