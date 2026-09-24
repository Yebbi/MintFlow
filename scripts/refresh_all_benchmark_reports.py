#!/usr/bin/env python3
"""Regenerate and verify all four production benchmark reports."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.pipeline_common import run_logged

PYTHON = REPO / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class Benchmark:
    root: Path
    dataset: str
    task: str
    config: str
    seed: int


BENCHMARKS = (
    Benchmark(Path("results/heat_mass_12way_benchmark"), "diffusion",
              "heat_mass_conservation", "configs/heat.yml", 20260811),
    Benchmark(Path("results/heat_mass_scenario2_benchmark"), "diffusion",
              "heat_mass_conservation", "configs/heat.yml", 20260811),
    Benchmark(Path("results/rd_global_balance_12way_benchmark"), "rd1d",
              "global_balance", "configs/rd1d.yml", 20260820),
    Benchmark(Path("results/rd_global_balance_scenario2_benchmark"), "rd1d",
              "global_balance", "configs/rd1d.yml", 20260820),
)


def main(args: argparse.Namespace) -> None:
    for benchmark in BENCHMARKS:
        root = benchmark.root
        common = [
            "--benchmark-root", str(root), "--dataset", benchmark.dataset,
            "--task", benchmark.task, "--config", benchmark.config,
        ]
        run_logged([
            str(PYTHON), "scripts/evaluate_type1_benchmark.py", *common,
            "--seed", str(benchmark.seed),
        ], root / "logs" / "evaluate_production.log")
        run_logged([
            str(PYTHON), "scripts/evaluate_physical_consistency.py", *common,
            "--workers", str(args.physical_workers),
        ], root / "logs" / "physical_consistency_production.log")
        run_logged([
            str(PYTHON), "scripts/analyze_type1_benchmark.py", *common,
            "--figure-seed", str(args.figure_seed),
        ], root / "logs" / "analyze_production.log")
        run_logged([
            str(PYTHON), "scripts/verify_benchmark_integrity.py", *common,
        ], root / "logs" / "integrity_production.log")
        print(f"VERIFIED {root}", flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-workers", type=int, default=16)
    p.add_argument("--figure-seed", type=int, default=20260902)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
