#!/usr/bin/env python3
"""Evaluate completed Scenario 1/2 Heat or RD1D samples without resampling."""
from __future__ import annotations

import argparse

from eval_pde.config import build_pde_config
from eval_pde.type1_evaluation import evaluate_type1_benchmark


def main(args: argparse.Namespace) -> None:
    cfg = build_pde_config(args.dataset, task=args.task, config_path=args.config)
    outputs = evaluate_type1_benchmark(args.benchmark_root, cfg, seed=args.seed)
    print("Wrote streamlined benchmark metrics:")
    for category, path in outputs.items():
        print(f"  {category}: {path}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--dataset", required=True, choices=("diffusion", "rd1d"))
    p.add_argument("--task", required=True, choices=("heat_mass_conservation", "global_balance"))
    p.add_argument("--config", required=True)
    p.add_argument("--seed", required=True, type=int)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
