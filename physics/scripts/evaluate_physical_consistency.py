#!/usr/bin/env python3
"""Evaluate local PDE residuals and same-IC solver errors for a benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_pde.artifacts import write_csv
from eval_pde.config import build_pde_config
from eval_pde.physical_consistency import evaluate_physical_consistency


def main(args: argparse.Namespace) -> None:
    root = Path(args.benchmark_root).resolve()
    manifest = json.loads((root / "benchmark_manifest.json").read_text())
    sample_sets = {
        method: np.load(Path(run) / "samples.npy", mmap_mode="r")
        for method, run in manifest["runs"].items()
    }
    vanilla = np.asarray(sample_sets["vanilla_ffm"])
    cfg = build_pde_config(args.dataset, task=args.task, config_path=args.config)
    target_bank = None
    if manifest.get("constraint_scenario") == "conservation_ic_bc":
        from sampling.scenario_targets import load_scenario_target_bank

        target_bank = load_scenario_target_bank(
            manifest["constraint_target_file"], args.dataset
        )
    per_sample, summary, metadata = evaluate_physical_consistency(
        sample_sets, vanilla, cfg, workers=args.workers,
        scenario_target_bank=target_bank,
    )
    evaluation = root / "evaluation"
    tables = root / "tables"
    write_csv(evaluation / "physical_consistency_per_sample.csv", per_sample)
    write_csv(tables / "physical_consistency_summary.csv", summary)
    (evaluation / "physical_consistency_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote physical-consistency metrics for {len(sample_sets)} methods to {root}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--dataset", required=True, choices=("diffusion", "rd1d"))
    p.add_argument("--task", required=True, choices=("heat_mass_conservation", "global_balance"))
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=16)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
