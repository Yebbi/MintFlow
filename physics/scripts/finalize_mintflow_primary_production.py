#!/usr/bin/env python3
"""Merge production MintFlow shards and update the four benchmark manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.merge_sampling_shards import merge

PRODUCTION_CONFIG = {
    "steps": 200,
    "num_candidates": 10,
    "candidate_t_min": 0.1,
    "candidate_t_max": 0.98,
    "time_sampling": "end_biased",
    "end_bias_power": 2.0,
    "time_penalty": 0.01,
    "correction_scale": 1.0,
    "correction_mode": "pseudoinverse",
    "final_projection": True,
    "projection_tolerance": 1e-6,
    "evaluation_pass_threshold": 1e-5,
}

ROOTS = {
    "heat1": Path("results/heat_mass_12way_benchmark"),
    "heat2": Path("results/heat_mass_scenario2_benchmark"),
    "rd1": Path("results/rd_global_balance_12way_benchmark"),
    "rd2": Path("results/rd_global_balance_scenario2_benchmark"),
}


def _metadata_matches(path: Path, *, final_projection: bool = True) -> bool:
    try:
        metadata = json.loads((path / "metadata.json").read_text())
    except (FileNotFoundError, OSError, ValueError):
        return False
    expected = (
        int(metadata.get("num_samples", -1)),
        int(metadata.get("mintflow_forward_steps", -1)),
        int(metadata.get("mintflow_num_candidates", -1)),
        float(metadata.get("mintflow_candidate_t_max", -1.0)),
        float(metadata.get("mintflow_time_penalty", -1.0)),
        float(metadata.get("mintflow_correction_scale", -1.0)),
        metadata.get("mintflow_time_sampling"),
        metadata.get("mintflow_correction_mode"),
        bool(metadata.get("post_sampling_final_projection_enabled", False)),
    )
    return expected == (1000, 200, 10, 0.98, 0.01, 1.0,
                        "end_biased", "pseudoinverse", final_projection)


def _shard_runs(root: Path) -> list[Path]:
    """Select an exact tiling of sample IDs 0..999 from valid mixed-size shards."""
    candidates: dict[int, list[tuple[int, Path]]] = {}
    shard_root = root / "production_primary_shards"
    for metadata_path in shard_root.rglob("metadata.json") if shard_root.exists() else ():
        try:
            metadata = json.loads(metadata_path.read_text())
            offset = int(metadata.get("sample_offset", -1))
            count = int(metadata.get("num_samples", -1))
        except (OSError, TypeError, ValueError):
            continue
        samples_path = metadata_path.parent / "samples.npy"
        if (
            offset >= 0 and count >= 1 and offset + count <= 1000
            and _metadata_matches_shard(metadata) and samples_path.is_file()
        ):
            samples = np.load(samples_path, mmap_mode="r")
            ids = metadata.get("sample_ids", list(range(offset, offset + count)))
            if len(samples) == count and ids == list(range(offset, offset + count)):
                candidates.setdefault(offset, []).append(
                    (offset + count, metadata_path.parent)
                )

    runs: list[Path] = []
    position = 0
    while position < 1000:
        options = candidates.get(position, [])
        if not options:
            raise RuntimeError(
                f"Production shards do not cover sample ID {position} under {shard_root}"
            )
        # Prefer the largest valid interval; mtime breaks duplicate-run ties.
        stop, run = max(options, key=lambda item: (item[0], item[1].stat().st_mtime))
        runs.append(run)
        position = stop
    return runs


def _metadata_matches_shard(metadata: dict) -> bool:
    return (
        int(metadata.get("mintflow_forward_steps", -1)),
        int(metadata.get("mintflow_num_candidates", -1)),
        float(metadata.get("mintflow_candidate_t_max", -1.0)),
        float(metadata.get("mintflow_time_penalty", -1.0)),
        float(metadata.get("mintflow_correction_scale", -1.0)),
        metadata.get("mintflow_time_sampling"),
        metadata.get("mintflow_correction_mode"),
        bool(metadata.get("post_sampling_final_projection_enabled", False)),
    ) == (200, 10, 0.98, 0.01, 1.0, "end_biased", "pseudoinverse", True)


def _merge_scenario(root: Path) -> Path:
    dataset = "diffusion" if "heat" in root.name else "rd1d"
    output = (
        root / "raw" / "mintflow_pseudoinverse_with_final_projection"
        / dataset / "mintflow" / "production_steps200_k10_smax0p98_n1000"
    )
    if output.is_dir() and _metadata_matches(output):
        return output.resolve()
    if output.exists():
        raise RuntimeError(f"Invalid production output already exists: {output}")
    return merge(_shard_runs(root), output, 1000).resolve()


def _update_manifest(root: Path, production_run: Path) -> None:
    manifest_path = root / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"]["mintflow_pseudoinverse_with_final_projection"] = str(
        production_run.resolve()
    )
    manifest["mintflow_production_config"] = PRODUCTION_CONFIG
    manifest["primary_mintflow_configuration"] = (
        "mintflow_pseudoinverse_with_final_projection"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(args: argparse.Namespace) -> None:
    heat1 = ROOTS["heat1"]
    heat1_run = Path(
        "results/heat_mass_12way_benchmark/raw/"
        "mintflow_pseudoinverse_with_final_projection/diffusion/mintflow/"
        "production_steps200_k10_smax0p98_n1000"
    )
    if not _metadata_matches(heat1_run):
        raise RuntimeError("Validated Heat Scenario-1 production pool is missing or stale")
    _update_manifest(heat1, heat1_run)
    print(f"heat1: {heat1_run}")

    selected = (
        ("heat2", "rd1", "rd2")
        if args.scenario == "all"
        else (() if args.scenario == "heat1" else (args.scenario,))
    )
    for name in selected:
        root = ROOTS[name]
        production_run = _merge_scenario(root)
        _update_manifest(root, production_run)
        print(f"{name}: {production_run}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario", choices=("all", "heat1", "heat2", "rd1", "rd2"), default="all"
    )
    return p


if __name__ == "__main__":
    main(parser().parse_args())
