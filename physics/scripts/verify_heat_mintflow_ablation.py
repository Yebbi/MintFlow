#!/usr/bin/env python3
"""Fail closed if a Heat ablation's metrics or paper assets are stale."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval_pde.artifacts import file_sha256
from scripts.run_heat_mintflow_ablation import study_for


def main(args: argparse.Namespace) -> None:
    study = study_for(args.scenario)
    root = args.root or study.root
    manifest = json.loads((root / "ablation_manifest.json").read_text())
    if manifest.get("scenario_key", "type1") != args.scenario:
        raise RuntimeError("Requested scenario does not match manifest")
    with (root / "ablation_summary.csv").open(newline="") as handle:
        table = {row["config_id"]: row for row in csv.DictReader(handle)}
    expected = {item["config_id"] for item in manifest["configurations"]}
    expected_n = int(manifest["num_samples_per_configuration"])
    if set(table) != expected:
        raise RuntimeError(f"Summary membership mismatch: {set(table) ^ expected}")
    reference_hash = file_sha256(study.reference)
    noise_hash = file_sha256(study.noise)
    scalar_keys = (
        "FID", "KID", "MMD", "W2", "conservation_R_inf_mean",
        "conservation_R_inf_max", "conservation_pass_rate_at_1e-5",
        "pde_residual_rms_mean", "simulation_relative_l2_mean",
        "time_per_sample_s", "total_NFE", "peak_gpu_memory_GiB",
        "selected_time_mean", "correction_norm_mean",
    )
    if "scenario_key" in manifest:
        scalar_keys += (
            "combined_normalized_R_inf_mean", "combined_normalized_R_inf_max",
            "combined_pass_rate_at_1e-5",
            "selected_candidate_earliest_fraction",
            "selected_candidate_interior_fraction",
            "selected_candidate_latest_fraction",
            "mintflow_pre_correction_normalized_R_inf_mean",
            "mintflow_pre_projection_normalized_R_inf_mean",
        )
    if args.scenario == "type2":
        scalar_keys += (
            "IC_R_inf_mean", "IC_R_inf_max", "IC_pass_rate_at_1e-5",
            "BC_R_inf_mean", "BC_R_inf_max", "BC_pass_rate_at_1e-5",
        )
    for config_id in sorted(expected):
        metric_path = root / "runs" / config_id / "metrics.json"
        sample_metric_path = root / "runs" / config_id / "per_sample_metrics.npz"
        payload = json.loads(metric_path.read_text())
        if payload["reference_sha256"] != reference_hash:
            raise RuntimeError(f"Reference hash mismatch: {config_id}")
        if payload["shared_noise_sha256"] != noise_hash:
            raise RuntimeError(f"Noise hash mismatch: {config_id}")
        if int(payload["n"]) != expected_n:
            raise RuntimeError(f"Sample count mismatch: {config_id}")
        for key in scalar_keys:
            if not np.isclose(float(table[config_id][key]), float(payload[key]),
                              rtol=1e-12, atol=1e-14, equal_nan=True):
                raise RuntimeError(f"Stale summary value {config_id}:{key}")
        with np.load(sample_metric_path) as per_sample:
            if any(
                np.asarray(per_sample[key]).shape != (expected_n,)
                for key in per_sample.files
            ):
                raise RuntimeError(f"Per-sample metric length mismatch: {config_id}")
    required = [root / "REPORT.md"]
    if "scenario_key" in manifest:
        required.append(root / "implementation_activation_audit.json")
    required += [root / "tables" / f"{group}.tex"
                 for group in manifest["factor_groups"]]
    required += [root / "figures" / f"{stem}.{suffix}"
                 for stem in ("fid_pass_pareto", "runtime_constraint_pareto")
                 for suffix in ("png", "pdf")]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing report artifacts: {missing}")
    result = {
        "status": "passed",
        "configurations": len(expected),
        "scenario": args.scenario,
        "samples_per_configuration": int(manifest["num_samples_per_configuration"]),
        "reference_sha256": reference_hash,
        "shared_noise_sha256": noise_hash,
        "raw_metrics_match_summary": True,
        "paper_assets_present": True,
    }
    (root / "integrity_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=("type1", "type2"), default="type1")
    p.add_argument("--root", type=Path)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
