#!/usr/bin/env python3
"""Cross-check raw samples, canonical CSV tables, and generated REPORT.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval_pde.artifacts import file_sha256, read_csv
from eval_pde.config import build_pde_config
from eval_pde.type1_evaluation import (
    METHOD_ORDER,
    constraint_rows,
    distribution_rows,
    efficiency_rows,
    load_benchmark_runs,
)
from sampling.scenario_targets import load_scenario_target_bank
from scripts.analyze_type1_benchmark import (
    _constraint_table,
    _distribution_table,
    _efficiency_table,
    _physical_table,
)


def _assert_rows_close(actual: list[dict], expected: list[dict], label: str) -> None:
    if [row["method"] for row in actual] != [row["method"] for row in expected]:
        raise AssertionError(f"{label}: method ordering mismatch")
    for actual_row, expected_row in zip(actual, expected):
        for key, expected_value in expected_row.items():
            if key not in actual_row:
                raise AssertionError(f"{label}: missing {key} for {actual_row['method']}")
            if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                if not np.isclose(
                    float(actual_row[key]), float(expected_value), rtol=1e-10, atol=1e-12,
                    equal_nan=True,
                ):
                    raise AssertionError(
                        f"{label}: {actual_row['method']} {key}: "
                        f"{actual_row[key]} != {expected_value}"
                    )
            elif str(actual_row[key]) != str(expected_value):
                raise AssertionError(
                    f"{label}: {actual_row['method']} {key}: "
                    f"{actual_row[key]!r} != {expected_value!r}"
                )


def _physical_from_per_sample(path: Path) -> list[dict]:
    rows = read_csv(path)
    result = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 1000:
            raise AssertionError(f"physical: {method} has {len(selected)} rows")
        row = {"method": method, "n": len(selected)}
        for source, prefix in (
            ("pde_residual_rms", "pde_residual_rms"),
            ("simulation_relative_l2", "simulation_relative_l2"),
        ):
            values = np.asarray([float(item[source]) for item in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            row.update({
                f"{prefix}_mean": float(np.mean(finite)),
                f"{prefix}_std": float(np.std(finite, ddof=1)),
                f"{prefix}_median": float(np.median(finite)),
                f"{prefix}_n_finite": len(finite),
            })
        result.append(row)
    return result


def main(args: argparse.Namespace) -> None:
    root = args.benchmark_root.resolve()
    cfg = build_pde_config(args.dataset, task=args.task, config_path=args.config)
    manifest, sample_sets, metadata = load_benchmark_runs(root)
    if manifest["num_samples"] != 1000 or manifest["reference_samples"] != 1000:
        raise AssertionError("Production sample/reference counts must both be 1000")

    target_bank = None
    if manifest["constraint_scenario"] == "conservation_ic_bc":
        target_bank = load_scenario_target_bank(manifest["constraint_target_file"], cfg.name)
    expected_constraints = constraint_rows(
        sample_sets, cfg, scenario_target_bank=target_bank
    )
    expected_efficiency = efficiency_rows(sample_sets, metadata)
    reference = np.load(manifest["numerical_reference"], mmap_mode="r")
    expected_distribution, _ = distribution_rows(
        sample_sets, reference, cfg, root / "evaluation", manifest["seed"]
    )
    expected_physical = _physical_from_per_sample(
        root / "evaluation" / "physical_consistency_per_sample.csv"
    )

    actual = {
        "distribution": read_csv(root / "tables" / "distribution_metrics.csv"),
        "efficiency": read_csv(root / "tables" / "efficiency_metrics.csv"),
        "constraints": read_csv(root / "tables" / "constraint_metrics.csv"),
        "physical": read_csv(root / "tables" / "physical_consistency_summary.csv"),
    }
    _assert_rows_close(actual["distribution"], expected_distribution, "distribution")
    _assert_rows_close(actual["efficiency"], expected_efficiency, "efficiency")
    _assert_rows_close(actual["constraints"], expected_constraints, "constraints")
    _assert_rows_close(actual["physical"], expected_physical, "physical")

    report = (root / "REPORT.md").read_text()
    expected_tables = (
        _distribution_table(actual["distribution"]),
        _efficiency_table(actual["efficiency"]),
        _constraint_table(
            actual["constraints"], scenario=manifest["constraint_scenario"],
            equation=cfg.name,
        ),
        _physical_table(actual["physical"]),
    )
    if any(table not in report for table in expected_tables):
        raise AssertionError("REPORT.md does not contain the exact canonical CSV tables")

    qualitative_path = root / "figures" / "qualitative_manifest.json"
    qualitative = json.loads(qualitative_path.read_text())
    expected_qualitative = [
        "vanilla_ffm", "dflow", "diffusionpde", "eci_native_standard",
        "pcfm_standard", "mintflow_pseudoinverse_with_final_projection",
    ]
    if [item["configuration"] for item in qualitative["methods"]] != expected_qualitative:
        raise AssertionError("Qualitative method ordering is stale or inconsistent")
    if len(set(qualitative["sample_ids"])) != 7:
        raise AssertionError("Qualitative figures must contain seven fixed sample IDs")
    figure_files = []
    for index in range(7):
        for stem in ("sample_comparison", "physical_consistency"):
            path = root / "figures" / f"{stem}_{index:03d}.png"
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"Missing qualitative artifact: {path}")
            if not path.read_bytes().endswith(b"\x00\x00\x00\x00IEND\xaeB\x60\x82"):
                raise AssertionError(f"Incomplete or padded PNG artifact: {path}")
            figure_files.append(path)

    primary = metadata["mintflow_pseudoinverse_with_final_projection"]
    signature = (
        primary["mintflow_forward_steps"], primary["mintflow_num_candidates"],
        primary["mintflow_candidate_t_max"], primary["mintflow_time_penalty"],
        primary["mintflow_correction_scale"], primary["mintflow_time_sampling"],
        primary["mintflow_correction_mode"],
        primary["post_sampling_final_projection_enabled"],
    )
    if signature != (200, 10, 0.98, 0.01, 1.0, "end_biased", "pseudoinverse", True):
        raise AssertionError(f"Primary MintFlow signature mismatch: {signature}")

    files = [
        root / "REPORT.md", root / "benchmark_manifest.json",
        root / "tables" / "distribution_metrics.csv",
        root / "tables" / "efficiency_metrics.csv",
        root / "tables" / "constraint_metrics.csv",
        root / "tables" / "physical_consistency_summary.csv",
        root / "evaluation" / "physical_consistency_per_sample.csv",
        root / "evaluation" / "frozen_pod_embedding.npz",
        qualitative_path,
        *figure_files,
    ]
    result = {
        "status": "pass",
        "raw_methods_verified": len(METHOD_ORDER),
        "samples_per_method": 1000,
        "reference_samples": len(reference),
        "report_tables_match_csv": True,
        "csv_metrics_recomputed_from_raw": True,
        "physical_summary_recomputed_from_per_sample": True,
        "qualitative_artifacts_verified": len(figure_files),
        "primary_mintflow_signature": list(signature),
        "sha256": {str(path.relative_to(root)): file_sha256(path) for path in files},
    }
    output = root / "evaluation" / "integrity_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=("diffusion", "rd1d"))
    p.add_argument("--task", required=True, choices=("heat_mass_conservation", "global_balance"))
    p.add_argument("--config", required=True)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
