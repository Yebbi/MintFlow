"""Streamlined metrics for paired Scenario 1 and Scenario 2 benchmarks.

The evaluation contract is deliberately narrow:

* constraints are evaluated in the raw field representation;
* FID, KID, MMD, and empirical Wasserstein-2 are evaluated in one frozen POD
  space fitted exclusively on the independent numerical reference pool;
* efficiency comes only from synchronized sampler wall time and measured peak
  CUDA memory; and
* local PDE residual and same-IC simulation errors live in
  :mod:`eval_pde.physical_consistency`.

No Vanilla-reference distribution metric or paired-displacement diagnostic is
defined in this module.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_csv
from .config import PDEConfig, load_grid
from .distances import (
    empirical_wasserstein2,
    frechet_feature_distance,
    median_pairwise_distance,
    mmd,
    polynomial_kid,
)
from .embeddings import fit_or_load_frozen_pod
from .residuals.reaction_diffusion import mass_residual_rd_batched

METHOD_ORDER = (
    "vanilla_ffm",
    "pcfm_standard",
    "pcfm_intermediate_only",
    "eci_native_standard",
    "eci_native_intermediate_only",
    "eci_gn_standard",
    "eci_gn_intermediate_only",
    "diffusionpde",
    "dflow",
    "mintflow_pseudoinverse_no_final_projection",
    "mintflow_pseudoinverse_with_final_projection",
    "mintflow_damped_no_final_projection",
    "mintflow_damped_with_final_projection",
    "standalone_final_projection",
)

DISPLAY_NAMES = {
    "vanilla_ffm": "Vanilla FFM",
    "pcfm_standard": "PCFM standard",
    "pcfm_intermediate_only": "PCFM intermediate-only",
    "eci_native_standard": "ECI-native standard",
    "eci_native_intermediate_only": "ECI-native intermediate-only",
    "eci_gn_standard": "ECI-GN standard",
    "eci_gn_intermediate_only": "ECI-GN intermediate-only",
    "diffusionpde": "DiffusionPDE",
    "dflow": "D-Flow",
    "mintflow_pseudoinverse_no_final_projection": "MintFlow-pinv -FP",
    "mintflow_pseudoinverse_with_final_projection": "MintFlow-pinv +FP",
    "mintflow_damped_no_final_projection": "MintFlow-damped -FP",
    "mintflow_damped_with_final_projection": "MintFlow-damped +FP",
    "standalone_final_projection": "Standalone FP",
}

QUALITATIVE_METHODS = (
    ("vanilla_ffm", "Vanilla FFM"),
    ("dflow", "D-Flow"),
    ("diffusionpde", "DiffusionPDE"),
    ("eci_native_standard", "ECI"),
    ("pcfm_standard", "PCFM"),
    ("mintflow_pseudoinverse_with_final_projection", "MintFlow"),
)


def _run_path(value: str | Mapping[str, Any]) -> Path:
    raw = value.get("path") if isinstance(value, Mapping) else value
    if not raw:
        raise ValueError(f"Invalid run entry: {value!r}")
    return Path(str(raw)).resolve()


def _noise_path(metadata: Mapping[str, Any]) -> Path | None:
    value = metadata.get("initial_noise_path")
    if not value:
        return None
    path = Path(str(value)).resolve()
    return path / "initial_noise.npy" if path.is_dir() else path


def load_benchmark_runs(
    benchmark_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Load and validate the complete, shared-noise Type 1 run manifest."""
    root = Path(benchmark_root).resolve()
    manifest = json.loads((root / "benchmark_manifest.json").read_text())
    runs = manifest.get("runs", {})
    missing = [method for method in METHOD_ORDER if method not in runs]
    extra = [method for method in runs if method not in METHOD_ORDER]
    if missing or extra:
        raise ValueError(f"Run manifest mismatch: missing={missing}, extra={extra}")

    sample_sets: dict[str, np.ndarray] = {}
    metadata_by_method: dict[str, dict[str, Any]] = {}
    expected_ids: np.ndarray | None = None
    expected_shape: tuple[int, ...] | None = None
    expected_noise: Path | None = None
    for method in METHOD_ORDER:
        run = _run_path(runs[method])
        samples_path, metadata_path = run / "samples.npy", run / "metadata.json"
        if not samples_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Incomplete run for {method}: {run}")
        samples = np.load(samples_path, mmap_mode="r")
        metadata = json.loads(metadata_path.read_text())
        ids = np.asarray(metadata.get("sample_ids", np.arange(len(samples))), dtype=np.int64)
        if expected_ids is None:
            expected_ids = ids
            expected_shape = samples.shape
            expected_noise = _noise_path(metadata)
        if samples.shape != expected_shape or not np.array_equal(ids, expected_ids):
            raise ValueError(f"{method}: samples or sample IDs are not aligned")
        noise = _noise_path(metadata)
        if expected_noise is not None and noise is not None and noise != expected_noise:
            raise ValueError(f"{method}: initial-noise bank differs from Vanilla FFM")
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"{method}: non-finite raw samples are not permitted")
        sample_sets[method] = samples
        metadata_by_method[method] = metadata
    if expected_ids is None or not np.array_equal(expected_ids, np.arange(len(expected_ids))):
        raise ValueError("Type 1 runs require ordered sample IDs 0..N-1")
    return manifest, sample_sets, metadata_by_method


def constraint_values(samples: np.ndarray, cfg: PDEConfig) -> np.ndarray:
    """Return one raw-space conservation violation per sample."""
    values = np.asarray(samples, dtype=np.float64)
    x, t = load_grid(cfg)
    if cfg.name == "diffusion":
        dx = float((cfg.x_max - cfg.x_min) / cfg.nx)
        mass = dx * np.sum(values, axis=1)
        residual = mass[:, 1:] - mass[:, [0]]
    elif cfg.name == "rd1d":
        residual = mass_residual_rd_batched(
            values,
            x=x,
            t_grid=t,
            rho=float(cfg.rd_rho),
            nu=float(cfg.rd_nu),
        )
    else:
        raise ValueError(f"Unsupported Type 1 equation: {cfg.name!r}")
    return np.max(np.abs(residual), axis=1)


def scenario2_constraint_components(
    samples: np.ndarray, cfg: PDEConfig, bank,
) -> dict[str, np.ndarray]:
    """Return raw per-sample Scenario-2 component violations.

    Every component is reduced to an ``L-infinity`` value over its own
    residual vector before statistics are aggregated across samples.  The
    returned conservation, IC, and BC values remain in their physical units.
    ``combined_normalized`` is the sampler feasibility statistic: RD1D flux
    errors are divided by the task scale ``0.05`` before taking the maximum.
    Heat has no stored BC residual because periodicity is structural on its
    endpoint-excluded grid, so its BC value is identically zero.
    """
    values = np.asarray(samples, dtype=np.float64)
    if len(values) != bank.num_targets:
        raise ValueError("Scenario-2 samples and target bank must pair one-to-one")
    ic = np.max(
        np.abs(values[:, :, 0] - bank.initial_conditions.astype(np.float64)),
        axis=1,
    )
    if cfg.name == "diffusion":
        conservation = constraint_values(values, cfg)
        boundary = np.zeros(len(values), dtype=np.float64)
        return {
            "conservation": conservation,
            "initial_condition": ic,
            "boundary_condition": boundary,
            "combined_normalized": np.maximum(conservation, ic),
        }

    x, t = load_grid(cfg)
    dx = float(x[1] - x[0])
    nu = float(cfg.rd_nu)
    # Scenario 2 supplies the physical Neumann flux. Its conservation law
    # therefore uses that prescribed flux, rather than independently
    # reconstructing a second flux from the generated boundary cells (the
    # self-contained Scenario-1 convention).
    mass = values.sum(axis=1) * dx
    source = float(cfg.rd_rho) * (values * (1.0 - values)).sum(axis=1) * dx
    source_mid = 0.5 * (source[:, :-1] + source[:, 1:])
    source_cumulative = np.concatenate([
        np.zeros((len(values), 1), dtype=np.float64),
        np.cumsum(source_mid * np.diff(t)[None, :], axis=1),
    ], axis=1)
    boundary_net = (
        bank.boundary_values[:, 0] - bank.boundary_values[:, 1]
    ).astype(np.float64)
    boundary_cumulative = np.concatenate([
        np.zeros((len(values), 1), dtype=np.float64),
        np.cumsum(boundary_net[:, None] * np.diff(t)[None, :], axis=1),
    ], axis=1)
    conservation = np.max(np.abs(
        mass[:, 1:]
        - (mass[:, :1] + source_cumulative + boundary_cumulative)[:, 1:]
    ), axis=1)
    left = -nu * (
        -25.0 * values[:, 0] + 48.0 * values[:, 1]
        - 36.0 * values[:, 2] + 16.0 * values[:, 3] - 3.0 * values[:, 4]
    ) / (12.0 * dx)
    right = -nu * (
        25.0 * values[:, -1] - 48.0 * values[:, -2]
        + 36.0 * values[:, -3] - 16.0 * values[:, -4]
        + 3.0 * values[:, -5]
    ) / (12.0 * dx)
    boundary = np.maximum(
        np.max(np.abs(left[:, 1:] - bank.boundary_values[:, [0]]), axis=1),
        np.max(np.abs(right[:, 1:] - bank.boundary_values[:, [1]]), axis=1),
    )
    return {
        "conservation": conservation,
        "initial_condition": ic,
        "boundary_condition": boundary,
        "combined_normalized": np.maximum.reduce([
            conservation, ic, boundary / 0.05,
        ]),
    }


def constraint_rows(
    sample_sets: Mapping[str, np.ndarray], cfg: PDEConfig,
    *, scenario_target_bank=None, tolerance: float = 1e-5,
) -> list[dict[str, Any]]:
    """Summarize raw constraint components and normalized feasibility.

    Scenario 1 contains only the conservation residual, hence its
    ``conservation_R_inf`` is also its total residual.  Scenario 2 reports
    conservation, IC, and BC separately so a small combined statistic cannot
    conceal which physical condition was or was not satisfied.
    """
    rows = []
    for method in METHOD_ORDER:
        conservation = constraint_values(sample_sets[method], cfg)
        row = {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "n": len(conservation),
            "conservation_R_inf_mean": float(np.mean(conservation)),
            "conservation_R_inf_std": float(np.std(conservation, ddof=1)),
            "conservation_R_inf_max": float(np.max(conservation)),
        }
        if scenario_target_bank is None:
            row["conservation_violation_rate_at_1e-5"] = float(
                np.mean(conservation > tolerance)
            )
            row["conservation_pass_rate_at_1e-5"] = float(
                np.mean(conservation <= tolerance)
            )
        else:
            components = scenario2_constraint_components(
                sample_sets[method], cfg, scenario_target_bank
            )
            # Scenario-2 conservation can differ from the Scenario-1
            # self-inferred-flux definition for RD1D, so overwrite it with
            # the prescribed-external-flux component used by the sampler.
            conservation = components["conservation"]
            row.update({
                "conservation_R_inf_mean": float(np.mean(conservation)),
                "conservation_R_inf_std": float(
                    np.std(conservation, ddof=1)
                ),
                "conservation_R_inf_max": float(np.max(conservation)),
                "IC_R_inf_mean": float(
                    np.mean(components["initial_condition"])
                ),
                "IC_R_inf_std": float(
                    np.std(components["initial_condition"], ddof=1)
                ),
                "IC_R_inf_max": float(
                    np.max(components["initial_condition"])
                ),
                "BC_R_inf_mean": float(
                    np.mean(components["boundary_condition"])
                ),
                "BC_R_inf_std": float(
                    np.std(components["boundary_condition"], ddof=1)
                ),
                "BC_R_inf_max": float(
                    np.max(components["boundary_condition"])
                ),
                "combined_normalized_R_inf_mean": float(
                    np.mean(components["combined_normalized"])
                ),
                "combined_normalized_R_inf_std": float(
                    np.std(components["combined_normalized"], ddof=1)
                ),
                "combined_normalized_R_inf_max": float(
                    np.max(components["combined_normalized"])
                ),
                "combined_violation_rate_at_1e-5": float(np.mean(
                    components["combined_normalized"] > tolerance
                )),
                "conservation_pass_rate_at_1e-5": float(np.mean(
                    components["conservation"] <= tolerance
                )),
                "IC_pass_rate_at_1e-5": float(np.mean(
                    components["initial_condition"] <= tolerance
                )),
                "BC_pass_rate_at_1e-5": float(np.mean(
                    components["boundary_condition"] <= tolerance
                )),
                "combined_pass_rate_at_1e-5": float(np.mean(
                    components["combined_normalized"] <= tolerance
                )),
            })
        rows.append(row)
    return rows


def distribution_rows(
    sample_sets: Mapping[str, np.ndarray],
    numerical_reference: np.ndarray,
    cfg: PDEConfig,
    evaluation_dir: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate only against the independent numerical reference pool."""
    reference = np.asarray(numerical_reference)
    counts = {method: len(samples) for method, samples in sample_sets.items()}
    if any(count != len(reference) for count in counts.values()):
        raise ValueError(
            "Exact empirical W2 requires every evaluated set and the numerical "
            f"reference to have equal size: reference={len(reference)}, methods={counts}"
        )
    embedding_path = Path(evaluation_dir) / "frozen_pod_embedding.npz"
    embedding = fit_or_load_frozen_pod(embedding_path, reference, cfg)
    reference_features = embedding.transform(reference)
    median_distance = median_pairwise_distance(reference_features, seed=seed)
    rows = []
    for method in METHOD_ORDER:
        features = embedding.transform(sample_sets[method])
        mmd_result = mmd(
            features,
            reference_features,
            seed=seed,
            median_distance=median_distance,
        )
        rows.append({
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "n": len(features),
            "FID": frechet_feature_distance(features, reference_features),
            "KID": polynomial_kid(features, reference_features),
            "MMD": float(np.sqrt(mmd_result["mmd2_biased_clipped"])),
            "W2": empirical_wasserstein2(features, reference_features),
        })
    metadata = {
        "reference": "independent numerical ground-truth pool",
        "reference_contains_ffm_samples": False,
        "reference_n": len(reference),
        "embedding_file": str(embedding_path.resolve()),
        "embedding": embedding.metadata(),
        "mmd": (
            "square root of the nonnegative biased multiscale-RBF MMD^2; "
            "bandwidth median fitted once on numerical-reference POD coefficients"
        ),
        "wasserstein": "exact equal-weight empirical W2 in frozen POD space",
        "seed": int(seed),
    }
    return rows, metadata


def efficiency_rows(
    sample_sets: Mapping[str, np.ndarray],
    metadata_by_method: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for method in METHOD_ORDER:
        metadata = metadata_by_method[method]
        total = float(metadata.get("generation_time_s", np.nan))
        peak = float(metadata.get("peak_gpu_memory_bytes", np.nan))
        rows.append({
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "n": len(sample_sets[method]),
            "time_per_sample_s": total / len(sample_sets[method]),
            "peak_gpu_memory_GiB": peak / (1024.0**3),
        })
    return rows


def evaluate_type1_benchmark(
    benchmark_root: Path,
    cfg: PDEConfig,
    *,
    seed: int,
) -> dict[str, Path]:
    """Generate the three non-physical metric-category tables."""
    root = Path(benchmark_root).resolve()
    evaluation_dir, tables_dir = root / "evaluation", root / "tables"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest, sample_sets, metadata = load_benchmark_runs(root)
    reference_path = Path(manifest["numerical_reference"]).resolve()
    reference = np.load(reference_path, mmap_mode="r")
    reference_metadata_path = reference_path.with_name(
        "numerical_reference_metadata.json"
    )
    reference_metadata = (
        json.loads(reference_metadata_path.read_text())
        if reference_metadata_path.exists() else {}
    )

    scenario_target_bank = None
    if manifest.get("constraint_scenario") == "conservation_ic_bc":
        from sampling.scenario_targets import load_scenario_target_bank

        scenario_target_bank = load_scenario_target_bank(
            manifest["constraint_target_file"], cfg.name
        )
    constraints = constraint_rows(
        sample_sets, cfg, scenario_target_bank=scenario_target_bank
    )
    distribution, distribution_metadata = distribution_rows(
        sample_sets, reference, cfg, evaluation_dir, seed
    )
    distribution_metadata.update({
        "reference": reference_metadata.get(
            "source", "numerical ground-truth pool"
        ),
        "reference_target_id": reference_metadata.get("target_id"),
        "reference_pairing": reference_metadata.get("target_pairing"),
        "reference_uses_ffm_derived_conditions": (
            reference_metadata.get("source")
            == "paired_scenario2_numerical_solutions"
        ),
    })
    efficiency = efficiency_rows(sample_sets, metadata)
    outputs = {
        "constraint": tables_dir / "constraint_metrics.csv",
        "distribution": tables_dir / "distribution_metrics.csv",
        "efficiency": tables_dir / "efficiency_metrics.csv",
    }
    write_csv(outputs["constraint"], constraints)
    write_csv(outputs["distribution"], distribution)
    write_csv(outputs["efficiency"], efficiency)
    scenario2 = scenario_target_bank is not None
    conservation_definition = (
        "max_j |dx*sum_i u[i,j] - dx*sum_i u[i,0]|"
        if cfg.name == "diffusion"
        else (
            "max_j |M_j-M_0-integral(source + prescribed boundary flux)|"
            if scenario2
            else "max_j |M_j-M_0-integral(source + field-inferred boundary flux)|"
        )
    )
    (evaluation_dir / "constraint_metadata.json").write_text(json.dumps({
        "scenario": manifest.get("constraint_scenario", "conservation_only"),
        "per_sample_reduction": (
            "R_inf is the maximum absolute entry of each raw component residual"
        ),
        "cross_sample_statistics": "mean and sample standard deviation (ddof=1)",
        "conservation_definition": conservation_definition,
        "initial_condition_definition": (
            "max_i |u[i,0]-IC_target[i]|" if scenario2 else None
        ),
        "boundary_condition_definition": (
            "structurally periodic; no independent residual (reported as zero)"
            if scenario2 and cfg.name == "diffusion"
            else (
                "max over t>0 and both faces of the absolute five-point "
                "Neumann-flux observable error on stored cell centres; this "
                "is distinct from the external face-flux variable of the "
                "finite-volume solver"
                if scenario2 else None
            )
        ),
        "combined_normalized_definition": (
            "max(conservation_R_inf, IC_R_inf, BC_R_inf/0.05); Heat omits the structural BC term"
            if scenario2 else None
        ),
        "feasibility_tolerance": 1e-5,
    }, indent=2) + "\n")
    (evaluation_dir / "distribution_metadata.json").write_text(
        json.dumps(distribution_metadata, indent=2) + "\n"
    )
    (evaluation_dir / "manifest.json").write_text(json.dumps({
        "protocol": (
            "Scenario 2 conservation+IC+BC evaluation"
            if manifest.get("constraint_scenario") == "conservation_ic_bc"
            else "Scenario 1 conservation-only evaluation"
        ),
        "metric_categories": [
            "distribution", "efficiency", "constraints", "physical_consistency"
        ],
        "distribution_reference": str(reference_path),
        "distribution_reference_contains_ffm_samples": False,
        "seed": int(seed),
        "methods": list(METHOD_ORDER),
    }, indent=2) + "\n")
    return outputs
