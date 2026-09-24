"""Local PDE residuals and same-IC numerical-simulation errors."""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .config import PDEConfig, load_grid
from .numerical_solvers import (
    estimate_heat_diffusivity,
    estimate_rd_boundary_fluxes,
    solve_fisher_kpp_neumann,
    solve_periodic_heat,
)


def _heat_residual(sample: np.ndarray, t: np.ndarray, dx: float, alpha: float) -> np.ndarray:
    values = np.asarray(sample, dtype=np.float64)
    temporal = np.gradient(values, t, axis=1, edge_order=2)
    laplacian = (np.roll(values, -1, axis=0) - 2.0 * values + np.roll(values, 1, axis=0)) / dx**2
    return temporal - alpha * laplacian


def _rd_residual(sample: np.ndarray, t: np.ndarray, dx: float, rho: float, nu: float) -> np.ndarray:
    values = np.asarray(sample, dtype=np.float64)
    temporal = np.gradient(values, t, axis=1, edge_order=2)
    laplacian = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / dx**2
    interior = values[1:-1]
    return temporal[1:-1] - nu * laplacian - rho * interior * (1.0 - interior)


def solve_same_ic_reference(
    equation: str,
    sample: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    parameters: tuple[float, ...],
) -> np.ndarray:
    """Solve the PDE from this generated sample's own initial-time column.

    ``parameters`` contains the paired physical problem definition: Heat uses
    ``(diffusivity, period)`` and RD1D uses ``(rho, nu, left_flux,
    right_flux)``.  The initial condition is never taken from another method.
    """
    sample = np.asarray(sample, dtype=np.float64)
    initial_condition = sample[:, 0].copy()
    if equation == "diffusion":
        simulation = solve_periodic_heat(
            initial_condition, t, parameters[0], period=float(parameters[1])
        )
    elif equation == "rd1d":
        simulation = solve_fisher_kpp_neumann(
            initial_condition,
            x,
            t,
            reaction_rate=parameters[0],
            diffusivity=parameters[1],
            left_flux=parameters[2],
            right_flux=parameters[3],
        )
    else:
        raise ValueError(f"Unsupported equation {equation!r}")
    if not np.allclose(simulation[:, 0], initial_condition, rtol=0.0, atol=1e-12):
        raise RuntimeError("Numerical solver did not preserve the method-specific IC")
    return simulation


def _solve_and_measure(job: tuple) -> float:
    equation, sample, x, t, parameters = job
    sample = np.asarray(sample, dtype=np.float64)
    if not np.all(np.isfinite(sample)):
        return float("nan")
    simulation = solve_same_ic_reference(equation, sample, x, t, parameters)
    difference = sample - simulation
    return float(
        np.linalg.norm(difference.reshape(-1))
        / max(np.linalg.norm(simulation.reshape(-1)), np.finfo(np.float64).tiny)
    )


def evaluate_physical_consistency(
    sample_sets: Mapping[str, np.ndarray],
    vanilla_samples: np.ndarray,
    cfg: PDEConfig,
    *,
    workers: int = 1,
    scenario_target_bank=None,
) -> tuple[list[dict], list[dict], dict]:
    """Evaluate raw local residuals and same-IC solver errors for every sample.

    Latent parameters absent from FFM outputs are estimated once from each
    aligned Vanilla trajectory and then shared across methods with that sample
    index. Thus methods differ only in their generated field/IC, not in the
    PDE parameter selected for an identical initial-noise index.
    """
    x, t = load_grid(cfg)
    dx = float(x[1] - x[0])
    vanilla = np.asarray(vanilla_samples)
    counts = {name: len(samples) for name, samples in sample_sets.items()}
    if set(counts.values()) != {len(vanilla)}:
        raise ValueError(f"All methods must align with Vanilla: {counts}, vanilla={len(vanilla)}")

    if scenario_target_bank is not None:
        if scenario_target_bank.num_targets != len(vanilla):
            raise ValueError("Scenario target bank and evaluated sample count differ")
        if cfg.name == "diffusion":
            alphas = scenario_target_bank.physical_parameters[:, 0]
            parameter_rows = [
                (float(alpha), cfg.x_max - cfg.x_min) for alpha in alphas
            ]
            parameter_metadata = {
                "heat_diffusivity_source": "paired Scenario-2 target bank",
                "heat_diffusivities": alphas.tolist(),
            }
        elif cfg.name == "rd1d":
            parameter_rows = [
                tuple(float(value) for value in row)
                for row in scenario_target_bank.physical_parameters
            ]
            parameter_metadata = {
                "rd_coefficients": {
                    "reaction_rate": cfg.rd_rho, "diffusivity": cfg.rd_nu,
                },
                "rd_boundary_flux_source": "paired Scenario-2 target bank",
                "boundary_fluxes": scenario_target_bank.boundary_values.tolist(),
            }
        else:
            raise ValueError(f"Unsupported equation {cfg.name!r}")
        pairing_source = "paired Scenario-2 target bank"
    elif cfg.name == "diffusion":
        bounds = tuple(float(value) for value in cfg.ic_source["visc_range"])
        alphas = np.asarray(
            [estimate_heat_diffusivity(sample, t, bounds) for sample in vanilla],
            dtype=np.float64,
        )
        parameter_rows = [(float(alpha), cfg.x_max - cfg.x_min) for alpha in alphas]
        parameter_metadata = {
            "heat_diffusivity_source": "estimated once from each complete aligned Vanilla FFM trajectory",
            "heat_diffusivity_bounds": list(bounds),
            "heat_diffusivities": alphas.tolist(),
        }
        pairing_source = "aligned Vanilla trajectory"
    elif cfg.name == "rd1d":
        left_bounds, right_bounds = (0.0, 0.05), (-0.05, 0.0)
        fluxes = np.asarray([
            estimate_rd_boundary_fluxes(sample, dx, cfg.rd_nu, left_bounds, right_bounds)
            for sample in vanilla
        ], dtype=np.float64)
        parameter_rows = [
            (float(cfg.rd_rho), float(cfg.rd_nu), float(left), float(right))
            for left, right in fluxes
        ]
        parameter_metadata = {
            "rd_coefficients": {"reaction_rate": cfg.rd_rho, "diffusivity": cfg.rd_nu},
            "rd_boundary_flux_source": "estimated once from each complete aligned Vanilla FFM trajectory",
            "left_flux_bounds": list(left_bounds),
            "right_flux_bounds": list(right_bounds),
            "boundary_fluxes": fluxes.tolist(),
        }
        pairing_source = "aligned Vanilla trajectory"
    else:
        raise ValueError(f"Unsupported equation {cfg.name!r}")

    per_sample: list[dict] = []
    for method, raw_samples in sample_sets.items():
        samples = np.asarray(raw_samples)
        jobs = [
            (cfg.name, samples[index], x, t, parameter_rows[index])
            for index in range(len(samples))
        ]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                simulation_metrics = list(pool.map(_solve_and_measure, jobs, chunksize=8))
        else:
            simulation_metrics = [_solve_and_measure(job) for job in jobs]
        for index, sample in enumerate(samples):
            if np.all(np.isfinite(sample)):
                residual = (
                    _heat_residual(sample, t, dx, parameter_rows[index][0])
                    if cfg.name == "diffusion" else
                    _rd_residual(sample, t, dx, float(cfg.rd_rho), float(cfg.rd_nu))
                )
                residual_rms = float(np.sqrt(np.mean(residual**2)))
            else:
                residual_rms = float("nan")
            per_sample.append({
                "method": method,
                "sample_id": index,
                "pde_residual_rms": residual_rms,
                "simulation_relative_l2": simulation_metrics[index],
            })

    summary: list[dict] = []
    metrics = ("pde_residual_rms", "simulation_relative_l2")
    for method in sample_sets:
        rows = [row for row in per_sample if row["method"] == method]
        summary_row: dict[str, object] = {"method": method, "n": len(rows)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary_row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            summary_row[f"{metric}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
            summary_row[f"{metric}_median"] = float(np.median(finite)) if len(finite) else float("nan")
            summary_row[f"{metric}_n_finite"] = int(len(finite))
        summary.append(summary_row)
    metadata = {
        "equation": cfg.name,
        "local_residual_discretization": (
            "second-order periodic spatial difference and numpy edge_order=2 temporal derivative"
            if cfg.name == "diffusion" else
            "second-order interior spatial difference and numpy edge_order=2 temporal derivative"
        ),
        "initial_condition_pairing": "each method uses its own generated sample[:, 0]",
        "boundary_condition_pairing": (
            "periodic for every method"
            if cfg.name == "diffusion"
            else (
                "paired target-bank left/right Neumann fluxes shared across "
                "methods for each sample ID"
                if scenario_target_bank is not None else
                "aligned Vanilla-inferred left/right Neumann fluxes shared "
                "across methods for each sample ID"
            )
        ),
        "simulation_parameter_pairing": (
            f"latent parameters taken from {pairing_source} and shared across methods"
        ),
        **parameter_metadata,
    }
    return per_sample, summary, metadata
