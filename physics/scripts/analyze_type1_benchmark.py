#!/usr/bin/env python3
"""Build streamlined Scenario 1/2 reports and publication figure sets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  Registers the SciencePlots styles.

from eval_pde.artifacts import read_csv
from eval_pde.config import build_pde_config, load_grid
from eval_pde.physical_consistency import solve_same_ic_reference
from eval_pde.type1_evaluation import (
    DISPLAY_NAMES,
    METHOD_ORDER,
    QUALITATIVE_METHODS,
    load_benchmark_runs,
)


def _float(value: str | float) -> float:
    return float(value)


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def _load_metric_tables(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the four canonical metric tables in the fixed method order."""
    tables = root / "tables"
    names = {
        "distribution": "distribution_metrics.csv",
        "efficiency": "efficiency_metrics.csv",
        "constraints": "constraint_metrics.csv",
        "physical_consistency": "physical_consistency_summary.csv",
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for category, filename in names.items():
        indexed = _index(read_csv(tables / filename))
        missing = [method for method in METHOD_ORDER if method not in indexed]
        if missing:
            raise ValueError(f"{filename} is missing methods: {missing}")
        result[category] = [indexed[method] for method in METHOD_ORDER]
    return result


def _solver_parameters(root: Path, equation: str) -> dict[str, Any]:
    path = root / "evaluation" / "physical_consistency_metadata.json"
    metadata = json.loads(path.read_text())
    if metadata.get("equation") != equation:
        raise ValueError(f"Physical-consistency metadata does not match {equation}")
    return metadata


def _simulation(
    sample: np.ndarray,
    sample_id: int,
    cfg,
    parameter_metadata: dict[str, Any],
) -> np.ndarray:
    x, t = load_grid(cfg)
    if cfg.name == "diffusion":
        parameters = (
            float(parameter_metadata["heat_diffusivities"][sample_id]),
            float(cfg.x_max - cfg.x_min),
        )
    else:
        left, right = parameter_metadata["boundary_fluxes"][sample_id]
        parameters = (
            float(cfg.rd_rho), float(cfg.rd_nu), float(left), float(right)
        )
    return solve_same_ic_reference(cfg.name, sample, x, t, parameters)


def _solution_limits(
    fields: list[np.ndarray], equation: str
) -> tuple[float, float]:
    low = float(min(np.min(field) for field in fields))
    high = float(max(np.max(field) for field in fields))
    if equation == "diffusion":
        # Heat fields fluctuate around zero, so equal-magnitude values should
        # receive equal visual weight on either side of the diverging map.
        bound = max(abs(low), abs(high), np.finfo(float).eps)
        return -bound, bound
    if equation != "rd1d":
        raise ValueError(f"Unsupported equation for plot limits: {equation!r}")
    # RD1D trajectories are predominantly positive and contain only sparse,
    # small negative excursions. A symmetric range therefore compresses
    # almost all variation into the positive half of PiYG. Preserve the true
    # row extrema so both low and high RD1D structure remain visible.
    if high <= low:
        high = low + np.finfo(float).eps
    return low, high


def _error_limits(fields: list[np.ndarray]) -> tuple[float, float]:
    """Return a nonnegative scale for absolute-error fields."""
    high = float(max(np.max(field) for field in fields))
    return 0.0, max(high, np.finfo(float).eps)


def _strip_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")


def _atomic_write_text(path: Path, value: str) -> None:
    """Commit text only after its complete contents reach the filesystem."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    """Render and atomically publish a complete PNG on parallel filesystems."""
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.png")
    try:
        fig.savefig(
            temporary, format="png", dpi=dpi, bbox_inches="tight",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _plot_qualitative_rows(
    row_fields: list[list[np.ndarray]],
    titles: list[str],
    row_cmaps: list[str],
    row_limits: list[tuple[float, float]],
    *,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Plot six methods with one right-side colorbar for each data row."""
    num_rows = len(row_fields)
    if not (
        num_rows == len(row_cmaps) == len(row_limits)
        and all(len(fields) == len(titles) for fields in row_fields)
    ):
        raise ValueError("Qualitative row specifications have inconsistent lengths")

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(
        num_rows,
        len(titles) + 1,
        width_ratios=[1.0] * len(titles) + [0.045],
        wspace=0.04,
        hspace=0.04,
    )
    for row, (fields, cmap, (vmin, vmax)) in enumerate(
        zip(row_fields, row_cmaps, row_limits)
    ):
        row_image = None
        for column, (title, field) in enumerate(zip(titles, fields)):
            axis = fig.add_subplot(grid[row, column])
            if row == 0:
                axis.set_title(title, fontsize=12)
            row_image = axis.imshow(
                field,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            _strip_axis(axis)
        color_axis = fig.add_subplot(grid[row, -1])
        colorbar = fig.colorbar(row_image, cax=color_axis, orientation="vertical")
        colorbar.ax.tick_params(labelsize=8)
    return fig


def _generate_qualitative_figures(
    root: Path,
    cfg,
    sample_sets: dict[str, np.ndarray],
    manifest: dict[str, Any],
    *,
    figure_seed: int,
    num_figures: int = 7,
) -> dict[str, Any]:
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    n = len(sample_sets["vanilla_ffm"])
    if num_figures > n:
        raise ValueError(f"Requested {num_figures} figures from only {n} samples")
    rng = np.random.default_rng(figure_seed)
    sample_ids = sorted(int(value) for value in rng.choice(n, num_figures, replace=False))
    parameters = _solver_parameters(root, cfg.name)
    cmap = "RdBu_r" if cfg.name == "diffusion" else "PiYG"
    error_cmap = "magma"
    plt.style.use(["science", "no-latex"])

    for figure_index, sample_id in enumerate(sample_ids):
        generated = [
            np.asarray(sample_sets[method][sample_id], dtype=np.float64)
            for method, _ in QUALITATIVE_METHODS
        ]
        simulations = [
            _simulation(field, sample_id, cfg, parameters) for field in generated
        ]
        differences = [
            np.abs(field - simulation)
            for field, simulation in zip(generated, simulations)
        ]
        titles = [title for _, title in QUALITATIVE_METHODS]
        fig = _plot_qualitative_rows(
            [generated, differences],
            titles,
            [cmap, error_cmap],
            [_solution_limits(generated, cfg.name), _error_limits(differences)],
            figsize=(18.0, 5.8),
        )
        _atomic_save_figure(
            fig,
            figure_dir / f"sample_comparison_{figure_index:03d}.png",
            dpi=300,
        )
        plt.close(fig)

        fig = _plot_qualitative_rows(
            [generated, simulations, differences],
            titles,
            [cmap, cmap, error_cmap],
            [
                _solution_limits(generated, cfg.name),
                _solution_limits(simulations, cfg.name),
                _error_limits(differences),
            ],
            figsize=(18.0, 8.4),
        )
        _atomic_save_figure(
            fig,
            figure_dir / f"physical_consistency_{figure_index:03d}.png",
            dpi=300,
        )
        plt.close(fig)

    parameter_records: dict[str, Any]
    if cfg.name == "diffusion":
        parameter_records = {
            str(sample_id): {"diffusivity": parameters["heat_diffusivities"][sample_id]}
            for sample_id in sample_ids
        }
    else:
        parameter_records = {
            str(sample_id): {
                "left_flux": parameters["boundary_fluxes"][sample_id][0],
                "right_flux": parameters["boundary_fluxes"][sample_id][1],
                "reaction_rate": cfg.rd_rho,
                "diffusivity": cfg.rd_nu,
            }
            for sample_id in sample_ids
        }
    result = {
        "figure_seed": int(figure_seed),
        "sample_ids": sample_ids,
        "shared_initial_noise": manifest["shared_noise"],
        "methods": [
            {"configuration": method, "title": title} for method, title in QUALITATIVE_METHODS
        ],
        "simulation_protocol": (
            "For every column, the numerical trajectory starts from that method's own "
            "generated sample[:, 0]. Heat uses the same periodic boundary condition in "
            "every column. RD1D uses the aligned sample ID's Vanilla-inferred left/right "
            "Neumann fluxes, held fixed across methods so all columns represent the same "
            "paired physical problem."
        ),
        "solver_parameters": parameter_records,
        "sample_comparison_layout": (
            "2x6: generated field; absolute error to the method-specific "
            "same-IC numerical simulation"
        ),
        "physical_consistency_layout": (
            "3x6: generated field; method-specific same-IC numerical "
            "simulation; absolute error"
        ),
        "color_scale": {
            "solution_rows": (
                f"{cmap}, "
                + (
                    "zero-centred symmetric scale fitted per row"
                    if cfg.name == "diffusion"
                    else "true data-minimum/data-maximum scale fitted per row"
                )
            ),
            "absolute_error_rows": (
                f"{error_cmap}, nonnegative scale from zero to the row maximum"
            ),
            "colorbars": "one vertical colorbar per row at the right edge",
        },
        "sample_comparison_pattern": "sample_comparison_000.png .. sample_comparison_006.png",
        "physical_consistency_pattern": "physical_consistency_000.png .. physical_consistency_006.png",
    }
    _atomic_write_text(
        figure_dir / "qualitative_manifest.json", json.dumps(result, indent=2) + "\n"
    )
    return result


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _distribution_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Sampling method | FID | KID | MMD | W2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {_fmt(_float(row['FID']))} | "
            f"{_fmt(_float(row['KID']))} | {_fmt(_float(row['MMD']))} | "
            f"{_fmt(_float(row['W2']))} |"
        )
    return "\n".join(lines)


def _efficiency_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Sampling method | Time/sample (s) | Peak GPU memory (GiB) |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {_fmt(_float(row['time_per_sample_s']))} | "
            f"{_fmt(_float(row['peak_gpu_memory_GiB']))} |"
        )
    return "\n".join(lines)


def _constraint_table(
    rows: list[dict[str, Any]], *, scenario: str, equation: str,
) -> str:
    conservation = "Mass" if equation == "diffusion" else "Global balance"
    if scenario == "conservation_only":
        lines = [
            f"| Sampling method | {conservation} $R_\\infty$ mean | "
            f"{conservation} $R_\\infty$ std | Maximum $R_\\infty$ | "
            "Pass rate ($\\leq 10^{-5}$) |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['display_name']} | "
                f"{_fmt(_float(row['conservation_R_inf_mean']))} | "
                f"{_fmt(_float(row['conservation_R_inf_std']))} | "
                f"{_fmt(_float(row['conservation_R_inf_max']))} | "
                f"{100.0 * _float(row['conservation_pass_rate_at_1e-5']):.1f}% |"
            )
        return "\n".join(lines)

    lines = [
        f"| Sampling method | {conservation} $R_\\infty$ mean | "
        f"{conservation} $R_\\infty$ std | {conservation} pass | "
        "IC $R_\\infty$ mean | IC $R_\\infty$ std | IC pass | "
        "BC $R_\\infty$ mean | BC $R_\\infty$ std | BC pass | "
        "Combined normalized $R_\\infty$ mean | Combined normalized "
        "$R_\\infty$ std | Combined pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | "
            f"{_fmt(_float(row['conservation_R_inf_mean']))} | "
            f"{_fmt(_float(row['conservation_R_inf_std']))} | "
            f"{100.0 * _float(row['conservation_pass_rate_at_1e-5']):.1f}% | "
            f"{_fmt(_float(row['IC_R_inf_mean']))} | "
            f"{_fmt(_float(row['IC_R_inf_std']))} | "
            f"{100.0 * _float(row['IC_pass_rate_at_1e-5']):.1f}% | "
            f"{_fmt(_float(row['BC_R_inf_mean']))} | "
            f"{_fmt(_float(row['BC_R_inf_std']))} | "
            f"{100.0 * _float(row['BC_pass_rate_at_1e-5']):.1f}% | "
            f"{_fmt(_float(row['combined_normalized_R_inf_mean']))} | "
            f"{_fmt(_float(row['combined_normalized_R_inf_std']))} | "
            f"{100.0 * _float(row['combined_pass_rate_at_1e-5']):.1f}% |"
        )
    return "\n".join(lines)


def _configuration_table(manifest: dict[str, Any]) -> str:
    config = manifest["mintflow_production_config"]
    values = (
        ("Samples per method", manifest["num_samples"]),
        ("Numerical reference trajectories", manifest["reference_samples"]),
        ("Distribution embedding", "frozen POD fitted once on numerical reference"),
        ("Distribution-metric precision", "float64"),
        ("Discrete-Euler steps", config["steps"]),
        ("Candidate count $K$", config["num_candidates"]),
        (
            "Candidate interval",
            f"[{config['candidate_t_min']}, {config['candidate_t_max']}]",
        ),
        ("Candidate spacing", config["time_sampling"]),
        ("End-bias power", config["end_bias_power"]),
        ("Horizon regularization $\\lambda_{\\mathrm{reg}}$", config["time_penalty"]),
        ("Correction scale $\\gamma$", config["correction_scale"]),
        ("Linear solver", config["correction_mode"]),
        ("Terminal projection", "enabled" if config["final_projection"] else "disabled"),
        ("Projection tolerance", config["projection_tolerance"]),
        ("Evaluation pass threshold", config["evaluation_pass_threshold"]),
    )
    lines = ["| Parameter | Value |", "|---|---:|"]
    lines.extend(f"| {name} | `{value}` |" for name, value in values)
    return "\n".join(lines)


def _findings(tables: dict[str, list[dict[str, Any]]]) -> str:
    distribution = _index(tables["distribution"])
    constraints = _index(tables["constraints"])
    physical = _index(tables["physical_consistency"])
    mintflow = "mintflow_pseudoinverse_with_final_projection"
    lines = []
    for baseline in ("diffusionpde", "dflow", "vanilla_ffm"):
        mf = distribution[mintflow]
        other = distribution[baseline]
        comparisons = ", ".join(
            f"{metric} {_fmt(_float(mf[metric]))} vs. {_fmt(_float(other[metric]))}"
            for metric in ("FID", "KID", "MMD", "W2")
        )
        lines.append(f"- Against {DISPLAY_NAMES[baseline]}: {comparisons}.")
    constraint_row = constraints[mintflow]
    pass_key = (
        "combined_pass_rate_at_1e-5"
        if "combined_pass_rate_at_1e-5" in constraint_row
        else "conservation_pass_rate_at_1e-5"
    )
    physical_row = physical[mintflow]
    lines.append(
        f"- MintFlow passes the $10^{{-5}}$ constraint threshold on "
        f"{100.0 * _float(constraint_row[pass_key]):.1f}% of samples; its mean PDE "
        f"residual is {_fmt(_float(physical_row['pde_residual_rms_mean']))} and mean "
        f"same-IC simulation error is "
        f"{_fmt(_float(physical_row['simulation_relative_l2_mean']))}."
    )
    return "\n".join(lines)


def _physical_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Sampling method | PDE RMS mean +/- std | PDE RMS median | Simulation rel-L2 mean +/- std | Simulation rel-L2 median |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {DISPLAY_NAMES[row['method']]} | "
            f"{_fmt(_float(row['pde_residual_rms_mean']))} +/- "
            f"{_fmt(_float(row['pde_residual_rms_std']))} | "
            f"{_fmt(_float(row['pde_residual_rms_median']))} | "
            f"{_fmt(_float(row['simulation_relative_l2_mean']))} +/- "
            f"{_fmt(_float(row['simulation_relative_l2_std']))} | "
            f"{_fmt(_float(row['simulation_relative_l2_median']))} |"
        )
    return "\n".join(lines)


def _write_report(
    root: Path,
    cfg,
    manifest: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    qualitative: dict[str, Any],
) -> None:
    equation = "Heat constant-mass" if cfg.name == "diffusion" else "RD1D global-balance"
    scenario = manifest.get("constraint_scenario", "conservation_only")
    scenario_title = (
        "Scenario 1 (conservation only)"
        if scenario == "conservation_only"
        else "Scenario 2 (conservation + IC + BC)"
    )
    rd_boundary_note = (
        "\nFor RD1D Scenario 2, the BC column is the configured five-point "
        "boundary-flux observable on the stored cell-centred field. The "
        "finite-volume solver applies Neumann fluxes externally at cell faces, "
        "so this observable is a sampling constraint, not an exact reconstruction "
        "of the solver's face-flux variable.\n"
        if cfg.name == "rd1d" and scenario == "conservation_ic_bc" else ""
    )
    report = f"""# {equation}: {scenario_title}

This benchmark evaluates `{manifest['num_samples']}` aligned samples per sampling method. All methods use seed `{manifest['seed']}` and the same persisted initial-noise bank. Distribution metrics use the `{manifest['reference_samples']}`-trajectory numerical reference pool recorded by the benchmark manifest; Vanilla FFM is an evaluated method, not a distribution reference.

## Production MintFlow configuration

{_configuration_table(manifest)}

## Quantitative results

### Table 1: Distribution

{_distribution_table(tables['distribution'])}

### Table 2: Efficiency

{_efficiency_table(tables['efficiency'])}

### Table 3: Constraints

{_constraint_table(tables['constraints'], scenario=scenario, equation=cfg.name)}

### Table 4: Physical consistency

{_physical_table(tables['physical_consistency'])}

## Results summary

{_findings(tables)}

The four retained evaluation categories are:

1. **Distribution:** FID, KID, MMD, and exact equal-weight empirical W2 in one quadrature-weighted, mean-centered POD space fitted on the numerical reference pool.
2. **Efficiency:** synchronized generation wall time per sample and peak allocated CUDA memory.
3. **Constraints:** each reported component `R_inf` is the maximum absolute entry of that component's raw-space residual vector for one sample; the table reports its mean and sample standard deviation across all samples. In Scenario 1 this is the mass/global-balance residual alone. Scenario 2 additionally reports IC and BC residuals. Its combined feasibility statistic is normalized exactly as in sampling (RD1D BC flux errors are divided by `0.05`; Heat periodicity is structural and has zero independent BC residual).
4. **Physical consistency:** finite-difference PDE residual and error to a numerical trajectory initialized from each generated sample's own `t=0` state.
{rd_boundary_note}

PDE parameters that are not identifiable from the initial state alone are inferred once from the aligned Vanilla FFM trajectory and held fixed across methods for that sample ID. Heat uses periodic boundaries. RD1D uses the corresponding inferred left/right Neumann fluxes.

## Qualitative results

The figure seed is `{qualitative['figure_seed']}` and the fixed shared-noise sample IDs are `{qualitative['sample_ids']}`.

- `figures/sample_comparison_000.png` through `sample_comparison_006.png`: 2 x 6 generated-sample and absolute-difference comparisons.
- `figures/physical_consistency_000.png` through `physical_consistency_006.png`: 3 x 6 generated, numerical-simulation, and absolute-difference comparisons.

Every numerical panel extracts its IC from the generated field directly above it. Heat uses periodic BCs. For RD1D, the paired sample ID's inferred Neumann fluxes are fixed across methods so the six method-specific ICs are compared under the same BCs. Complete figure provenance is stored in `figures/qualitative_manifest.json`.

## Output files

- `tables/distribution_metrics.csv`
- `tables/efficiency_metrics.csv`
- `tables/constraint_metrics.csv`
- `tables/physical_consistency_summary.csv`
- `evaluation/physical_consistency_per_sample.csv`
- `evaluation/frozen_pod_embedding.npz`
- `evaluation/distribution_metadata.json`
"""
    _atomic_write_text(root / "REPORT.md", report)


def main(args: argparse.Namespace) -> None:
    root = Path(args.benchmark_root).resolve()
    cfg = build_pde_config(args.dataset, task=args.task, config_path=args.config)
    manifest, sample_sets, _ = load_benchmark_runs(root)
    tables = _load_metric_tables(root)
    qualitative = _generate_qualitative_figures(
        root, cfg, sample_sets, manifest,
        figure_seed=args.figure_seed,
        num_figures=7,
    )
    _write_report(root, cfg, manifest, tables, qualitative)
    print(f"Wrote streamlined report and 14 figures to {root}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--dataset", required=True, choices=("diffusion", "rd1d"))
    p.add_argument("--task", required=True, choices=("heat_mass_conservation", "global_balance"))
    p.add_argument("--config", required=True)
    p.add_argument("--figure-seed", type=int, default=20260902)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
