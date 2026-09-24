"""Paired model-derived targets for the IC+BC constraint scenario.

Scenario 2 uses a bank of unconstrained Vanilla-FFM trajectories.  Target
``i`` contains the exact ``t=0`` column of Vanilla sample ``i`` and the
physical boundary specification inferred for that same sample.  Constrained
sample ``i`` is always evaluated against target ``i``; targets are never
broadcast or cycled.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval_pde.numerical_solvers import (
    estimate_heat_diffusivity,
    estimate_rd_boundary_fluxes,
)
from sampling.constraint_factory import _build_residuals

SCHEMA_VERSION = 2


def _array_hash(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@dataclass(frozen=True)
class ScenarioTargetBank:
    """One target specification for every paired Vanilla-FFM sample."""

    initial_conditions: np.ndarray
    boundary_values: np.ndarray
    physical_parameters: np.ndarray
    x_grid: np.ndarray
    t_grid: np.ndarray
    metadata: dict[str, Any]

    @property
    def target_id(self) -> str:
        return str(self.metadata["target_id"])

    @property
    def num_targets(self) -> int:
        return int(len(self.initial_conditions))

    def validate_slice(self, offset: int, count: int) -> None:
        if offset < 0 or count < 1 or offset + count > self.num_targets:
            raise ValueError(
                f"Scenario target slice [{offset}:{offset + count}] exceeds "
                f"bank size {self.num_targets}"
            )


def derive_scenario_target_bank(
    dataset_name: str,
    samples: np.ndarray,
    config: Any,
    *,
    seed: int,
    checkpoint_path: str,
    config_path: str,
    n_eval: int,
    source_samples_path: str,
) -> ScenarioTargetBank:
    """Derive a paired target bank from a complete Vanilla-FFM sample set."""
    samples = np.asarray(samples, dtype=np.float32)
    expected = tuple(int(value) for value in config.sample_dims)
    if samples.ndim != 3 or tuple(samples.shape[1:]) != expected:
        raise ValueError(
            f"Scenario source has shape {samples.shape}, expected (N, {expected})"
        )
    if len(samples) < 1 or not np.all(np.isfinite(samples)):
        raise ValueError("Scenario source must contain finite Vanilla samples")

    template = _build_residuals(dataset_name, config)
    x_grid = template.x.detach().cpu().numpy().astype(np.float64)
    t_grid = template.t_grid.detach().cpu().numpy().astype(np.float64)
    initial_conditions = np.ascontiguousarray(samples[:, :, 0])

    if dataset_name == "diffusion":
        bounds = tuple(float(value) for value in config.datasets.visc_range)
        diffusivities = np.asarray(
            [estimate_heat_diffusivity(sample, t_grid, bounds) for sample in samples],
            dtype=np.float64,
        )
        boundary_values = np.empty((len(samples), 0), dtype=np.float32)
        physical_parameters = diffusivities[:, None]
        physical = {
            "pde": "u_t = diffusivity * u_xx",
            "boundary_type": "periodic",
            "boundary_enforcement": "endpoint-excluded periodic discrete topology",
            "boundary_target_dimension_per_sample": 0,
            "parameter_columns": ["diffusivity"],
            "diffusivity_source": "fit separately to each complete Vanilla trajectory",
            "diffusivity_bounds": list(bounds),
        }
    elif dataset_name == "rd1d":
        dx = float(template.dx)
        nu = float(template.nu)
        left_bounds, right_bounds = (0.0, 0.05), (-0.05, 0.0)
        boundary_values = np.asarray(
            [
                estimate_rd_boundary_fluxes(
                    sample, dx, nu, left_bounds, right_bounds
                )
                for sample in samples
            ],
            dtype=np.float32,
        )
        physical_parameters = np.column_stack(
            [
                np.full(len(samples), float(template.rho)),
                np.full(len(samples), nu),
                boundary_values[:, 0],
                boundary_values[:, 1],
            ]
        ).astype(np.float64)
        physical = {
            "pde": "u_t = diffusivity * u_xx + reaction_rate * u * (1-u)",
            "boundary_type": "constant_neumann_flux",
            "boundary_sign_convention": "g = -diffusivity * du/dx at each face",
            "boundary_source": "fit separately to each complete Vanilla trajectory",
            "boundary_bounds": [list(left_bounds), list(right_bounds)],
            "boundary_target_dimension_per_sample": 2,
            "boundary_trace_enforcement": (
                "five-point observable at t[1:] (t=0 compatibility is not imposed)"
            ),
            "global_balance_flux_source": "prescribed target-bank flux",
            "parameter_columns": [
                "reaction_rate", "diffusivity", "left_flux", "right_flux"
            ],
        }
    else:
        raise ValueError(
            "Scenario target banks are implemented only for diffusion and rd1d"
        )

    source_hash = _array_hash(samples)
    ic_hash = _array_hash(initial_conditions)
    boundary_hash = _array_hash(boundary_values)
    physical_hash = _array_hash(physical_parameters)
    x_hash = _array_hash(x_grid)
    t_hash = _array_hash(t_grid)
    target_digest = hashlib.sha256(
        (source_hash + ic_hash + boundary_hash).encode("ascii")
    ).hexdigest()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "source": "unconstrained_vanilla_ffm",
        "source_seed": int(seed),
        "source_n_eval": int(n_eval),
        "source_num_samples": int(len(samples)),
        "source_samples_path": str(Path(source_samples_path).resolve()),
        "source_samples_sha256": source_hash,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "sample_shape": list(samples.shape[1:]),
        "initial_conditions_shape": list(initial_conditions.shape),
        "initial_conditions_sha256": ic_hash,
        "boundary_values_shape": list(boundary_values.shape),
        "boundary_values_sha256": boundary_hash,
        "physical_parameters_shape": list(physical_parameters.shape),
        "physical_parameters_sha256": physical_hash,
        "x_grid_sha256": x_hash,
        "t_grid_sha256": t_hash,
        "target_id": f"vanilla_ffm_bank_{target_digest[:16]}",
        "pairing": "sample_id i uses Vanilla sample i IC and BC",
        "ic_extraction": "samples[:, :, 0] copied exactly",
        "conservation_constraint": (
            "constant_mass" if dataset_name == "diffusion"
            else "nonlinear_global_balance"
        ),
        "physical_specification": physical,
    }
    return ScenarioTargetBank(
        initial_conditions=initial_conditions,
        boundary_values=np.ascontiguousarray(boundary_values),
        physical_parameters=np.ascontiguousarray(physical_parameters),
        x_grid=x_grid,
        t_grid=t_grid,
        metadata=metadata,
    )


def save_scenario_target_bank(
    path: str | Path, bank: ScenarioTargetBank
) -> Path:
    """Persist numeric targets and auditable provenance."""
    path = Path(path).resolve()
    if path.suffix != ".npz":
        raise ValueError("Scenario target bank path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        initial_conditions=np.asarray(bank.initial_conditions, dtype=np.float32),
        boundary_values=np.asarray(bank.boundary_values, dtype=np.float32),
        physical_parameters=np.asarray(bank.physical_parameters, dtype=np.float64),
        x_grid=np.asarray(bank.x_grid, dtype=np.float64),
        t_grid=np.asarray(bank.t_grid, dtype=np.float64),
    )
    path.with_suffix(".json").write_text(
        json.dumps(bank.metadata, indent=2) + "\n"
    )
    return path


def load_scenario_target_bank(
    path: str | Path, dataset_name: str | None = None
) -> ScenarioTargetBank:
    """Load and fully validate a paired Scenario-2 target bank."""
    path = Path(path).resolve()
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Scenario target bank requires {path} and {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    with np.load(path, allow_pickle=False) as arrays:
        bank = ScenarioTargetBank(
            initial_conditions=np.asarray(
                arrays["initial_conditions"], dtype=np.float32
            ),
            boundary_values=np.asarray(arrays["boundary_values"], dtype=np.float32),
            physical_parameters=np.asarray(
                arrays["physical_parameters"], dtype=np.float64
            ),
            x_grid=np.asarray(arrays["x_grid"], dtype=np.float64),
            t_grid=np.asarray(arrays["t_grid"], dtype=np.float64),
            metadata=metadata,
        )
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Scenario target schema: {metadata.get('schema_version')}"
        )
    if dataset_name is not None and metadata.get("dataset") != dataset_name:
        raise ValueError(
            f"Scenario target dataset is {metadata.get('dataset')!r}, "
            f"expected {dataset_name!r}"
        )
    n = int(metadata.get("source_num_samples", -1))
    nx = int(metadata.get("sample_shape", [0])[0])
    expected_boundary = 0 if metadata.get("dataset") == "diffusion" else 2
    expected_parameters = 1 if metadata.get("dataset") == "diffusion" else 4
    if bank.initial_conditions.shape != (n, nx):
        raise ValueError("Scenario target IC bank has an invalid shape")
    if bank.boundary_values.shape != (n, expected_boundary):
        raise ValueError("Scenario target BC bank has an invalid shape")
    if bank.physical_parameters.shape != (n, expected_parameters):
        raise ValueError("Scenario target physical-parameter bank has an invalid shape")
    if bank.x_grid.shape != (nx,) or bank.t_grid.shape != (
        int(metadata.get("sample_shape", [0, 0])[1]),
    ):
        raise ValueError("Scenario target coordinate grids have invalid shapes")
    if _array_hash(bank.initial_conditions) != metadata.get(
        "initial_conditions_sha256"
    ):
        raise ValueError("Scenario target IC hash does not match metadata")
    if _array_hash(bank.boundary_values) != metadata.get("boundary_values_sha256"):
        raise ValueError("Scenario target BC hash does not match metadata")
    for name, value in (
        ("physical_parameters", bank.physical_parameters),
        ("x_grid", bank.x_grid),
        ("t_grid", bank.t_grid),
    ):
        if _array_hash(value) != metadata.get(f"{name}_sha256"):
            raise ValueError(f"Scenario target {name} hash does not match metadata")
    if not all(
        np.all(np.isfinite(value))
        for value in (
            bank.initial_conditions,
            bank.boundary_values,
            bank.physical_parameters,
        )
    ):
        raise ValueError("Scenario target bank contains non-finite values")
    return bank
