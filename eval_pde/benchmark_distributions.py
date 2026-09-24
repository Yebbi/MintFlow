"""Numerical reference pools for Scenario 1/2 PDE benchmarks."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .artifacts import file_sha256
from .config import PDEConfig, load_grid
from .numerical_solvers import solve_fisher_kpp_neumann, solve_periodic_heat

GROUND_TRUTH_CACHE_VERSION = 1
PAIRED_REFERENCE_CACHE_VERSION = 1


def _ground_truth_cache_paths(reference_dir: Path) -> tuple[Path, Path]:
    return (
        reference_dir / "numerical_reference_samples.npy",
        reference_dir / "numerical_reference_metadata.json",
    )


def load_or_generate_ground_truth_reference(
    reference_dir: Path,
    cfg: PDEConfig,
    *,
    num_samples: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Create an independent numerical/data-distribution reference pool.

    Heat trajectories are generated from independently sampled training-law
    phase/diffusivity parameters and propagated with the periodic Fourier
    solver. RD1D trajectories are sampled without replacement from the
    production training HDF5, whose fields are outputs of the repository's
    finite-volume Fisher--KPP solver. No FFM output enters either pool.
    """
    if num_samples < 2:
        raise ValueError("Ground-truth reference requires at least two trajectories")
    reference_dir = Path(reference_dir).resolve()
    reference_dir.mkdir(parents=True, exist_ok=True)
    samples_path, metadata_path = _ground_truth_cache_paths(reference_dir)
    if samples_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        samples = np.load(samples_path)
        if (
            metadata.get("cache_version") == GROUND_TRUTH_CACHE_VERSION
            and metadata.get("equation") == cfg.name
            and metadata.get("seed") == int(seed)
            and metadata.get("num_samples") == int(num_samples)
            and tuple(metadata.get("shape", ())) == tuple(samples.shape)
            and samples.shape == (num_samples, cfg.nx, cfg.nt)
        ):
            return samples, metadata

    rng = np.random.default_rng(seed)
    if cfg.name == "diffusion":
        x_grid, t_grid = load_grid(cfg)
        visc_range = tuple(float(value) for value in cfg.ic_source["visc_range"])
        phi_range = tuple(float(value) for value in cfg.ic_source["phi_range"])
        diffusivities = rng.uniform(*visc_range, size=num_samples)
        phases = rng.uniform(*phi_range, size=num_samples)
        samples = np.empty((num_samples, cfg.nx, cfg.nt), dtype=np.float32)
        for index, (diffusivity, phase) in enumerate(zip(diffusivities, phases)):
            initial_condition = np.sin(x_grid + phase)
            samples[index] = solve_periodic_heat(
                initial_condition,
                t_grid,
                float(diffusivity),
                period=cfg.x_max - cfg.x_min,
            ).astype(np.float32)
        source_metadata = {
            "solver": "periodic Fourier semigroup",
            "sampling_law": "independent uniform phase and diffusivity from Heat training ranges",
            "viscosity_range": list(visc_range),
            "phase_range": list(phi_range),
            "diffusivities": diffusivities.tolist(),
            "phases": phases.tolist(),
        }
    elif cfg.name == "rd1d":
        import h5py

        from scripts.training.utils import load_config

        yaml = load_config(cfg.config_path)
        train_path = Path(yaml.datasets.root) / yaml.datasets.train.data_file
        with h5py.File(train_path, "r") as handle:
            fields = handle["u"]
            n_ic, n_bc, nx, nt = fields.shape
            total = n_ic * n_bc
            if num_samples > total:
                raise ValueError(
                    f"Requested {num_samples} RD1D references but training solver bank has {total}"
                )
            flat_indices = rng.choice(total, size=num_samples, replace=False)
            samples = np.empty((num_samples, nx, nt), dtype=np.float32)
            for output_index, flat_index in enumerate(flat_indices):
                ic_index, bc_index = divmod(int(flat_index), n_bc)
                samples[output_index] = np.asarray(fields[ic_index, bc_index], dtype=np.float32)
            rho, nu = float(handle.attrs["rho"]), float(handle.attrs["nu"])
        source_metadata = {
            "solver": "production exact-reaction explicit-midpoint Fisher-KPP finite-volume solver",
            "source_hdf5": str(train_path.resolve()),
            "source_hdf5_sha256": file_sha256(train_path.resolve()),
            "sampling_law": "uniform without replacement over the 80x80 training solver bank",
            "source_total_trajectories": total,
            "selected_flat_indices": flat_indices.tolist(),
            "reaction_rate": rho,
            "diffusivity": nu,
        }
    else:
        raise ValueError(f"Ground-truth reference is not implemented for {cfg.name!r}")

    metadata = {
        "cache_version": GROUND_TRUTH_CACHE_VERSION,
        "equation": cfg.name,
        "seed": int(seed),
        "num_samples": int(num_samples),
        "shape": list(samples.shape),
        "dtype": str(samples.dtype),
        "source": "independent_numerical_ground_truth",
        "contains_ffm_samples": False,
        **source_metadata,
    }
    np.save(samples_path, samples)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return samples, metadata


def _solve_paired_job(job: tuple) -> np.ndarray:
    equation, initial_condition, x, t, parameters = job
    if equation == "diffusion":
        return solve_periodic_heat(
            initial_condition, t, parameters[0], period=parameters[1]
        ).astype(np.float32)
    return solve_fisher_kpp_neumann(
        initial_condition, x, t,
        reaction_rate=parameters[0], diffusivity=parameters[1],
        left_flux=parameters[2], right_flux=parameters[3],
    ).astype(np.float32)


def load_or_generate_paired_scenario_reference(
    reference_dir: Path,
    cfg: PDEConfig,
    bank,
    *,
    workers: int = 1,
) -> tuple[np.ndarray, dict]:
    """Solve one numerical trajectory for every Scenario-2 IC/BC target."""
    reference_dir = Path(reference_dir).resolve()
    reference_dir.mkdir(parents=True, exist_ok=True)
    samples_path, metadata_path = _ground_truth_cache_paths(reference_dir)
    if samples_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        samples = np.load(samples_path)
        if (
            metadata.get("cache_version") == PAIRED_REFERENCE_CACHE_VERSION
            and metadata.get("source") == "paired_scenario2_numerical_solutions"
            and metadata.get("target_id") == bank.target_id
            and tuple(metadata.get("shape", ())) == tuple(samples.shape)
            and samples.shape == (bank.num_targets, cfg.nx, cfg.nt)
        ):
            if not np.allclose(
                samples[:, :, 0], bank.initial_conditions, rtol=0.0, atol=1e-6
            ):
                raise RuntimeError(
                    "Cached paired numerical reference does not preserve target ICs"
                )
            if cfg.name == "rd1d":
                metadata.update({
                    "boundary_condition": "prescribed constant Neumann face flux",
                    "global_balance_flux_source": "prescribed external target-bank flux",
                    "initial_boundary_compatibility": (
                        "BC applies for t>0; the generated t=0 IC is preserved exactly"
                    ),
                })
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            return samples, metadata

    x, t = load_grid(cfg)
    jobs = []
    for index in range(bank.num_targets):
        if cfg.name == "diffusion":
            parameters = (
                float(bank.physical_parameters[index, 0]),
                float(cfg.x_max - cfg.x_min),
            )
        else:
            parameters = tuple(
                float(value) for value in bank.physical_parameters[index]
            )
        jobs.append(
            (cfg.name, bank.initial_conditions[index], x, t, parameters)
        )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            solved = list(pool.map(_solve_paired_job, jobs, chunksize=8))
    else:
        solved = [_solve_paired_job(job) for job in jobs]
    samples = np.stack(solved)
    if not np.allclose(
        samples[:, :, 0], bank.initial_conditions, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("Paired numerical reference did not preserve target ICs")
    metadata = {
        "cache_version": PAIRED_REFERENCE_CACHE_VERSION,
        "equation": cfg.name,
        "num_samples": bank.num_targets,
        "shape": list(samples.shape),
        "dtype": str(samples.dtype),
        "source": "paired_scenario2_numerical_solutions",
        "contains_ffm_samples": False,
        "target_id": bank.target_id,
        "target_pairing": "reference i solves target-bank IC/BC i",
        "solver": (
            "periodic Fourier semigroup" if cfg.name == "diffusion"
            else "exact-reaction explicit-midpoint Fisher-KPP solver"
        ),
    }
    if cfg.name == "rd1d":
        metadata.update({
            "boundary_condition": "prescribed constant Neumann face flux",
            "global_balance_flux_source": "prescribed external target-bank flux",
            "initial_boundary_compatibility": (
                "BC applies for t>0; the generated t=0 IC is preserved exactly"
            ),
        })
    np.save(samples_path, samples)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return samples, metadata
