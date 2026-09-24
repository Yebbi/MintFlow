"""
eval_pde/config.py
====================
``PDEConfig`` dataclass and per-dataset builders.

Builders fill fields from the Heat/RD1D repository configs and,
where applicable, from the HDF5 test files under ``datasets/data/`` — never
from invented defaults.  See the reconnaissance report (§3) for the full
per-dataset filling rationale, summarized in each builder's docstring below.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

_CONFIG_MAP = {
    "diffusion": "configs/heat.yml",
    "rd1d": "configs/rd1d.yml",
}


@dataclass
class PDEConfig:
    name: str
    task: str
    nx: int
    nt: int
    ny: Optional[int] = None
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    t_min: float = 0.0
    t_max: float = 1.0
    periodic_x: bool = False
    periodic_y: bool = False

    # RD1D's historical names are retained: rho is the Fisher--KPP reaction
    # rate and nu is its diffusivity.
    heat_alpha: Optional[float] = None    # diffusion: None — per-sample, not persisted (see below)
    rd_rho: Optional[float] = None        # rd1d reaction rate
    rd_nu: Optional[float] = None         # rd1d diffusion coefficient

    ic_source: Optional[object] = None
    bc_source: Optional[object] = None
    left_bc_key: Optional[str] = None

    rd_operator: str = "config_required"
    spectrum_type: str = "auto"

    grid_source: str = "closed_form"      # "closed_form" | "hdf5"
    hdf5_path: Optional[str] = None
    config_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_pde_config(dataset_name: str, task: str = "ic", config_path: Optional[str] = None) -> PDEConfig:
    """Dispatch to the per-dataset :class:`PDEConfig` builder."""
    if dataset_name == "diffusion":
        return build_diffusion_config(task=task, config_path=config_path)
    if dataset_name == "rd1d":
        return build_rd1d_config(task=task, config_path=config_path)
    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Supported: diffusion, rd1d."
    )


def build_diffusion_config(task: str = "ic", config_path: Optional[str] = None) -> PDEConfig:
    """Build the diffusion/heat :class:`PDEConfig`.

    Grid: periodic ``x in [0, 2*pi)`` (``nx`` points), ``t in t_range``
    (``nt`` points), from ``configs/heat.yml``.

    ``heat_alpha`` is intentionally left ``None``: samples are generated
    on-the-fly (``datasets/diffusion.py::DiffusionDataset``) as
    ``u = sin(x+phi) * exp(-t*v)`` with ``v, phi ~ random.uniform(...)`` drawn
    *per sample* and never persisted to disk (see reconnaissance report §5,
    blocker #1). ``ic_source`` instead carries the generative-process
    description needed to *statistically* re-derive a seeded reference set
    (see ``eval_pde.io.load_ground_truth_reference``); it is NOT a source of
    per-sample ground truth for already-generated outputs.
    """
    config_path = config_path or _CONFIG_MAP["diffusion"]
    cfg_yaml = _load_yaml(config_path)

    nx, nt = int(cfg_yaml.sample_dims[0]), int(cfg_yaml.sample_dims[1])
    visc_range = tuple(cfg_yaml.datasets.get("visc_range", [1.0, 5.0]))
    phi_range = tuple(cfg_yaml.datasets.get("phi_range", [0.0, math.pi]))
    t_range = tuple(cfg_yaml.datasets.get("t_range", [0.0, 1.0]))

    return PDEConfig(
        name="diffusion",
        task=task,
        nx=nx,
        nt=nt,
        x_min=0.0,
        x_max=2.0 * math.pi,
        t_min=float(t_range[0]),
        t_max=float(t_range[1]),
        periodic_x=True,
        heat_alpha=None,
        ic_source={
            "type": "on_the_fly",
            "visc_range": visc_range,
            "phi_range": phi_range,
            "t_range": t_range,
            "n_data_test": int(cfg_yaml.datasets.get("test", {}).get("n_data", 1000)),
        },
        rd_operator="n/a",
        grid_source="closed_form",
        config_path=config_path,
    )


def build_rd1d_config(task: str = "ic", config_path: Optional[str] = None) -> PDEConfig:
    """Build the rd1d :class:`PDEConfig`.

    Grid + PDE constants are loaded directly from the HDF5 test file
    (``rho``/``nu`` root attrs), mirroring
    ``sampling/constraint_factory.py::_residuals_rd`` exactly (the YAML
    config does not carry these — they only exist in the data file).
    """
    import h5py

    config_path = config_path or _CONFIG_MAP["rd1d"]
    cfg_yaml = _load_yaml(config_path)

    root = cfg_yaml.datasets.root
    test_file = cfg_yaml.datasets.test.data_file
    hdf5_path = os.path.join(root, test_file)

    with h5py.File(hdf5_path, "r") as f:
        x = f["x"][:]
        t_grid = f["t"][:]
        rho = float(f.attrs["rho"])
        nu = float(f.attrs["nu"])
        nx = int(f["u"].shape[2])
        nt = int(f["u"].shape[3])

    return PDEConfig(
        name="rd1d",
        task=task,
        nx=nx,
        nt=nt,
        x_min=float(x[0]),
        x_max=float(x[-1]),
        t_min=float(t_grid[0]),
        t_max=float(t_grid[-1]),
        periodic_x=False,
        rd_rho=rho,
        rd_nu=nu,
        ic_source={"type": "hdf5", "key": "ic", "path": hdf5_path},
        bc_source={"type": "hdf5", "key": "bc", "path": hdf5_path},
        rd_operator="fisher_kpp_neumann",
        grid_source="hdf5",
        hdf5_path=hdf5_path,
        config_path=config_path,
    )


def load_grid(cfg: PDEConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, t_grid)`` 1-D arrays for *cfg*, using the same source
    convention as ``sampling/constraint_factory.py``'s per-dataset
    ``Residuals`` builders (HDF5 for ``rd1d``, closed-form
    ``linspace`` for ``diffusion``).
    """
    if cfg.grid_source == "hdf5":
        import h5py
        with h5py.File(cfg.hdf5_path, "r") as f:
            x = np.asarray(f["x"][:], dtype=np.float64)
            t_grid = np.asarray(f["t"][:], dtype=np.float64)
        return x, t_grid
    if cfg.periodic_x:
        x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, endpoint=False)
    else:
        x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    t_grid = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)
    return x, t_grid


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_yaml(config_path: str):
    from scripts.training.utils import load_config
    return load_config(config_path)
