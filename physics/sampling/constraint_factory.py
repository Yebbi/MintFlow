"""Construct Heat and RD1D residual templates from production configs."""
from __future__ import annotations

import math
import os

import torch

from .constraints import Residuals


def _build_residuals(dataset_name: str, config) -> Residuals:
    """Build the canonical residual object for a supported benchmark PDE."""
    if dataset_name == "diffusion":
        return _residuals_heat(config)
    if dataset_name == "rd1d":
        return _residuals_rd(config)
    raise NotImplementedError(
        f"No production residual builder for {dataset_name!r}; "
        "supported datasets are diffusion and rd1d"
    )


def _residuals_heat(config) -> Residuals:
    """Build the endpoint-excluded periodic Heat grid and residual template."""
    nx, nt = (int(value) for value in config.sample_dims)
    t0, t1 = (float(value) for value in config.datasets.t_range)
    x = torch.linspace(0.0, 2.0 * math.pi, nx + 1)[:-1]
    t_grid = torch.linspace(t0, t1, nt)
    dx = torch.tensor(2.0 * math.pi / nx)
    return Residuals(
        x, t_grid, dx=dx, nx=nx, nt=nt,
    )


def _residuals_rd(config) -> Residuals:
    """Build the RD1D residual from its cell-centred HDF5 grid and constants."""
    nx, nt = (int(value) for value in config.sample_dims)
    hdf5 = os.path.join(config.datasets.root, config.datasets.test.data_file)
    try:
        import h5py

        with h5py.File(hdf5, "r") as handle:
            x = torch.tensor(handle["x"][:], dtype=torch.float32)
            t_grid = torch.tensor(handle["t"][:], dtype=torch.float32)
            rho = float(handle.attrs["rho"])
            nu = float(handle.attrs["nu"])
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load RD1D grid metadata from {hdf5!r}: {exc}. "
            "Run the data-generation stage first."
        ) from exc
    return Residuals(
        x, t_grid, dx=x[1] - x[0], nx=nx, nt=nt, rho=rho, nu=nu,
    )
