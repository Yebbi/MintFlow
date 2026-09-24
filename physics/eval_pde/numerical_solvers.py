"""Numerical PDE reconstructions used in qualitative benchmark figures.

The pretrained generators do not retain the latent Heat diffusivity or RD1D
Neumann fluxes.  The helpers below therefore estimate those parameters from
the complete Vanilla FFM trajectory, while taking the numerical solver's
initial state exclusively from its ``t=0`` column.
"""
from __future__ import annotations

import numpy as np


def estimate_heat_diffusivity(
    reference: np.ndarray,
    t_grid: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    """Fit the single-mode Heat decay rate and constrain it to training bounds."""
    reference = np.asarray(reference, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != len(t_grid):
        raise ValueError("reference must have shape (nx, len(t_grid))")
    if reference.shape[0] < 3 or len(t_grid) < 2:
        raise ValueError("Heat diffusivity fitting requires at least 3 x points and 2 times")

    amplitude = np.abs(np.fft.rfft(reference, axis=0)[1])
    floor = np.finfo(np.float64).eps * max(float(amplitude[0]), 1.0)
    valid = (t_grid > t_grid[0]) & np.isfinite(amplitude) & (amplitude > floor)
    elapsed = t_grid[valid] - t_grid[0]
    if not np.any(valid) or amplitude[0] <= floor:
        estimate = 0.5 * (bounds[0] + bounds[1])
    else:
        log_decay = -np.log(amplitude[valid] / amplitude[0])
        estimate = float(np.dot(elapsed, log_decay) / np.dot(elapsed, elapsed))
    if not np.isfinite(estimate):
        estimate = 0.5 * (bounds[0] + bounds[1])
    return float(np.clip(estimate, *bounds))


def solve_periodic_heat(
    initial_condition: np.ndarray,
    t_grid: np.ndarray,
    diffusivity: float,
    period: float = 2.0 * np.pi,
) -> np.ndarray:
    """Solve ``u_t = diffusivity*u_xx`` with a stable Fourier propagator."""
    initial_condition = np.asarray(initial_condition, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    if initial_condition.ndim != 1 or len(initial_condition) < 2:
        raise ValueError("initial_condition must be a one-dimensional spatial field")
    if np.any(np.diff(t_grid) < 0) or diffusivity <= 0 or period <= 0:
        raise ValueError("t_grid must be ordered and diffusivity/period must be positive")

    dx = period / len(initial_condition)
    wave_number = 2.0 * np.pi * np.fft.rfftfreq(len(initial_condition), d=dx)
    coefficients = np.fft.rfft(initial_condition)
    elapsed = t_grid - t_grid[0]
    propagated = coefficients[:, None] * np.exp(
        -diffusivity * wave_number[:, None] ** 2 * elapsed[None, :]
    )
    return np.fft.irfft(propagated, n=len(initial_condition), axis=0)


def estimate_rd_boundary_fluxes(
    reference: np.ndarray,
    dx: float,
    diffusivity: float,
    left_bounds: tuple[float, float],
    right_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Estimate constant Neumann fluxes using the benchmark's five-point stencil."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 5:
        raise ValueError("reference must have shape (nx>=5, nt)")
    values = reference.T
    left = -diffusivity * (
        -25 * values[:, 0] + 48 * values[:, 1] - 36 * values[:, 2]
        + 16 * values[:, 3] - 3 * values[:, 4]
    ) / (12 * dx)
    right = -diffusivity * (
        25 * values[:, -1] - 48 * values[:, -2] + 36 * values[:, -3]
        - 16 * values[:, -4] + 3 * values[:, -5]
    ) / (12 * dx)
    finite_left, finite_right = left[np.isfinite(left)], right[np.isfinite(right)]
    if not len(finite_left) or not len(finite_right):
        raise ValueError("reference does not contain finite boundary-flux estimates")
    left_flux = float(np.clip(np.median(finite_left), *left_bounds))
    right_flux = float(np.clip(np.median(finite_right), *right_bounds))
    return left_flux, right_flux


def solve_fisher_kpp_neumann(
    initial_condition: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    reaction_rate: float,
    diffusivity: float,
    left_flux: float,
    right_flux: float,
    cfl: float = 0.25,
) -> np.ndarray:
    """Solve the repository's cell-centred Fisher--KPP system.

    This is the production data generator's exact-reaction/explicit-midpoint
    scheme generalized to an arbitrary increasing output-time grid.  Its
    internal step obeys the same conservative diffusion CFL restriction.
    """
    initial_condition = np.asarray(initial_condition, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    if initial_condition.ndim != 1 or initial_condition.shape != x_grid.shape:
        raise ValueError("initial_condition and x_grid must be matching 1-D arrays")
    if len(x_grid) < 2 or len(t_grid) < 1 or np.any(np.diff(t_grid) <= 0):
        raise ValueError("spatial and strictly increasing temporal grids are required")
    spacing = np.diff(x_grid)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("the RD1D solver requires a uniform cell-centred grid")
    if diffusivity <= 0 or reaction_rate < 0 or not 0 < cfl <= 1:
        raise ValueError("invalid PDE coefficient or CFL value")

    dx = float(spacing[0])
    stable_step = cfl * 0.5 * dx**2 / (diffusivity + 1e-8)
    solution = np.empty((len(initial_condition), len(t_grid)), dtype=np.float64)
    solution[:, 0] = initial_condition
    state = initial_condition.copy()

    def flux(values: np.ndarray) -> np.ndarray:
        padded = np.zeros(len(values) + 4, dtype=np.float64)
        padded[2:2 + len(values)] = values
        result = -diffusivity * (
            padded[2:len(values) + 3] - padded[1:len(values) + 2]
        ) / dx
        result[0], result[-1] = left_flux, right_flux
        return result

    def reaction(values: np.ndarray, step: float) -> np.ndarray:
        decay = np.exp(-reaction_rate * step)
        denominator = values + (1.0 - values) * decay
        return values / denominator

    def update(base: np.ndarray, flux_state: np.ndarray, step: float) -> np.ndarray:
        face_flux = flux(flux_state)
        return reaction(base, step) - step * np.diff(face_flux) / dx

    current_time = float(t_grid[0])
    for output_index, target_time in enumerate(t_grid[1:], start=1):
        while current_time < target_time - 1e-14:
            step = min(stable_step, float(target_time - current_time))
            midpoint = update(state, state, 0.5 * step)
            state = update(state, midpoint, step)
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("RD1D numerical simulation became non-finite")
            current_time += step
        solution[:, output_index] = state
    return solution
