"""Differentiable conservation residuals for the production PDE tasks."""
from __future__ import annotations

import torch


class Residuals:
    """Grid-aware Heat mass and RD1D global-balance operators."""

    def __init__(
        self,
        x: torch.Tensor,
        t_grid: torch.Tensor,
        *,
        dx: torch.Tensor,
        nx: int,
        nt: int,
        rho: float | None = None,
        nu: float | None = None,
    ) -> None:
        self.x = x
        self.t_grid = t_grid
        self.dx = dx
        self.nx = nx
        self.nt = nt
        self.rho = rho
        self.nu = nu

    def mass_residual_heat(self, u_flat: torch.Tensor) -> torch.Tensor:
        """Return ``M(t_j)-M(t_0)`` for ``j=1,...,nt-1``."""
        u = u_flat.view(self.nx, self.nt)
        dx = self.dx.to(u.device, u.dtype)
        mass = torch.sum(u, dim=0) * dx
        return mass[1:] - mass[0]

    def mass_residual_rd(self, u_flat: torch.Tensor) -> torch.Tensor:
        """Return the cumulative Fisher--KPP source/flux balance residual."""
        if self.rho is None or self.nu is None:
            raise RuntimeError("RD1D residual requires reaction rate and diffusivity")
        sol = u_flat.view(self.nx, self.nt).T
        dx = self.dx.to(sol.device, sol.dtype)
        dt = (self.t_grid[1:] - self.t_grid[:-1]).to(sol.device, sol.dtype)

        mass = sol.sum(dim=1) * dx
        source = self.rho * (sol * (1.0 - sol)).sum(dim=1) * dx
        source_mid = 0.5 * (source[:-1] + source[1:])
        source_cumulative = torch.cat([
            torch.zeros(1, device=sol.device, dtype=sol.dtype),
            torch.cumsum(source_mid * dt, dim=0),
        ])

        left_flux = -self.nu * (
            -25.0 * sol[:, 0] + 48.0 * sol[:, 1] - 36.0 * sol[:, 2]
            + 16.0 * sol[:, 3] - 3.0 * sol[:, 4]
        ) / (12.0 * dx)
        right_flux = -self.nu * (
            25.0 * sol[:, -1] - 48.0 * sol[:, -2] + 36.0 * sol[:, -3]
            - 16.0 * sol[:, -4] + 3.0 * sol[:, -5]
        ) / (12.0 * dx)
        net_flux_mid = 0.5 * (
            (left_flux - right_flux)[:-1] + (left_flux - right_flux)[1:]
        )
        flux_cumulative = torch.cat([
            torch.zeros(1, device=sol.device, dtype=sol.dtype),
            torch.cumsum(net_flux_mid * dt, dim=0),
        ])
        return mass - (mass[0] + source_cumulative + flux_cumulative)

    def mass_residual_rd_jacobian(
        self, u_flat: torch.Tensor, *, include_boundary_flux: bool = True,
    ) -> torch.Tensor:
        """Exact Jacobian of the nontrivial RD1D balance rows.

        Returns ``d mass_residual_rd(u)[1:] / d u`` with shape
        ``(nt - 1, nx * nt)`` in the repository's flattened ``(x, t)`` order.
        The nonlinear Fisher--KPP source derivative is evaluated at ``u``;
        mass and boundary-flux contributions are exact constant stencils.
        """
        if self.rho is None or self.nu is None:
            raise RuntimeError("RD1D residual requires reaction rate and diffusivity")
        u = u_flat.view(self.nx, self.nt)
        dx = self.dx.to(u.device, u.dtype)
        dt = (self.t_grid[1:] - self.t_grid[:-1]).to(u.device, u.dtype)
        m = self.nt - 1

        # W[j, q] is the trapezoidal integration weight of a time-local
        # quantity at t_q in the cumulative integral ending at t_{j+1}.
        included_intervals = torch.tril(
            torch.ones((m, m), device=u.device, dtype=u.dtype)
        )
        interval_weights = 0.5 * included_intervals * dt.unsqueeze(0)
        weights = torch.zeros((m, self.nt), device=u.device, dtype=u.dtype)
        weights[:, :-1].add_(interval_weights)
        weights[:, 1:].add_(interval_weights)

        jacobian = torch.zeros(
            (m, self.nx, self.nt), device=u.device, dtype=u.dtype
        )
        rows = torch.arange(m, device=u.device)
        jacobian[:, :, 0] = -dx
        jacobian[rows, :, rows + 1] += dx

        source_gradient = self.rho * dx * (1.0 - 2.0 * u)
        jacobian.sub_(weights[:, None, :] * source_gradient[None, :, :])

        if include_boundary_flux:
            flux_gradient = torch.zeros(self.nx, device=u.device, dtype=u.dtype)
            left_coefficients = torch.tensor(
                [-25.0, 48.0, -36.0, 16.0, -3.0], device=u.device, dtype=u.dtype
            )
            right_coefficients = torch.tensor(
                [3.0, -16.0, 36.0, -48.0, 25.0], device=u.device, dtype=u.dtype
            )
            flux_gradient[:5] += (-self.nu / (12.0 * dx)) * left_coefficients
            # net_flux = left_flux - right_flux
            flux_gradient[-5:] -= (-self.nu / (12.0 * dx)) * right_coefficients
            jacobian.sub_(weights[:, None, :] * flux_gradient[None, :, None])
        return jacobian.reshape(m, self.nx * self.nt)
