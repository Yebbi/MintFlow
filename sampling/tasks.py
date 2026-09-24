"""
sampling/tasks.py
===================
:class:`ConstraintTask` — a configuration-driven abstraction describing one
constraint in a form reused by PCFM, ECI, MintFlow, DiffusionPDE, D-Flow, final
projection, and evaluation. Production Heat/RD1D tasks support two explicit
scenarios: conservation only, or conservation plus a paired target IC/BC.

Task inventory
---------------
* ``diffusion`` / ``heat_mass_conservation`` — affine constant mass.
* ``rd1d`` / ``global_balance`` — nonlinear reaction/source/flux balance.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import numpy as np
import torch

from sampling.constraint_factory import _build_residuals
from sampling.constraints import Residuals
from sampling.projectors import (
    ProjectionConfig,
    ProjectionResult,
    damped_gauss_newton_projection,
    feasibility_check,
    linear_minimum_norm_projection,
)

__all__ = [
    "CONSTRAINT_SCENARIOS",
    "ConstraintTask",
    "ConstraintTaskProvider",
    "build_task",
    "list_tasks",
]

CONSTRAINT_SCENARIOS = ("conservation_only", "conservation_ic_bc")

_NEW_TASKS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "diffusion": ("heat_mass_conservation",),
    "rd1d": ("global_balance",),
}


# ---------------------------------------------------------------------------
# ConstraintTask
# ---------------------------------------------------------------------------

@dataclass
class ConstraintTask:
    """A configuration-driven description of one constraint task.

    Fields map directly to the request:

    * ``name``, ``dataset`` — identifiers.
    * ``output_dim`` — dimension ``m`` of the residual vector.
    * ``enforcement_residual`` — ``H(U) -> (m,)`` torch, differentiable,
      *target already baked in* (matches the existing ``hfunc`` convention
      used by PCFM/MintFlow — see ``sampling/methods.py::_make_hfunc``). Takes a
      flattened ``(nx*nt,)`` tensor.
    * ``eval_residual`` — batched NumPy evaluation-side counterpart,
      ``(N, nx, nt) -> {component_name: (N, k)}``.
    * ``target`` — the target vector (or ``None`` for self-referential
      tasks like ``global_balance``, where the residual is already
      zero-targeted).
    * ``metadata`` — grid/physical metadata (``dx``, ``dt``, ``nx``, ``nt``,
      ``x``, ``t_grid``, ``rho``, ``nu``, ...) needed to reproduce the
      residual definitions elsewhere.
    * ``scales`` — ``(m,)`` normalization scale per residual component.
    * ``component_names`` / ``component_dims`` / ``units`` /
      ``display_name`` — human-readable bookkeeping.
    * ``differentiable`` / ``differentiability_notes`` — whether
      ``enforcement_residual`` is autograd-traceable end-to-end (required
      for PCFM Newton / MintFlow adjoint), and any caveats (e.g. TV smoothing).
    * ``native_projector`` — optional ``Callable[[Tensor, ProjectionConfig],
      ProjectionResult]`` — a closed-form projector specialized to this
      task's structure (linear or affine), or ``None`` if only the generic
      damped Gauss-Newton projector applies.
    * ``feasibility_checker`` — optional override of the default
      ``feasibility_check`` behavior.
    """

    name: str
    dataset: str
    output_dim: int
    enforcement_residual: Callable[[torch.Tensor], torch.Tensor]
    eval_residual: Optional[Callable[..., dict[str, np.ndarray]]]
    target: Optional[torch.Tensor]
    metadata: dict[str, Any]
    scales: torch.Tensor
    component_names: list[str]
    enforcement_jacobian: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    units: list[str] = field(default_factory=list)
    component_dims: list[int] = field(default_factory=list)
    differentiable: bool = True
    differentiability_notes: str = ""
    native_projector: Optional[Callable[[torch.Tensor, Optional[ProjectionConfig]], ProjectionResult]] = None
    feasibility_checker: Optional[Callable[[torch.Tensor, float], dict[str, Any]]] = None
    display_name: str = ""

    def normalized_residual(self, u_flat: torch.Tensor) -> torch.Tensor:
        """``enforcement_residual(u) / scales`` — for fair cross-component comparison."""
        res = self.enforcement_residual(u_flat)
        if res.ndim == 0:
            res = res.unsqueeze(0)
        return res / self.scales.to(res.device, res.dtype).clamp_min(1e-300)

    def normalized_jacobian(self, u_flat: torch.Tensor) -> torch.Tensor:
        """Exact task Jacobian with the same row scaling as the residual.

        Tasks expose this only when a closed-form Jacobian is available.  It
        avoids a full ``jacrev`` over the constraint vector in MintFlow while
        preserving the identical residual map and correction mathematics.
        """
        if self.enforcement_jacobian is None:
            raise RuntimeError(f"Task {self.name!r} has no analytic Jacobian")
        jacobian = self.enforcement_jacobian(u_flat)
        if jacobian.ndim == 1:
            jacobian = jacobian.unsqueeze(0)
        scales = self.scales.to(jacobian.device, jacobian.dtype).clamp_min(1e-300)
        return jacobian / scales[:, None]

    def check_feasibility(self, u_flat: torch.Tensor, tol: float = 1e-3) -> dict[str, Any]:
        """Feasibility of *u_flat* under this task's residual (uses
        ``feasibility_checker`` if provided, else the shared default)."""
        if self.feasibility_checker is not None:
            return self.feasibility_checker(u_flat, tol)
        with torch.no_grad():
            residual = self.enforcement_residual(u_flat)
        return feasibility_check(residual, tol=tol, scales=self.scales)

    @property
    def device(self) -> torch.device:
        """Device this task's closures/tensors (``A``, ``target``, ``scales``)
        were built on — see :func:`build_task`'s ``device=`` argument."""
        return self.scales.device

    def project(self, xi_flat: torch.Tensor, config: Optional[ProjectionConfig] = None) -> ProjectionResult:
        """Project *xi_flat* onto this task's constraint manifold.

        Uses ``native_projector`` when available (exact for linear tasks,
        closed-form for affine ones); otherwise falls back to the generic
        :func:`~sampling.projectors.damped_gauss_newton_projection` applied
        to ``enforcement_residual``.

        *xi_flat* may live on any device/dtype (e.g. a vanilla-FFM sample
        that was moved to CPU by ``generate_samples_batched`` while this
        task's closures were built on ``cuda``, as happens for the
        ``final_projection`` baseline) — it is moved to ``self.device`` /
        ``self.scales.dtype`` before projecting, and the result is moved
        back so callers see the same device/dtype they passed in.
        """
        config = config or ProjectionConfig(scales=self.scales)
        orig_device, orig_dtype = xi_flat.device, xi_flat.dtype
        xi_flat = xi_flat.to(device=self.device, dtype=self.scales.dtype)
        if self.native_projector is not None:
            result = self.native_projector(xi_flat, config)
        else:
            # Normalize inside the residual passed to the numerical solve,
            # not only in its stopping diagnostic. This gives PCFM, MintFlow,
            # ECI, guided baselines, and final projection the same relative
            # weighting for heterogeneous Scenario-2 components.
            result = damped_gauss_newton_projection(
                xi_flat, self.normalized_residual, replace(config, scales=None),
            )
        if result.u.device != orig_device or result.u.dtype != orig_dtype:
            result.u = result.u.to(device=orig_device, dtype=orig_dtype)
        return result


@dataclass
class ConstraintTaskProvider:
    """Lazily build one paired constraint task per generated sample.

    Scenario-2 Heat tasks contain a large affine projection matrix and RD1D
    tasks contain sample-specific IC/BC closures.  Keeping all 1,000 tasks on
    a GPU simultaneously is wasteful, so samplers request only the current
    sample or batch through :meth:`task_for`.
    """

    num_tasks: int
    task_builder: Callable[[int], ConstraintTask]
    metadata: dict[str, Any] = field(default_factory=dict)
    _prototype: ConstraintTask = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_tasks < 1:
            raise ValueError("ConstraintTaskProvider requires at least one task")
        self._prototype = self.task_builder(0)

    def task_for(self, index: int) -> ConstraintTask:
        if not 0 <= index < self.num_tasks:
            raise IndexError(
                f"Paired task index {index} is outside [0, {self.num_tasks})"
            )
        return self._prototype if index == 0 else self.task_builder(index)

    @property
    def name(self) -> str:
        return self._prototype.name

    @property
    def dataset(self) -> str:
        return self._prototype.dataset

    @property
    def output_dim(self) -> int:
        return self._prototype.output_dim

    @property
    def scales(self) -> torch.Tensor:
        return self._prototype.scales

    @property
    def component_names(self) -> list[str]:
        return self._prototype.component_names

    @property
    def component_dims(self) -> list[int]:
        return self._prototype.component_dims

    @property
    def units(self) -> list[str]:
        return self._prototype.units

    @property
    def display_name(self) -> str:
        return self._prototype.display_name

    @property
    def device(self) -> torch.device:
        return self._prototype.device

    @property
    def native_projector(self):
        return self._prototype.native_projector

    @property
    def target(self):
        # There is no single target vector for a paired bank.
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _heat_mass_matrix(nx: int, nt: int, dx: float, device, dtype) -> torch.Tensor:
    """Exact matrix for ``M(t_j)-M(t_0)``, avoiding a large autograd Jacobian."""
    matrix = torch.zeros(nt - 1, nx, nt, device=device, dtype=dtype)
    matrix[:, :, 0] = -dx
    time_rows = torch.arange(nt - 1, device=device)
    matrix[time_rows, :, time_rows + 1] = dx
    return matrix.reshape(nt - 1, nx * nt)


def list_tasks(dataset_name: Optional[str] = None) -> list[str]:
    """List registered new global-task names, optionally filtered by dataset."""
    if dataset_name is not None:
        return list(_NEW_TASKS_BY_DATASET.get(dataset_name, ()))
    out: list[str] = []
    for names in _NEW_TASKS_BY_DATASET.values():
        out.extend(names)
    return out


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_task(
    dataset_name: str,
    task_name: str,
    config: Any = None,
    config_path: Optional[str] = None,
    reference_sample: Optional[torch.Tensor] = None,
    device: str = "cpu",
    target_override: Optional[torch.Tensor] = None,
    scenario: str = "conservation_only",
    boundary_target: Optional[torch.Tensor] = None,
) -> ConstraintTask:
    """Build a concrete :class:`ConstraintTask` for one (dataset, task) pair.

    Parameters
    ----------
    dataset_name     : Dataset owning the requested registered task.
    task_name        : One of :func:`list_tasks` for *dataset_name*.
    config           : Pre-loaded EasyDict config (loaded from
                        ``config_path``/default if ``None``).
    config_path      : Explicit YAML path (used when *config* is ``None``).
    reference_sample : ``(nx, nt)`` reference field used to derive the task's
                        target values. Required for every task except
                        ``"global_balance"``, *unless* ``target_override`` is
                        given (in which case it is optional and, if omitted,
                        the task's target is taken entirely from
                        ``target_override`` — see below).
    device           : Device for the returned tensors/closures.
    target_override  : ``(output_dim,)`` explicit target vector. When given,
                        this *replaces* the target that would otherwise be
                        derived from ``reference_sample`` (e.g. a target
                        selected from a saved Scenario-2 target bank).
                        A standalone task contains one target. Scenario 2
                        wraps these tasks in :class:`ConstraintTaskProvider`
                        so sample ``i`` receives target ``i`` from the paired
                        Vanilla-FFM target bank.
    scenario         : ``"conservation_only"`` preserves the current focused
                        benchmarks. ``"conservation_ic_bc"`` augments the
                        conservation law with a fixed reference IC and the
                        dataset's physical BC representation. A single task
                        holds one target; :class:`ConstraintTaskProvider`
                        supplies one such task per generated sample.
    boundary_target  : RD1D prescribed Neumann flux pair ``(gL, gR)`` or
                        time series ``(2, nt)``. Required for exact RD1D BC
                        targeting; when omitted it is reconstructed from the
                        reference field with the benchmark boundary stencil.
    """
    if scenario not in CONSTRAINT_SCENARIOS:
        raise ValueError(
            f"Unknown constraint scenario {scenario!r}; choose from {CONSTRAINT_SCENARIOS}."
        )
    if dataset_name not in _NEW_TASKS_BY_DATASET:
        raise ValueError(
            f"No new global tasks registered for dataset '{dataset_name}'. "
            f"Supported: {list(_NEW_TASKS_BY_DATASET)}"
        )
    if task_name not in _NEW_TASKS_BY_DATASET[dataset_name]:
        raise ValueError(
            f"Unknown task '{task_name}' for dataset '{dataset_name}'. "
            f"Available: {_NEW_TASKS_BY_DATASET[dataset_name]}"
        )

    if config is None:
        from scripts.training.utils import load_config
        from visualization.datasets import default_config_for_dataset
        config_path = config_path or default_config_for_dataset(dataset_name)
        config = load_config(config_path)

    template = _build_residuals(dataset_name, config)
    template = Residuals(
        x=template.x.to(device), t_grid=template.t_grid.to(device),
        dx=template.dx.to(device),
        nx=template.nx, nt=template.nt, rho=template.rho, nu=template.nu,
    )

    if reference_sample is not None:
        reference_sample = reference_sample.to(device=device, dtype=torch.float32).view(template.nx, template.nt)
    elif task_name not in {"global_balance", "heat_mass_conservation"} and target_override is None:
        raise ValueError(
            f"Task '{task_name}' requires a reference_sample or a target_override."
        )

    if target_override is not None:
        target_override = torch.as_tensor(target_override, dtype=torch.float32, device=device).reshape(-1)

    builder = _TASK_BUILDERS[(dataset_name, task_name)]
    task = builder(template, reference_sample, device, target_override=target_override)
    if scenario == "conservation_only":
        task.metadata = {**task.metadata, "constraint_scenario": scenario}
        return task
    if reference_sample is None:
        raise ValueError(
            f"Scenario {scenario!r} requires one fixed reference_sample containing the target IC."
        )
    if dataset_name == "diffusion":
        return _augment_heat_with_ic_bc(task, template, reference_sample, device)
    if dataset_name == "rd1d":
        return _augment_rd_with_ic_bc(
            task, template, reference_sample, device, boundary_target=boundary_target,
        )
    raise ValueError(
        f"Scenario {scenario!r} is implemented only for diffusion and rd1d, got {dataset_name!r}."
    )


# ---------------------------------------------------------------------------
# Production scenario composition
# ---------------------------------------------------------------------------

def _rd_boundary_fluxes_torch(u: torch.Tensor, dx: float, nu: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the benchmark's five-point one-sided Neumann flux estimates.

    ``u`` has shape ``(nx, nt)`` and the returned tensors each have shape
    ``(nt,)``.  This is deliberately the same stencil used by
    :meth:`sampling.constraints.Residuals.mass_residual_rd`, so the BC and
    global-balance components cannot silently disagree about sign convention.
    """
    g_left = -nu * (
        -25.0 * u[0] + 48.0 * u[1] - 36.0 * u[2] + 16.0 * u[3] - 3.0 * u[4]
    ) / (12.0 * dx)
    g_right = -nu * (
        25.0 * u[-1] - 48.0 * u[-2] + 36.0 * u[-3] - 16.0 * u[-4] + 3.0 * u[-5]
    ) / (12.0 * dx)
    return g_left, g_right


def _rd_boundary_fluxes_numpy(samples: np.ndarray, dx: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized NumPy counterpart of :func:`_rd_boundary_fluxes_torch`."""
    values = np.asarray(samples, dtype=np.float64)
    left = -nu * (
        -25.0 * values[:, 0] + 48.0 * values[:, 1] - 36.0 * values[:, 2]
        + 16.0 * values[:, 3] - 3.0 * values[:, 4]
    ) / (12.0 * dx)
    right = -nu * (
        25.0 * values[:, -1] - 48.0 * values[:, -2] + 36.0 * values[:, -3]
        - 16.0 * values[:, -4] + 3.0 * values[:, -5]
    ) / (12.0 * dx)
    return left, right


def _augment_heat_with_ic_bc(
    conservation: ConstraintTask,
    template: Residuals,
    reference_sample: torch.Tensor,
    device: str,
) -> ConstraintTask:
    """Heat constant-mass task plus a fixed IC target.

    Heat lives on a periodic grid with the duplicated right endpoint omitted.
    Periodicity is therefore part of the discrete function space rather than
    an independent vector of boundary degrees of freedom.  Adding an equality
    between the first and last stored cells would be physically wrong (they
    are adjacent cell locations, not duplicate endpoints), so Scenario 2 adds
    the target IC while recording periodic BC enforcement as structural.
    """
    nx, nt = template.nx, template.nt
    ic_target = reference_sample[:, 0].detach().to(device=device, dtype=torch.float32)

    def enforcement_residual(u_flat: torch.Tensor) -> torch.Tensor:
        u = u_flat.view(nx, nt)
        return torch.cat([
            conservation.enforcement_residual(u_flat),
            u[:, 0] - ic_target.to(u.device, u.dtype),
        ])

    def eval_residual(samples: np.ndarray, reference: Optional[dict] = None) -> dict[str, np.ndarray]:
        result = dict(conservation.eval_residual(samples, reference=reference))
        result["initial_condition"] = (
            np.asarray(samples, dtype=np.float64)[:, :, 0]
            - ic_target.detach().cpu().numpy()[None, :]
        )
        return result

    output_dim = conservation.output_dim + nx
    scales = torch.cat([
        conservation.scales.to(device=device, dtype=torch.float32),
        torch.ones(nx, device=device, dtype=torch.float32),
    ])
    # The composed Heat constraint is affine.  Recover its exact A u = c
    # representation once and retain the minimum-norm projector used by the
    # conservation-only task.
    mass_A = _heat_mass_matrix(nx, nt, float(template.dx), device, torch.float32)
    ic_A = torch.zeros(nx, nx, nt, device=device, dtype=torch.float32)
    spatial_rows = torch.arange(nx, device=device)
    ic_A[spatial_rows, spatial_rows, 0] = 1.0
    A = torch.cat([mass_A, ic_A.reshape(nx, nx * nt)], dim=0)
    with torch.no_grad():
        c = -enforcement_residual(torch.zeros(nx * nt, device=device)).detach()

    def native_projector(xi_flat, config=None):
        config = config or ProjectionConfig(scales=scales.to(xi_flat.device))
        return linear_minimum_norm_projection(
            xi_flat,
            A.to(device=xi_flat.device, dtype=xi_flat.dtype),
            c.to(device=xi_flat.device, dtype=xi_flat.dtype),
            config,
        )

    return ConstraintTask(
        name="heat_mass_conservation_ic_bc",
        dataset="diffusion",
        output_dim=output_dim,
        enforcement_residual=enforcement_residual,
        eval_residual=eval_residual,
        target=c,
        metadata={
            **conservation.metadata,
            "constraint_scenario": "conservation_ic_bc",
            "target_kind": "paired_well_posed_pde_problem_per_sample",
            "ic_target_shape": [nx],
            "bc_enforcement": "periodic topology of the endpoint-excluded spatial grid",
            "bc_residual_dim": 0,
            "residual_order": ["mass", "initial_condition"],
        },
        scales=scales,
        component_names=["mass", "initial_condition"],
        enforcement_jacobian=lambda u_flat: A.to(
            device=u_flat.device, dtype=u_flat.dtype
        ),
        units=["u*length", "u"],
        component_dims=[conservation.output_dim, nx],
        differentiable=True,
        differentiability_notes=(
            "Affine: whole-trajectory constant mass plus a fixed target IC. "
            "Periodic Heat BCs are structural on the endpoint-excluded periodic grid."
        ),
        native_projector=native_projector,
        display_name="Heat: constant mass + target IC + periodic BC",
    )


def _augment_rd_with_ic_bc(
    conservation: ConstraintTask,
    template: Residuals,
    reference_sample: torch.Tensor,
    device: str,
    boundary_target: Optional[torch.Tensor],
) -> ConstraintTask:
    """RD1D external-flux balance plus fixed IC and Neumann fluxes.

    Scenario 1 must infer boundary fluxes from the generated field because no
    physical BC is supplied. Scenario 2 has a prescribed BC, so its integral
    balance must use that external flux directly. Reusing the Scenario-1
    field-inferred flux here would define two different boundary fluxes in the
    same constraint system. The Neumann trace starts at ``t[1]``: arbitrary
    Vanilla-FFM initial conditions need not satisfy the classical corner
    compatibility condition at ``t=0``, while the parabolic initial-boundary
    value problem remains well posed for positive time.
    """
    nx, nt = template.nx, template.nt
    dx, nu = float(template.dx), float(template.nu)
    ic_target = reference_sample[:, 0].detach().to(device=device, dtype=torch.float32)
    if boundary_target is None:
        with torch.no_grad():
            left, right = _rd_boundary_fluxes_torch(reference_sample, dx, nu)
        bc_target = torch.stack([left, right]).to(device=device, dtype=torch.float32)
        bc_source = "reconstructed_from_target_solution"
    else:
        bc_target = torch.as_tensor(boundary_target, device=device, dtype=torch.float32)
        if bc_target.shape == (2,):
            bc_target = bc_target[:, None].expand(2, nt).clone()
        elif bc_target.shape != (2, nt):
            raise ValueError(
                f"RD1D boundary_target must have shape (2,) or (2, {nt}), got {tuple(bc_target.shape)}"
            )
        bc_source = "prescribed_external_neumann_flux"

    def external_balance(u: torch.Tensor) -> torch.Tensor:
        dx_tensor = template.dx.to(u.device, u.dtype)
        dt = (template.t_grid[1:] - template.t_grid[:-1]).to(u.device, u.dtype)
        sol = u.T
        mass = sol.sum(dim=1) * dx_tensor
        source = template.rho * (sol * (1.0 - sol)).sum(dim=1) * dx_tensor
        source_mid = 0.5 * (source[:-1] + source[1:])
        source_cumulative = torch.cat([
            torch.zeros(1, device=u.device, dtype=u.dtype),
            torch.cumsum(source_mid * dt, dim=0),
        ])
        target = bc_target.to(u.device, u.dtype)
        boundary_net = target[0] - target[1]
        boundary_mid = 0.5 * (boundary_net[:-1] + boundary_net[1:])
        boundary_cumulative = torch.cat([
            torch.zeros(1, device=u.device, dtype=u.dtype),
            torch.cumsum(boundary_mid * dt, dim=0),
        ])
        return (
            mass - (mass[0] + source_cumulative + boundary_cumulative)
        )[1:]

    def enforcement_residual(u_flat: torch.Tensor) -> torch.Tensor:
        u = u_flat.view(nx, nt)
        left, right = _rd_boundary_fluxes_torch(u, dx, nu)
        target = bc_target.to(u.device, u.dtype)
        return torch.cat([
            external_balance(u),
            u[:, 0] - ic_target.to(u.device, u.dtype),
            left[1:] - target[0, 1:],
            right[1:] - target[1, 1:],
        ])

    def enforcement_jacobian(u_flat: torch.Tensor) -> torch.Tensor:
        """Exact Jacobian of external balance, paired IC, and flux traces."""
        u = u_flat.view(nx, nt)
        m = nt - 1
        output_dim = m + nx + 2 * m
        jacobian = torch.zeros(
            (output_dim, nx, nt), device=u.device, dtype=u.dtype
        )
        jacobian[:m].copy_(
            template.mass_residual_rd_jacobian(
                u_flat, include_boundary_flux=False,
            ).view(m, nx, nt)
        )

        spatial_rows = torch.arange(nx, device=u.device)
        jacobian[m + spatial_rows, spatial_rows, 0] = 1.0

        dx_tensor = template.dx.to(u.device, u.dtype)
        left_coefficients = (
            -nu / (12.0 * dx_tensor)
        ) * torch.tensor(
            [-25.0, 48.0, -36.0, 16.0, -3.0],
            device=u.device, dtype=u.dtype,
        )
        right_coefficients = (
            -nu / (12.0 * dx_tensor)
        ) * torch.tensor(
            [3.0, -16.0, 36.0, -48.0, 25.0],
            device=u.device, dtype=u.dtype,
        )
        time_rows = torch.arange(1, nt, device=u.device)
        left_start = m + nx
        right_start = left_start + m
        for offset in range(5):
            jacobian[left_start + time_rows - 1, offset, time_rows] = left_coefficients[offset]
            jacobian[right_start + time_rows - 1, nx - 5 + offset, time_rows] = right_coefficients[offset]
        return jacobian.reshape(output_dim, nx * nt)

    def eval_residual(samples: np.ndarray, reference: Optional[dict] = None) -> dict[str, np.ndarray]:
        values = np.asarray(samples, dtype=np.float64)
        target_np = bc_target.detach().cpu().numpy().astype(np.float64)
        left, right = _rd_boundary_fluxes_numpy(values, dx, nu)
        sol = np.transpose(values, (0, 2, 1))
        dt = np.diff(template.t_grid.detach().cpu().numpy().astype(np.float64))
        mass = sol.sum(axis=2) * dx
        source = float(template.rho) * (sol * (1.0 - sol)).sum(axis=2) * dx
        source_mid = 0.5 * (source[:, :-1] + source[:, 1:])
        source_cumulative = np.concatenate([
            np.zeros((len(values), 1), dtype=np.float64),
            np.cumsum(source_mid * dt[None, :], axis=1),
        ], axis=1)
        boundary_net = target_np[0] - target_np[1]
        boundary_mid = 0.5 * (boundary_net[:-1] + boundary_net[1:])
        boundary_cumulative = np.concatenate([
            np.zeros(1, dtype=np.float64),
            np.cumsum(boundary_mid * dt),
        ])
        result = {
            "global_balance": (
                mass - (
                    mass[:, :1] + source_cumulative
                    + boundary_cumulative[None, :]
                )
            )[:, 1:],
            "initial_condition": values[:, :, 0] - ic_target.detach().cpu().numpy()[None, :],
            "left_boundary_flux": left[:, 1:] - target_np[None, 0, 1:],
            "right_boundary_flux": right[:, 1:] - target_np[None, 1, 1:],
        }
        return result

    output_dim = conservation.output_dim + nx + 2 * (nt - 1)
    # IC values are O(1); training Neumann fluxes lie in [-0.05, 0.05].
    # Scaling the two flux blocks by that physical range prevents their small
    # units from being ignored by normalized guidance/projector objectives.
    scales = torch.cat([
        conservation.scales.to(device=device, dtype=torch.float32),
        torch.ones(nx, device=device, dtype=torch.float32),
        torch.full((2 * (nt - 1),), 0.05, device=device, dtype=torch.float32),
    ])
    target = torch.cat([
        conservation.target.to(device=device, dtype=torch.float32),
        ic_target,
        bc_target[:, 1:].reshape(-1),
    ])
    return ConstraintTask(
        name="global_balance_ic_bc",
        dataset="rd1d",
        output_dim=output_dim,
        enforcement_residual=enforcement_residual,
        eval_residual=eval_residual,
        target=target,
        metadata={
            **conservation.metadata,
            "constraint_scenario": "conservation_ic_bc",
            "target_kind": "paired_well_posed_pde_problem_per_sample",
            "ic_target_shape": [nx],
            "bc_type": "prescribed_neumann_flux",
            "bc_target_source": bc_source,
            "bc_target_shape": [2, nt],
            "bc_residual_shape": [2, nt - 1],
            "bc_time_indices": "1..nt-1 (t=0 corner compatibility not imposed)",
            "global_balance_flux_source": "prescribed_external_neumann_flux",
            "boundary_stencil": "five-point one-sided fourth-order derivative",
            "residual_order": [
                "global_balance", "initial_condition",
                "left_boundary_flux", "right_boundary_flux",
            ],
        },
        scales=scales,
        component_names=[
            "global_balance", "initial_condition",
            "left_boundary_flux", "right_boundary_flux",
        ],
        enforcement_jacobian=enforcement_jacobian,
        units=["mass", "u", "flux", "flux"],
        component_dims=[conservation.output_dim, nx, nt - 1, nt - 1],
        differentiable=True,
        differentiability_notes=(
            "Nonlinear global balance plus a fixed target IC and prescribed Neumann "
            "fluxes; generic damped Gauss-Newton projection is required."
        ),
        native_projector=None,
        display_name="RD1D: global mass balance + target IC + Neumann BC",
    )


# ---------------------------------------------------------------------------
# Per-task builders (private)
# ---------------------------------------------------------------------------

def _build_heat_mass_conservation(
    template: Residuals, ref: Optional[torch.Tensor], device: str,
    target_override: Optional[torch.Tensor] = None,
) -> ConstraintTask:
    """Whole-trajectory constant mass, relative to the sample's own t=0 mass."""
    nx, nt, dx = template.nx, template.nt, float(template.dx)
    m = nt - 1
    target = torch.zeros(m, device=device, dtype=torch.float32)
    if target_override is not None:
        if target_override.numel() != m:
            raise ValueError(f"heat_mass_conservation target must have {m} entries")
        target = target_override.to(device=device, dtype=torch.float32)

    def enforcement_residual(u_flat: torch.Tensor) -> torch.Tensor:
        # Canonical implementation: M(t_j) - M(t_0), j=1,...,nt-1.
        return template.mass_residual_heat(u_flat) - target.to(u_flat.device, u_flat.dtype)

    A = _heat_mass_matrix(nx, nt, dx, device, torch.float32)

    def enforcement_jacobian(u_flat: torch.Tensor) -> torch.Tensor:
        return A.to(device=u_flat.device, dtype=u_flat.dtype)

    def eval_residual(samples: np.ndarray, reference: Optional[dict] = None) -> dict[str, np.ndarray]:
        # Keep this evaluation-side definition colocated with the canonical
        # enforcement residual.  A former import referenced a removed
        # ``eval_pde.residuals.heat`` module and caused an otherwise completed
        # 1,000-sample Heat run to fail while writing metrics.
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (nx, nt):
            raise ValueError(
                f"Heat samples must have shape (N, {nx}, {nt}), got {values.shape}"
            )
        mass = dx * np.sum(values, axis=1)
        residual = mass[:, 1:] - mass[:, :1]
        return {
            "mass": residual - target.detach().cpu().numpy().astype(np.float64)[None, :]
        }

    def native_projector(xi_flat, config=None):
        config = config or ProjectionConfig(scales=torch.ones(m, device=xi_flat.device))
        result = linear_minimum_norm_projection(
            xi_flat, A, target.to(device=xi_flat.device, dtype=xi_flat.dtype), config,
        )
        # This task's frozen feasibility statistic is componentwise L-infinity,
        # not the generic projector library's vector L2 statistic.
        with torch.no_grad():
            before = float(enforcement_residual(xi_flat).abs().max().item())
            after = float(enforcement_residual(result.u).abs().max().item())
        result.initial_residual_norm = before
        result.final_residual_norm = after
        result.residual_norm_history = [before, after]
        result.feasible = (not result.warnings) and after <= config.tol
        result.converged = result.feasible
        return result

    return ConstraintTask(
        name="heat_mass_conservation", dataset="diffusion", output_dim=m,
        enforcement_residual=enforcement_residual, eval_residual=eval_residual,
        target=target,
        metadata={
            "nx": nx, "nt": nt, "dx": dx, "axis_order": ["x", "t"],
            "spatial_domain": [0.0, 2.0 * np.pi], "spatial_endpoint_included": False,
            "residual_definition": "H_j=dx*sum_i(u[i,j]-u[i,0]), j=1,...,nt-1",
            "mass_target": "generated_sample_M0", "primary_residual": "linf",
        },
        scales=torch.ones(m, device=device), component_names=["mass"],
        enforcement_jacobian=enforcement_jacobian,
        units=["u*length"], component_dims=[m], differentiable=True,
        differentiability_notes="Exactly linear and self-referential; no IC, BC, or external target.",
        native_projector=native_projector,
        display_name="Heat: whole-trajectory constant mass",
    )
def _build_rd_global_balance(
    template: Residuals, ref: Optional[torch.Tensor], device: str,
    target_override: Optional[torch.Tensor] = None,
) -> ConstraintTask:
    nx, nt = template.nx, template.nt
    m = nt - 1
    # Self-referential: target is fixed at zero by the mass-balance identity
    # itself, not something to select (see sampling/observable_bank.py
    # module docstring). target_override is accepted for interface
    # consistency but only honored if its shape actually matches m.
    target = torch.zeros(m)
    if target_override is not None and target_override.numel() == m:
        target = target_override

    def enforcement_residual(u_flat: torch.Tensor) -> torch.Tensor:
        # Reuses sampling.constraints.Residuals.mass_residual_rd directly —
        # already the codebase's existing global, whole-trajectory
        # mass-balance identity; dropping the trivial t=0 row matches
        # full_residual_rd's convention.
        return template.mass_residual_rd(u_flat)[1:] - target.to(u_flat.device, u_flat.dtype)

    def enforcement_jacobian(u_flat: torch.Tensor) -> torch.Tensor:
        return template.mass_residual_rd_jacobian(u_flat)

    def eval_residual(samples: np.ndarray, reference: Optional[dict] = None) -> dict[str, np.ndarray]:
        from eval_pde.residuals.reaction_diffusion import mass_residual_rd_batched
        x_np = template.x.detach().cpu().numpy()
        t_np = template.t_grid.detach().cpu().numpy()
        res = mass_residual_rd_batched(samples, x=x_np, t_grid=t_np, rho=float(template.rho), nu=float(template.nu))
        return {"global_balance": res - target.detach().cpu().numpy()[None, :]}

    return ConstraintTask(
        name="global_balance", dataset="rd1d", output_dim=m,
        enforcement_residual=enforcement_residual, eval_residual=eval_residual,
        target=target, metadata={"nx": nx, "nt": nt, "rho": float(template.rho), "nu": float(template.nu)},
        scales=torch.ones(m), component_names=["global_balance"], units=["mass"], component_dims=[m],
        enforcement_jacobian=enforcement_jacobian,
        differentiable=True,
        differentiability_notes="Self-referential (no reference sample needed); quadratic in u "
                                  "via the reaction term -> Gauss-Newton projector only.",
        native_projector=None,
        display_name="RD1D: whole-trajectory global mass-balance residual",
    )


_TASK_BUILDERS: dict[tuple[str, str], Callable[[Residuals, Optional[torch.Tensor], str], ConstraintTask]] = {
    ("diffusion", "heat_mass_conservation"): _build_heat_mass_conservation,
    ("rd1d", "global_balance"): _build_rd_global_balance,
}
