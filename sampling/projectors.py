# Shared projector library for enforcing constraint tasks onto generated fields.
"""
sampling/projectors.py
========================
Shared numerical projection primitives used to enforce a
:class:`~sampling.tasks.ConstraintTask`'s residual onto a candidate field.

This module implements the projector math shared by ECI, final projection,
and MintFlow's terminal-projection variants.

Two projector families are provided, matched to the production constraints:

* :func:`linear_minimum_norm_projection` — exact, one-shot, for constraints
  ``H(u) = A u - c`` with a constant Jacobian ``A`` (Heat).
* :func:`damped_gauss_newton_projection` — the general-purpose iterative
  fallback for arbitrary (possibly nonlinear) differentiable ``hfunc``,
  generalizing the original PCFM single-step projection's
  Newton step into a damped (Levenberg-Marquardt-style), tolerance-stopped,
  multi-iteration solver with explicit diagnostics.

Both return a :class:`ProjectionResult` with the same fields, so callers
do not need to special-case which projector produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

__all__ = [
    "ProjectionConfig",
    "ProjectionResult",
    "feasibility_check",
    "conditioning_diagnostics",
    "linear_minimum_norm_projection",
    "damped_gauss_newton_projection",
]


# ---------------------------------------------------------------------------
# Shared config / result types
# ---------------------------------------------------------------------------

@dataclass
class ProjectionConfig:
    """Shared knobs for every projector in this module.

    Parameters
    ----------
    max_iter : Maximum Gauss-Newton iterations (ignored by the one-shot
               linear projector, which reports ``iterations=1``).
    damping  : Levenberg-Marquardt-style ridge added to ``J J^T`` (or
               ``A A^T``) before solving — improves conditioning when the
               residual Jacobian is rank-deficient or near-singular.
    tol      : Convergence / feasibility threshold on the *scaled* residual
               norm (see ``scales``).
    scales   : Optional per-component normalization scales (broadcastable to
               the residual shape). Dividing by ``scales`` before computing
               norms/tolerances is what lets heterogeneous mass, IC, and BC
               residuals be combined fairly. ``None`` means no normalization.
    """
    max_iter: int = 10
    damping: float = 1e-6
    tol: float = 1e-6
    scales: Optional[torch.Tensor] = None
    # Computing a full Jacobian SVD is useful for terminal diagnostics but
    # prohibitively expensive when a nonlinear projector is called at every
    # ECI mixing step. Callers may disable it without changing the update.
    compute_condition_number: bool = True


@dataclass
class ProjectionResult:
    """Uniform output of every projector in this module."""
    u: torch.Tensor
    converged: bool
    iterations: int
    residual_norm_history: list[float]
    initial_residual_norm: float
    final_residual_norm: float
    feasible: bool
    condition_number: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    method: str = ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _scaled_norm(residual: Any, scales: Optional[torch.Tensor] = None) -> float:
    """L2 norm of *residual*, optionally divided elementwise by *scales*."""
    if not torch.is_tensor(residual):
        residual = torch.as_tensor(residual, dtype=torch.float64)
    residual = residual.reshape(-1).to(torch.float64)
    if scales is not None:
        scales_t = scales if torch.is_tensor(scales) else torch.as_tensor(scales, dtype=torch.float64)
        # Scales may follow the task's device while a diagnostic residual is
        # on CPU, so always align devices explicitly.
        scales_t = scales_t.reshape(-1).to(device=residual.device, dtype=torch.float64)
        residual = residual / scales_t.clamp_min(1e-300)
    return float(torch.linalg.norm(residual).item())


def feasibility_check(
    residual: torch.Tensor, tol: float, scales: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Check whether *residual* is within *tol* (optionally normalized by *scales*).

    Returns a dict with both the raw and scaled residual norms so callers can
    always recover the un-normalized magnitude for logging.
    """
    raw_norm = _scaled_norm(residual, scales=None)
    scaled_norm = _scaled_norm(residual, scales=scales)
    return {
        "feasible": scaled_norm <= tol,
        "residual_norm": raw_norm,
        "scaled_residual_norm": scaled_norm,
        "tol": float(tol),
    }


def conditioning_diagnostics(J: torch.Tensor) -> dict[str, Any]:
    """Singular-value-based conditioning diagnostics for a Jacobian *J* (m, n)."""
    if J.ndim == 1:
        J = J.unsqueeze(0)
    with torch.no_grad():
        try:
            svals = torch.linalg.svdvals(J.to(torch.float64))
        except RuntimeError:
            return {"cond_number": float("inf"), "min_singular_value": 0.0,
                    "max_singular_value": float("nan"), "rank_estimate": 0}
        smax = float(svals.max().item())
        smin = float(svals.min().item())
        cond = smax / smin if smin > 1e-300 else float("inf")
        rank_est = int((svals > 1e-8 * max(smax, 1e-300)).sum().item())
    return {
        "cond_number": cond,
        "min_singular_value": smin,
        "max_singular_value": smax,
        "rank_estimate": rank_est,
    }


# ---------------------------------------------------------------------------
# 1. Linear minimum-norm projection (exact, one-shot)
# ---------------------------------------------------------------------------

def linear_minimum_norm_projection(
    xi: torch.Tensor,
    A: torch.Tensor,
    c: torch.Tensor,
    config: Optional[ProjectionConfig] = None,
) -> ProjectionResult:
    """Minimum-norm correction of *xi* onto ``{u : A u = c}``.

    Closed form (no iteration required, since ``H(u) = A u - c`` has a
    *constant* Jacobian ``A``):

        u = xi - A^T (A A^T + damping*I)^-1 (A xi - c)

    Parameters
    ----------
    xi : ``(n,)`` flattened candidate field.
    A  : ``(m, n)`` constant Jacobian of the (affine) constraint.
    c  : ``(m,)`` target vector.
    """
    config = config or ProjectionConfig()
    A = A.to(device=xi.device, dtype=xi.dtype)
    c = c.to(device=xi.device, dtype=xi.dtype)

    residual0 = A @ xi - c
    initial_norm = _scaled_norm(residual0, config.scales)

    AAt = A @ A.transpose(-2, -1)
    damped = AAt + config.damping * torch.eye(AAt.shape[0], device=xi.device, dtype=xi.dtype)

    warnings: list[str] = []
    solve_failed = False
    try:
        lam = torch.linalg.solve(damped, residual0)
        u = xi - A.transpose(-2, -1) @ lam
    except RuntimeError:
        warnings.append("linalg_solve_failed")
        solve_failed = True
        u = xi.clone()

    residual1 = A @ u - c
    final_norm = _scaled_norm(residual1, config.scales)
    cond = conditioning_diagnostics(A)["cond_number"]

    return ProjectionResult(
        u=u,
        converged=(not solve_failed) and final_norm <= config.tol,
        iterations=1,
        residual_norm_history=[initial_norm, final_norm],
        initial_residual_norm=initial_norm,
        final_residual_norm=final_norm,
        feasible=(not solve_failed) and final_norm <= config.tol,
        condition_number=cond,
        warnings=warnings,
        method="linear_minimum_norm",
    )


# ---------------------------------------------------------------------------
# 2. Damped Gauss-Newton projection for nonlinear tasks
# ---------------------------------------------------------------------------
def damped_gauss_newton_projection(
    xi: torch.Tensor,
    hfunc: Callable[[torch.Tensor], torch.Tensor],
    config: Optional[ProjectionConfig] = None,
) -> ProjectionResult:
    """Iterative minimum-norm-relative-to-``xi`` projection for arbitrary
    differentiable ``hfunc``.

    Generalizes the original PCFM single-step projection's
    Newton step: at each iteration, linearize ``hfunc`` about the current
    iterate ``u`` and solve

        min_du  ||du - (xi - u)||^2   s.t.   J du = J(xi - u) - h(u)

    (a damped Levenberg-Marquardt normal-equations solve), then updates
    ``u <- u + du`` and stops early once the *scaled* residual norm drops
    below ``config.tol`` or ``config.max_iter`` is exhausted.

    ``hfunc`` must be a pure function of a flattened ``(n,)`` tensor,
    returning a residual vector with target already baked in (matching the
    production ``ConstraintTask`` convention).
    """
    from sampling.pcfm_sampling import compute_jacobian

    config = config or ProjectionConfig()
    u = xi.clone().detach().requires_grad_(True)
    warnings: list[str] = []
    history: list[float] = []
    converged = False
    iterations_used = 0
    cond: Optional[float] = None

    with torch.no_grad():
        h0 = hfunc(u)
        if h0.ndim == 0:
            h0 = h0.unsqueeze(0)
    initial_norm = _scaled_norm(h0, config.scales)
    history.append(initial_norm)
    if initial_norm <= config.tol:
        converged = True

    for it in range(config.max_iter):
        if converged:
            break
        h_val = hfunc(u)
        if h_val.ndim == 0:
            h_val = h_val.unsqueeze(0)
        try:
            J = compute_jacobian(hfunc, u)
        except RuntimeError as exc:
            warnings.append(f"jacobian_failed:{type(exc).__name__}")
            break
        if J.ndim == 1:
            J = J.unsqueeze(0)

        JJt = J @ J.transpose(-2, -1)
        damped = JJt + config.damping * torch.eye(JJt.shape[0], device=u.device, dtype=JJt.dtype)
        delta = (xi - u).unsqueeze(-1)
        rhs = J @ delta + h_val.detach().unsqueeze(-1)
        try:
            lam = torch.linalg.solve(damped, rhs)
        except RuntimeError:
            warnings.append("linalg_solve_failed")
            break
        du = delta - J.transpose(-2, -1) @ lam
        u = (u.detach() + du.squeeze(-1).detach()).requires_grad_(True)
        iterations_used = it + 1

        with torch.no_grad():
            h_new = hfunc(u)
            if h_new.ndim == 0:
                h_new = h_new.unsqueeze(0)
            res_norm = _scaled_norm(h_new, config.scales)
        if not np.isfinite(res_norm):
            warnings.append("nan_in_residual")
            history.append(res_norm)
            break
        history.append(res_norm)
        if res_norm <= config.tol:
            converged = True

    if config.compute_condition_number:
        try:
            with torch.no_grad():
                u_for_jac = u.detach().requires_grad_(True)
                J_final = compute_jacobian(hfunc, u_for_jac)
            cond = conditioning_diagnostics(J_final)["cond_number"]
        except Exception:
            cond = None

    final_norm = history[-1] if history else float("nan")
    return ProjectionResult(
        u=u.detach(),
        converged=converged,
        iterations=iterations_used,
        residual_norm_history=history,
        initial_residual_norm=history[0],
        final_residual_norm=final_norm,
        feasible=np.isfinite(final_norm) and final_norm <= config.tol,
        condition_number=cond,
        warnings=warnings,
        method="damped_gauss_newton",
    )
