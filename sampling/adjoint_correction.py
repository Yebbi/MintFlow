# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
"""
sampling/adjoint_correction.py
================================
Core mathematics for **MintFlow (Adjoint-Based Optimal Correction)**, a constrained
flow-matching sampling method (see ``Constrained_FM.pdf`` for the derivation).

Given a pretrained velocity field ``f_theta(u, t)`` and a differentiable terminal
constraint ``H(u_T) = 0``, MintFlow:

1. Forward-solves the flow ODE from noise ``u0`` to ``u_T`` (fixed-step Euler),
   storing the full trajectory.
2. Computes the terminal residual ``r = H(u_T)`` and uses the task's exact
   analytic Jacobian ``J = dH/du_T``.
3. Reverse-differentiates the stored explicit-Euler map using
   ``A_i = A_{i+1}(I + h_i J_f(t_i, u_i))``, using
   batched-cotangent :func:`torch.autograd.grad` VJPs, never forming
   ``df/du`` explicitly or replicating states across constraint rows.
4. At each candidate correction time ``s_k``, computes either the exact
   pseudoinverse or explicitly damped minimum-norm perturbation ``delta_k``
   that would reduce the linearised residual, and scores
   candidates by ``||delta_k||^2 + time_penalty * (T - s_k)^2``.
5. Applies the best perturbation and re-integrates the *original* flow from
   ``s*`` to ``T``.

Shape / storage convention
---------------------------
The production sampler operates on one sample at a time to keep FNO numerical
results invariant to the requested outer batch size. Constraint rows are
propagated together as batched cotangents, but the model state keeps batch size
one throughout the forward, reverse, and restarted solves.

We store the adjoint state directly as ``A(t) := lambda(t)^T``, shape
``(m, *spatial_shape)`` — i.e. **row ``k`` of ``A(t)`` equals column ``k`` of
``lambda(t)``**. This is a deliberate storage choice: since
``lambda(T) = J^T`` and the analytic task Jacobian ``J`` already has shape
``(m, d_flat)`` with row ``k`` = ``dH_k/du_T``, initialising ``A(T) := J``
requires no transpose. Downstream, ``A_k`` (the quantity in the correction
formula) is then exactly the stored tensor at each candidate time — no
transpose needed at any step. Only the *numerical values* matter for the VJP
update (each row/column carries a self-consistent set of numbers), not the
row/column labelling, so this is mathematically equivalent to the notes'
``lambda`` / ``A_k = lambda(s_k)^T`` formulation.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


class AdjointWorkspace:
    """Reusable adjoint/gradient buffers keyed by shape, device, and dtype.

    The CUDA caching allocator reuses raw storage, but allocating new Tensor
    objects for every sample and reverse step still adds host overhead. One
    workspace is owned by each sampler and reused throughout its evaluation
    loop. Candidate snapshots remain independent clones.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def acquire(self, template: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            template.device.type,
            template.device.index,
            template.dtype,
            tuple(template.shape),
        )
        buffers = self._buffers.get(key)
        if buffers is None:
            buffers = (torch.empty_like(template), torch.empty_like(template))
            self._buffers[key] = buffers
        return buffers

# ---------------------------------------------------------------------------
# 0. Parameter freezing (defensive: MintFlow never needs d(loss)/d(theta))
# ---------------------------------------------------------------------------

@contextmanager
def frozen_parameters(model: torch.nn.Module):
    """Temporarily force ``requires_grad=False`` on all of *model*'s parameters.

    MintFlow only ever needs ``d(f_theta)/d(u)`` (VJPs with respect to the
    *state*) — it never needs ``d(loss)/d(theta)``. Freezing parameters
    during the adjoint/backward passes (a) is a defensive guard against any
    future code path accidentally accumulating parameter ``.grad``, and
    (b) lets autograd skip saving tensors needed only for the (unused)
    weight-gradient branches of the graph, reducing peak memory. This has
    **no effect on computed values** — only on what autograd tracks/retains.
    """
    params = list(model.parameters())
    orig = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(params, orig):
            p.requires_grad_(flag)


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# ---------------------------------------------------------------------------
# 1. Forward solve
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_solve_with_trajectory(
    model: Callable, u0: torch.Tensor, n_step: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Fixed-step explicit-Euler forward solve, storing the full trajectory.

    Parameters
    ----------
    model : callable ``f_theta(t, u) -> du/dt``, batched-capable (``u`` keeps
            its leading batch dimension, here always ``1``).
    u0    : ``(1, *dims)`` initial noise sample.
    n_step: number of Euler steps.

    Returns
    -------
    traj : list of ``n_step + 1`` detached tensors, each ``(1, *dims)``,
           ``traj[i]`` = state at ``ts[i]``.
    ts   : ``(n_step + 1,)`` time grid, ``torch.linspace(0, 1, n_step + 1)``.
    """
    device = u0.device
    ts = torch.linspace(0.0, 1.0, n_step + 1, device=device, dtype=u0.dtype)
    u = u0.clone()
    traj = [u.detach().clone()]
    for i in range(n_step):
        vf = model(ts[i], u)
        # Use the actual stored grid increment. The exact discrete adjoint and
        # suffix restart reuse this same (t_i, h_i) pair.
        h_i = ts[i + 1] - ts[i]
        u = u + h_i * vf
        traj.append(u.detach().clone())
    return traj, ts


# ---------------------------------------------------------------------------
# 2. Terminal residual + Jacobian
# ---------------------------------------------------------------------------

def terminal_residual_and_jacobian(
    hfunc: Callable[[torch.Tensor], torch.Tensor],
    u_T_flat: torch.Tensor,
    jacobian_func: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute ``r = H(u_T)`` and ``J = dH/du_T`` for a single flattened sample.

    Parameters
    ----------
    hfunc    : differentiable residual function, ``(d_flat,) -> (m,)``.
    u_T_flat : ``(d_flat,)`` flattened terminal state.

    Returns
    -------
    r    : ``(m,)`` residual (supports scalar constraints via ``m == 1``).
    J    : ``(m, d_flat)`` analytic task Jacobian.
    info : dict with ``nan_in_residual`` / ``nan_in_jacobian`` bool flags.
    """
    u_flat = u_T_flat.detach().clone()
    with torch.no_grad():
        r = hfunc(u_flat)
        if r.ndim == 0:
            r = r.unsqueeze(0)
    with torch.no_grad():
        J = jacobian_func(u_flat)
    if J.ndim == 1:
        J = J.unsqueeze(0)
    r, J = r.detach(), J.detach()
    info = {
        "nan_in_residual": not _is_finite(r),
        "nan_in_jacobian": not _is_finite(J),
        "jacobian_backend": "analytic",
    }
    return r, J, info


# ---------------------------------------------------------------------------
# 3. Candidate time selection
# ---------------------------------------------------------------------------

def parse_candidate_time_range(range_str: str) -> Tuple[float, float]:
    """Parse ``\"t_min-t_max\"`` flow-time bounds, e.g. ``\"0.1-0.9\"``."""
    parts = range_str.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected candidate range 't_min-t_max', got {range_str!r}.")
    t_min, t_max = float(parts[0]), float(parts[1])
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError(f"Candidate range must satisfy 0 <= t_min < t_max <= 1, got {range_str!r}.")
    return t_min, t_max


def select_candidate_indices(
    n_step: int,
    num_candidates: int,
    t_min: float = 0.0,
    t_max: float = 1.0,
    time_sampling: str = "end_biased",
    end_bias_power: float = 2.0,
) -> List[int]:
    """Pick ``K`` candidate Euler-grid indices inside ``[t_min, t_max]``.

    ``time_sampling="end_biased"`` applies the endpoint-preserving warp
    ``w(u) = 1 - (1-u)**p`` to an evenly spaced unit grid, concentrating
    candidates near ``t_max`` when ``p > 1``. ``"uniform"`` retains the
    historical evenly spaced schedule. Requested continuous times are mapped
    to distinct nearest Euler-grid indices so rounding cannot silently reduce
    the candidate count when enough interior indices are available.
    """
    if n_step < 2:
        raise ValueError(
            f"MintFlow requires n_step >= 2 to have interior candidate times, got {n_step}."
        )
    idx_min = max(1, int(math.ceil(t_min * n_step)))
    idx_max = min(n_step - 1, int(math.floor(t_max * n_step)))
    if idx_min > idx_max:
        raise ValueError(
            f"No interior candidate indices for n_step={n_step}, "
            f"t_min={t_min}, t_max={t_max} (idx_min={idx_min}, idx_max={idx_max})."
        )
    span = idx_max - idx_min + 1
    num_candidates = max(1, min(num_candidates, span))
    if time_sampling not in {"end_biased", "uniform"}:
        raise ValueError(
            f"time_sampling must be 'end_biased' or 'uniform', got {time_sampling!r}."
        )
    if not math.isfinite(end_bias_power) or end_bias_power <= 0:
        raise ValueError(f"end_bias_power must be finite and positive, got {end_bias_power}.")
    if time_sampling == "end_biased" and end_bias_power <= 1:
        raise ValueError(
            f"end_biased sampling requires end_bias_power > 1, got {end_bias_power}."
        )

    unit = torch.linspace(0.0, 1.0, num_candidates, dtype=torch.float64)
    if time_sampling == "end_biased":
        unit = (
            torch.ones_like(unit)
            if num_candidates == 1
            else 1.0 - (1.0 - unit).pow(end_bias_power)
        )
    targets = idx_min + unit * (idx_max - idx_min)
    available = set(range(idx_min, idx_max + 1))
    selected: list[int] = []
    for target in targets.tolist():
        index = min(available, key=lambda candidate: (abs(candidate - target), candidate))
        selected.append(index)
        available.remove(index)
    return sorted(selected)


# ---------------------------------------------------------------------------
# 4. Backward adjoint solve
# ---------------------------------------------------------------------------
def discrete_euler_adjoint(
    model: Callable,
    traj: Sequence[torch.Tensor],
    ts: torch.Tensor,
    A_T: torch.Tensor,
    candidate_idxs: Sequence[int],
    batch_chunk_size: int = 64,
    workspace: Optional[AdjointWorkspace] = None,
) -> Tuple[Dict[int, torch.Tensor], bool]:
    """Exact reverse sensitivity of the stored explicit-Euler numerical map.

    For ``u_{i+1} = u_i + h_i f(t_i, u_i)``, rows of ``A`` obey

    ``A_i = A_{i+1} (I + h_i J_f(t_i, u_i))``.

    The VJP is therefore evaluated at the *earlier* stored state/time
    ``(traj[i], ts[i])``. No dense velocity-field Jacobian is formed.

    A chunk of adjoint rows is evaluated with :func:`torch.autograd.grad`'s
    batched-cotangent interface. The velocity model sees only the stored sample
    batch; constraint rows are VJP lanes rather than replicated model inputs.

    Finiteness checks stay on-device throughout the reverse solve.  Calling
    ``Tensor.item()`` inside this loop used to synchronize the GPU once per
    Euler step and materially reduced throughput.
    """
    n_step = len(traj) - 1
    if len(ts) != n_step + 1:
        raise ValueError("ts and traj must describe the same Euler grid")
    candidate_set = set(int(i) for i in candidate_idxs)
    if not candidate_set or min(candidate_set) < 0 or max(candidate_set) >= n_step:
        raise ValueError("candidate indices must lie in [0, n_step)")
    if batch_chunk_size <= 0:
        raise ValueError("batch_chunk_size must be positive")

    if int(traj[0].shape[0]) != 1:
        raise ValueError("MintFlow's production adjoint requires a one-sample trajectory")
    if A_T.ndim != traj[0].ndim:
        raise ValueError("A_T must have shape (m, *spatial_dims)")

    if workspace is None:
        A = torch.empty_like(A_T)
        grad_u = torch.empty_like(A_T)
    else:
        A, grad_u = workspace.acquire(A_T)
    A.copy_(A_T.detach())
    m = A.shape[0]
    results: Dict[int, torch.Tensor] = {}
    # Accumulate numerical failures on-device.  Convert to a Python bool only
    # once after the loop to avoid a device synchronization at every step.
    nan_detected_device = torch.zeros((), dtype=torch.bool, device=A.device)
    min_idx = min(candidate_set)
    step_sizes = (ts[1:] - ts[:-1]).detach().cpu().tolist()
    effective_chunk_size = min(m, batch_chunk_size)

    with frozen_parameters(model):
        for i in range(n_step - 1, min_idx - 1, -1):
            u_i = traj[i].detach().requires_grad_(True)
            for start in range(0, m, effective_chunk_size):
                chunk = A[start : start + effective_chunk_size]
                c = chunk.shape[0]
                if c > 1:
                    # One primal/model evaluation, c vector-Jacobian products.
                    # autograd's leading grad_outputs dimension enumerates the
                    # cotangent batch and is not a model/sample batch.
                    with torch.enable_grad():
                        vf = model(ts[i], u_i)
                        (grad_chunk,) = torch.autograd.grad(
                            vf,
                            u_i,
                            grad_outputs=chunk.unsqueeze(1),
                            is_grads_batched=True,
                            retain_graph=False,
                            create_graph=False,
                        )
                    grad_u[start : start + c].copy_(grad_chunk.squeeze(1).detach())
                else:
                    # A single cotangent row uses the standard VJP path; it
                    # needs neither state replication nor a batched transform.
                    with torch.enable_grad():
                        vf = model(ts[i], u_i)
                        (grad_chunk,) = torch.autograd.grad(
                            vf,
                            u_i,
                            grad_outputs=chunk[0].unsqueeze(0),
                            retain_graph=False,
                            create_graph=False,
                        )
                    grad_u[start].copy_(grad_chunk.squeeze(0).detach())

            nan_detected_device.logical_or_(~torch.isfinite(grad_u).all())
            torch.nan_to_num_(grad_u, nan=0.0, posinf=0.0, neginf=0.0)
            A.add_(grad_u, alpha=float(step_sizes[i]))
            if i in candidate_set:
                results[i] = A.clone()
    return results, bool(nan_detected_device.item())


# ---------------------------------------------------------------------------
# 5. Candidate correction (exact pseudoinverse or explicit damping)
# ---------------------------------------------------------------------------

def pseudoinverse_minimum_norm_correction(
    A_k_flat: torch.Tensor,
    r: torch.Tensor,
    rcond: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Document-faithful SVD correction ``delta = -pinv(A) r``.

    A compact SVD of ``A`` avoids squaring its condition number through the
    normal-equation matrix ``A A^T``. This is intentionally distinct from
    :func:`damped_minimum_norm_correction`.
    """
    effective_rank = 0
    try:
        # Compact SVD gives the pseudoinverse action without materialising the
        # potentially large (d x m) pseudoinverse and also supplies condition
        # diagnostics without a second decomposition.
        U, singular_values, Vh = torch.linalg.svd(A_k_flat, full_matrices=False)
        relative_cutoff = (
            float(rcond)
            if rcond is not None
            else max(A_k_flat.shape) * torch.finfo(A_k_flat.dtype).eps
        )
        cutoff = relative_cutoff * singular_values.max()
        keep = singular_values > cutoff
        effective_rank = int(keep.sum().item())
        coefficients = torch.zeros_like(singular_values)
        coefficients[keep] = (U.T @ r)[keep] / singular_values[keep]
        delta = -(Vh.T @ coefficients)
        solve_failed = False
        if effective_rank:
            kept = singular_values[keep]
            cond = float((kept.max() / kept.min()).item())
        else:
            cond = float("inf")
    except Exception:
        # ``lstsq`` is retained only as a recorded backend-failure fallback.
        delta = torch.linalg.lstsq(A_k_flat, -r.unsqueeze(-1)).solution.squeeze(-1)
        solve_failed = True
        cond = float("nan")
    predicted = A_k_flat @ delta + r
    info: Dict[str, Any] = {
        "correction_mode": "pseudoinverse",
        "ridge": 0.0,
        "rcond": rcond,
        "solve_failed": solve_failed,
        "ill_conditioned": False,
        "cond": cond,
        "effective_rank": effective_rank,
        "linearized_residual_norm": float(predicted.norm().item()),
    }
    info["ill_conditioned"] = (
        not math.isfinite(info["cond"])
        or effective_rank < min(A_k_flat.shape)
    )
    info["nan_in_delta"] = not _is_finite(delta)
    if info["nan_in_delta"]:
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    return delta, info


def damped_minimum_norm_correction(
    A_k_flat: torch.Tensor, r: torch.Tensor, ridge: float, cond_threshold: float = 1e8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Damped correction ``-A^T (A A^T + ridge I)^-1 r``.

    Reduces exactly to the scalar closed form
    ``delta = -r * lambda / (||lambda||^2 + ridge)`` when ``m == 1``.

    Numerical safety: checks the condition number of ``(A A^T + ridge*I)``
    and falls back to a least-squares solve (:func:`torch.linalg.lstsq`,
    robust to singular/near-singular matrices) if the direct
    :func:`torch.linalg.solve` raises (e.g. the matrix is exactly singular).
    Non-finite outputs are zeroed (treated as "no usable correction" for this
    candidate) rather than silently propagating NaNs/Infs into the sample.

    Parameters
    ----------
    A_k_flat       : ``(m, d_flat)``.
    r              : ``(m,)``.
    ridge          : regularisation added to the diagonal of ``A A^T``.
    cond_threshold : condition number above which ``ill_conditioned`` is flagged.

    Returns
    -------
    delta : ``(d_flat,)`` correction vector.
    info  : dict with ``cond``, ``ill_conditioned``, ``solve_failed``,
            ``nan_in_delta`` diagnostics for this candidate.
    """
    if ridge <= 0:
        raise ValueError("damped correction mode requires ridge > 0")
    m = A_k_flat.shape[0]
    AAt = A_k_flat @ A_k_flat.T
    reg = AAt + ridge * torch.eye(m, device=A_k_flat.device, dtype=A_k_flat.dtype)

    info: Dict[str, Any] = {
        "correction_mode": "damped", "ridge": float(ridge),
        "solve_failed": False, "ill_conditioned": False, "cond": float("nan"),
    }
    try:
        cond = float(torch.linalg.cond(reg).item())
        info["cond"] = cond
        if not math.isfinite(cond) or cond > cond_threshold:
            info["ill_conditioned"] = True
    except Exception:
        info["ill_conditioned"] = True

    try:
        lam_ = torch.linalg.solve(reg, r)
    except Exception:
        info["solve_failed"] = True
        lam_ = torch.linalg.lstsq(reg, r.unsqueeze(-1)).solution.squeeze(-1)

    delta = -A_k_flat.T @ lam_
    info["linearized_residual_norm"] = float((A_k_flat @ delta + r).norm().item())
    info["nan_in_delta"] = not _is_finite(delta)
    if info["nan_in_delta"]:
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    return delta, info


def solve_linearized_correction(
    A_k_flat: torch.Tensor,
    r: torch.Tensor,
    *,
    mode: str,
    ridge: float = 0.0,
    rcond: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Dispatch an explicitly named exact or damped linearized correction."""
    if mode == "pseudoinverse":
        return pseudoinverse_minimum_norm_correction(A_k_flat, r, rcond=rcond)
    if mode == "damped":
        return damped_minimum_norm_correction(A_k_flat, r, ridge=ridge)
    raise ValueError(f"Unknown MintFlow correction mode {mode!r}")


# ---------------------------------------------------------------------------
# 6. Candidate scoring / selection
# ---------------------------------------------------------------------------

def boundary_penalty(s: float, t0: float, t1: float, eps: float = 1e-3) -> float:
    """Symmetric penalty that grows near ``t0`` and ``t1`` without blowing up."""
    left = max(s - t0 + eps, eps)
    right = max(t1 - s + eps, eps)
    return 1.0 / (left * right)


def _normalize_terms(terms: Dict[int, float]) -> Dict[int, float]:
    if not terms:
        return {}
    mn = min(terms.values())
    mx = max(terms.values())
    if mx - mn <= 1e-12:
        return {k: 0.0 for k in terms}
    return {k: (v - mn) / (mx - mn) for k, v in terms.items()}


def compute_candidate_scores(
    deltas: Dict[int, torch.Tensor],
    ts: torch.Tensor,
    *,
    T: float = 1.0,
    time_penalty: float = 0.01,
    score_mode: str = "default",
    interior_penalty: float = 0.0,
    correction_weight: float = 1.0,
    r: Optional[torch.Tensor] = None,
    A_flat_by_idx: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    """Score every candidate; return ``all_scores`` and per-candidate term breakdown."""
    if not deltas:
        raise ValueError("compute_candidate_scores requires at least one candidate delta.")

    sorted_idxs = sorted(deltas.keys())
    t0 = ts[sorted_idxs[0]].item()
    t1 = ts[sorted_idxs[-1]].item()

    correction_terms = {idx: float(deltas[idx].pow(2).sum().item()) for idx in deltas}
    time_terms = {idx: (T - ts[idx].item()) ** 2 for idx in deltas}
    boundary_terms = {
        idx: boundary_penalty(ts[idx].item(), t0, t1) for idx in deltas
    }
    linearized_terms: Dict[int, float] = {}
    if score_mode == "residual_predictive":
        if r is None or A_flat_by_idx is None:
            raise ValueError("residual_predictive score mode requires r and A_flat_by_idx.")
        for idx, delta in deltas.items():
            A_k = A_flat_by_idx[idx]
            delta_flat = delta.reshape(-1)
            lin_res = A_k @ delta_flat + r
            linearized_terms[idx] = float(lin_res.pow(2).sum().item())

    norm_corr = _normalize_terms(correction_terms)
    norm_time = _normalize_terms(time_terms)

    all_scores: Dict[int, float] = {}
    breakdown: Dict[int, Dict[str, float]] = {}
    for idx in deltas:
        corr = correction_terms[idx]
        time_term = time_terms[idx]
        bnd = boundary_terms[idx]
        detail: Dict[str, float] = {
            "correction_term": corr,
            "time_term": time_term,
            "boundary_term": bnd,
        }
        if score_mode == "default":
            score = corr + time_penalty * time_term
        elif score_mode == "normalized":
            score = norm_corr[idx] + time_penalty * norm_time[idx]
        elif score_mode == "interior_penalty":
            score = corr + time_penalty * time_term + interior_penalty * bnd
        elif score_mode == "residual_predictive":
            lin = linearized_terms[idx]
            detail["linearized_residual_term"] = lin
            score = lin + correction_weight * corr + time_penalty * time_term
        else:
            raise ValueError(f"Unknown MintFlow score_mode {score_mode!r}.")
        detail["score"] = score
        all_scores[idx] = score
        breakdown[idx] = detail
    return all_scores, breakdown


def select_best_candidate(
    deltas: Dict[int, torch.Tensor],
    ts: torch.Tensor,
    T: float = 1.0,
    time_penalty: float = 0.01,
    *,
    score_mode: str = "default",
    interior_penalty: float = 0.0,
    correction_weight: float = 1.0,
    r: Optional[torch.Tensor] = None,
    A_flat_by_idx: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[int, torch.Tensor, float, Dict[int, float], Dict[int, Dict[str, float]]]:
    """Select the minimum-score MintFlow correction candidate."""
    all_scores, breakdown = compute_candidate_scores(
        deltas,
        ts,
        T=T,
        time_penalty=time_penalty,
        score_mode=score_mode,
        interior_penalty=interior_penalty,
        correction_weight=correction_weight,
        r=r,
        A_flat_by_idx=A_flat_by_idx,
    )
    best_idx = min(all_scores, key=all_scores.get)
    return best_idx, deltas[best_idx], all_scores[best_idx], all_scores, breakdown


# ---------------------------------------------------------------------------
# 7. Corrected reintegration
# ---------------------------------------------------------------------------
@torch.no_grad()
def reintegrate_euler_suffix(
    model: Callable,
    u_start: torch.Tensor,
    ts: torch.Tensor,
    start_idx: int,
) -> torch.Tensor:
    """Apply the exact remaining stored Euler grid ``start_idx,...,N-1``."""
    n_step = len(ts) - 1
    if not 0 <= int(start_idx) <= n_step:
        raise ValueError(f"start_idx must lie in [0,{n_step}], got {start_idx}")
    u = u_start.clone()
    for i in range(int(start_idx), n_step):
        h_i = ts[i + 1] - ts[i]
        u = u + h_i * model(ts[i], u)
    return u.detach()


@torch.no_grad()
def reintegrate_euler_suffix_with_trajectory(
    model: Callable,
    u_start: torch.Tensor,
    ts: torch.Tensor,
    start_idx: int,
) -> List[torch.Tensor]:
    """Apply the stored Euler suffix and retain every corrected state.

    The returned list starts at ``ts[start_idx]`` and ends at ``ts[-1]``.
    It is the path-valued counterpart of :func:`reintegrate_euler_suffix`.
    """
    n_step = len(ts) - 1
    if not 0 <= int(start_idx) <= n_step:
        raise ValueError(f"start_idx must lie in [0,{n_step}], got {start_idx}")
    u = u_start.clone()
    suffix = [u.detach().clone()]
    for i in range(int(start_idx), n_step):
        h_i = ts[i + 1] - ts[i]
        u = u + h_i * model(ts[i], u)
        suffix.append(u.detach().clone())
    return suffix
