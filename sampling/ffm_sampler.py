# Core sampler module to implement PCFM, vanilla flow matching, ECI, DiffusionPDE's guided sample, D-Flow sampling, and MintFlow

import gc
from typing import Optional

import torch
from torchdiffeq import odeint
from tqdm import tqdm

from .adjoint_correction import (
    AdjointWorkspace,
    discrete_euler_adjoint,
    forward_solve_with_trajectory,
    reintegrate_euler_suffix,
    reintegrate_euler_suffix_with_trajectory,
    select_best_candidate,
    select_candidate_indices,
    solve_linearized_correction,
    terminal_residual_and_jacobian,
)
from .pcfm_sampling import make_grid, pcfm_batched


class FFM_sampler:
    """
    Collection of samplers using a pretrained functional flow matching model
    """
    def __init__(self, model, gp):
        self.model = model
        self.gp = gp
        self._mintflow_adjoint_workspace = AdjointWorkspace()

    def pcfm_sample(self, u0, n_step, hfunc, mode='root', newtonsteps=1, eps=1e-6,
                    guided_interpolation=True, interpolation_params={}, use_vmap=False,
                    apply_terminal_projection=True):
        """
        PCFM sampler
        """
        dt = 1.0 / n_step
        u = u0.clone()
        for step_idx, t in enumerate(tqdm(
            torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1],
            desc="PCFM sampling",
        )):
            vf = self.model(t, u)
            if step_idx == n_step - 1 and not apply_terminal_projection:
                v_proj = vf
            else:
                v_proj = pcfm_batched(
                    ut=u, vf=vf, t=t, u0=u0, dt=dt,
                    hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
                    guided_interpolation=guided_interpolation,
                    interpolation_params=interpolation_params,
                    eps=eps,
                    use_vmap=use_vmap
                )
            u = u + dt * v_proj
        return u.detach()

    @torch.no_grad()
    def vanilla_sample(self, u0, n_step):
        """
        Vanilla FFM
        """
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="Vanilla"):
            vf = self.model(t, u)
            u = u + dt * vf
        return u.detach()

    @torch.no_grad()
    def vanilla_sample_with_paths(
        self,
        u0,
        n_step,
        path_fractions: list[float],
        record_indices: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
        """Fixed-step Euler vanilla sample plus flow-time snapshots."""
        dt = 1.0 / n_step
        u = u0.clone()
        if record_indices is None:
            record_mask = torch.ones(u0.shape[0], dtype=torch.bool, device=u0.device)
        else:
            record_mask = torch.zeros(u0.shape[0], dtype=torch.bool, device=u0.device)
            record_mask[record_indices] = True

        snapshots: dict[float, list[torch.Tensor]] = {f: [] for f in path_fractions}
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        for step_idx, t in enumerate(ts[:-1]):
            for frac in path_fractions:
                if step_idx == int(round(float(frac) * n_step)):
                    snapshots[frac].append(u[record_mask].detach().cpu())
            vf = self.model(t, u)
            u = u + dt * vf
        # terminal fraction (tau=1)
        for frac in path_fractions:
            if int(round(float(frac) * n_step)) == n_step:
                snapshots[frac].append(u[record_mask].detach().cpu())
        out = {f: torch.cat(v, dim=0) if v else torch.empty(0) for f, v in snapshots.items()}
        return u.detach(), out

    def pcfm_sample_with_diagnostics(
        self, u0, n_step, hfunc, mode='root', newtonsteps=1, eps=1e-6,
        guided_interpolation=True, interpolation_params={}, use_vmap=False,
        path_fractions: Optional[list[float]] = None,
        record_indices: Optional[list[int]] = None,
        apply_terminal_projection: bool = True,
    ):
        """PCFM sampler returning ``(u, diagnostics, path_snapshots)``."""
        from .pcfm_sampling import pcfm_batched

        dt = 1.0 / n_step
        u = u0.clone()
        if record_indices is None:
            record_mask = torch.ones(u0.shape[0], dtype=torch.bool, device=u0.device)
        else:
            record_mask = torch.zeros(u0.shape[0], dtype=torch.bool, device=u0.device)
            record_mask[record_indices] = True

        path_fractions = path_fractions or []
        path_snaps: dict[float, list[torch.Tensor]] = {f: [] for f in path_fractions}
        per_sample = []
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        for step_idx, t in enumerate(ts[:-1]):
            for frac in path_fractions:
                if step_idx == int(round(float(frac) * n_step)):
                    path_snaps[frac].append(u[record_mask].detach().cpu())
            vf = self.model(t, u)
            if step_idx == n_step - 1 and not apply_terminal_projection:
                v_proj = vf
            else:
                v_proj = pcfm_batched(
                    ut=u, vf=vf, t=t, u0=u0, dt=dt,
                    hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
                    guided_interpolation=guided_interpolation,
                    interpolation_params=interpolation_params,
                    eps=eps,
                    use_vmap=use_vmap,
                )
            u = u + dt * v_proj

        # terminal diagnostics (single-sample batches expected in PCFM orchestration)
        B = u0.shape[0]
        for i in range(B):
            u_flat = u[i].reshape(-1)
            with torch.enable_grad():
                returned_sample = u_flat.detach().clone().requires_grad_(True)
                res_before = hfunc(returned_sample)
                if res_before.ndim == 0:
                    res_before = res_before.unsqueeze(0)
                res_norm_before = float(res_before.norm().item())
                failed = False
                res_after = hfunc(returned_sample.detach())
                if res_after.ndim == 0:
                    res_after = res_after.unsqueeze(0)
                res_norm_after = float(res_after.norm().item())
            per_sample.append({
                "sample_index": i,
                "newton_iterations_estimated": int(
                    (n_step if apply_terminal_projection else n_step - 1) * newtonsteps
                ),
                "num_projection_steps": int(n_step if apply_terminal_projection else n_step - 1),
                "terminal_projection_applied": bool(apply_terminal_projection),
                "projection_residual_before": res_norm_before,
                "projection_residual_after": res_norm_after,
                "projection_residual_sample": "returned_sample",
                "projection_failed": failed,
            })

        for frac in path_fractions:
            if int(round(float(frac) * n_step)) == n_step:
                path_snaps[frac].append(u[record_mask].detach().cpu())
        paths = {f: torch.cat(v, dim=0) if v else torch.empty(0) for f, v in path_snaps.items()}
        return u.detach(), {"pcfm_per_sample": per_sample}, paths

    @torch.no_grad()
    def eci_sample_with_diagnostics(
        self, u0, n_step, n_mix, resample_step, constraint,
        path_fractions: Optional[list[float]] = None,
        record_indices: Optional[list[int]] = None,
        apply_terminal_correction: bool = True,
    ):
        """ECI sampling returning ``(u, diagnostics, path_snapshots)``."""
        u = u0.clone()
        if record_indices is None:
            record_mask = torch.ones(u0.shape[0], dtype=torch.bool, device=u0.device)
        else:
            record_mask = torch.zeros(u0.shape[0], dtype=torch.bool, device=u0.device)
            record_mask[record_indices] = True

        path_fractions = path_fractions or []
        path_snaps: dict[float, list[torch.Tensor]] = {f: [] for f in path_fractions}
        per_sample = [{
            "sample_index": i,
            "adjustment_norm": 0.0,
            "relative_adjustment_norm": 0.0,
            "constraint_names": ["InitialCondition"],
            "num_adjustments": 0,
        } for i in range(u0.shape[0])]

        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        cnt = 0
        dt = 1 / n_step
        grid = make_grid(u.shape[-2:], u.device)
        if resample_step == 0 or resample_step is None:
            resample_step = n_step * n_mix + 1

        for step_idx, t in enumerate(ts[:-1]):
            for frac in path_fractions:
                if step_idx == int(round(float(frac) * n_step)):
                    path_snaps[frac].append(u[record_mask].detach().cpu())
            for mix in range(n_mix):
                cnt += 1
                if cnt % resample_step == 0:
                    u0 = self.gp.sample(grid, u.shape[-2:], n_samples=u.shape[0])
                vf = self.model(t, u)
                u1_pred = u + vf * (1 - t)
                is_terminal_adjustment = step_idx == n_step - 1 and mix == n_mix - 1
                u1_adj = (
                    u1_pred
                    if is_terminal_adjustment and not apply_terminal_correction
                    else constraint.adjust(u1_pred.clone())
                )
                delta = (u1_adj - u1_pred).reshape(u0.shape[0], -1)
                norms = delta.norm(dim=1)
                base = u1_pred.reshape(u0.shape[0], -1).norm(dim=1).clamp_min(1e-12)
                if not (is_terminal_adjustment and not apply_terminal_correction):
                    for i in range(u0.shape[0]):
                        per_sample[i]["num_adjustments"] += 1
                        per_sample[i]["adjustment_norm"] += float(norms[i].item())
                        per_sample[i]["relative_adjustment_norm"] += float((norms[i] / base[i]).item())
                if mix < n_mix - 1:
                    u = u1_adj * t + u0 * (1 - t)
                else:
                    u = u1_adj * (t + dt) + u0 * (1 - t - dt)

        for frac in path_fractions:
            if int(round(float(frac) * n_step)) == n_step:
                path_snaps[frac].append(u[record_mask].detach().cpu())
        paths = {f: torch.cat(v, dim=0) if v else torch.empty(0) for f, v in path_snaps.items()}
        return u.detach(), {"eci_per_sample": per_sample}, paths

    @torch.no_grad()
    def eci_sample(
        self, u0, n_step, n_mix, resample_step, constraint,
        apply_terminal_correction=True,
    ):
        """
        ECI sampling
        """
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        cnt = 0
        dt = 1 / n_step
        grid = make_grid(u.shape[-2:], u.device)
        if resample_step == 0 or resample_step is None:
            resample_step = n_step * n_mix + 1

        for step_idx, t in enumerate(tqdm(ts[:-1], desc='ECI sampling')):
            for mix in range(n_mix):
                cnt += 1
                if cnt % resample_step == 0:
                    u0 = self.gp.sample(grid, u.shape[-2:], n_samples=u.shape[0])
                vf = self.model(t, u)
                u1 = u + vf * (1 - t)
                is_terminal_adjustment = step_idx == n_step - 1 and mix == n_mix - 1
                if not (is_terminal_adjustment and not apply_terminal_correction):
                    u1 = constraint.adjust(u1)
                if mix < n_mix - 1:
                    u = u1 * t + u0 * (1 - t)
                else:
                    u = u1 * (t + dt) + u0 * (1 - t - dt)
        return u.detach()

    def mintflow_sample(
        self, u0, hfunc, forward_steps, num_candidates, ridge, time_penalty,
        jacobian_func,
        large_correction_ratio=2.0, boundary_frac=0.05,
        score_mode: str = "default",
        interior_penalty: float = 0.0,
        correction_weight: float = 1.0,
        candidate_t_min: float = 0.0,
        candidate_t_max: float = 0.98,
        time_sampling: str = "end_biased",
        end_bias_power: float = 2.0,
        correction_mode: str = "pseudoinverse",
        correction_scale: float = 1.0,
        pseudoinverse_rcond: Optional[float] = None,
        adjoint_chunk_size: int = 64,
        save_mathematical_diagnostics: bool = False,
        path_fractions: Optional[list[float]] = None,
    ):
        """
        MintFlow: Adjoint-Based Optimal Correction.

        Single-sample sampler (``u0`` shape ``(1, *dims)``): forward-solves the
        flow ODE, computes the terminal constraint residual/Jacobian, solves the
        adjoint equation backward along the trajectory, selects the best
        correction time/perturbation, and re-integrates the original flow from
        that point.  See ``sampling/adjoint_correction.py`` for the underlying
        math and ``Constrained_FM.pdf`` for the derivation.

        Numerical safety: model parameters are frozen during the adjoint solve
        (see :func:`~sampling.adjoint_correction.frozen_parameters`); NaN/Inf
        are checked at every stage (residual, Jacobian, adjoint, per-candidate
        correction, corrected sample) and reported via ``diagnostics["warnings"]``
        rather than silently propagating.

        Returns
        -------
        u_T_corrected : ``(1, *dims)`` corrected terminal sample.
        diagnostics   : dict — see inline keys below, including
                        ``candidate_scores`` (``{idx: score}`` for every
                        candidate) and a ``warnings`` list of any numerical
                        safety flags raised for this sample.
        """
        warnings: list[str] = []
        if adjoint_chunk_size <= 0:
            raise ValueError("adjoint_chunk_size must be positive")
        if not 0.0 <= correction_scale <= 1.0:
            raise ValueError("correction_scale must lie in [0, 1]")

        traj, ts = forward_solve_with_trajectory(self.model, u0, forward_steps)
        spatial_shape = tuple(u0.shape[1:])

        u_T_flat = traj[-1].reshape(-1)
        with torch.enable_grad():
            r, J, jac_info = terminal_residual_and_jacobian(
                hfunc, u_T_flat, jacobian_func=jacobian_func,
            )
        if jac_info["nan_in_residual"]:
            warnings.append("nan_in_residual_before")
        if jac_info["nan_in_jacobian"]:
            warnings.append("nan_in_jacobian")
        m = J.shape[0]
        A_T = J.reshape(m, *spatial_shape)

        candidate_idxs = select_candidate_indices(
            forward_steps, num_candidates, t_min=candidate_t_min, t_max=candidate_t_max,
            time_sampling=time_sampling, end_bias_power=end_bias_power,
        )

        with torch.enable_grad():
            lambdas, adjoint_nan = discrete_euler_adjoint(
                self.model, traj, ts, A_T, candidate_idxs,
                batch_chunk_size=adjoint_chunk_size,
                workspace=self._mintflow_adjoint_workspace,
            )
        if adjoint_nan:
            warnings.append("nan_in_adjoint")

        deltas = {}
        A_flat_by_idx = {}
        cand_info = {}
        for idx, A_k in lambdas.items():
            A_k_flat = A_k.reshape(m, -1)
            A_flat_by_idx[idx] = A_k_flat
            delta_flat, info = solve_linearized_correction(
                A_k_flat, r, mode=correction_mode, ridge=ridge,
                rcond=pseudoinverse_rcond,
            )
            info["unscaled_correction_norm"] = float(delta_flat.norm().item())
            info["correction_scale"] = float(correction_scale)
            delta_flat = delta_flat * correction_scale
            deltas[idx] = delta_flat.reshape(*spatial_shape)
            cand_info[idx] = info
        if any(info["solve_failed"] for info in cand_info.values()):
            warnings.append("linalg_solve_failed")
        if any(info["ill_conditioned"] for info in cand_info.values()):
            warnings.append("ill_conditioned_AAt")
        if any(info["nan_in_delta"] for info in cand_info.values()):
            warnings.append("nan_in_delta")

        best_idx, best_delta, best_score, all_scores, score_breakdown = select_best_candidate(
            deltas,
            ts,
            T=1.0,
            time_penalty=time_penalty,
            score_mode=score_mode,
            interior_penalty=interior_penalty,
            correction_weight=correction_weight,
            r=r,
            A_flat_by_idx=A_flat_by_idx if score_mode == "residual_predictive" else None,
        )
        s_star = ts[best_idx].item()

        candidate_times = {str(idx): float(ts[idx].item()) for idx in sorted(deltas.keys())}
        candidate_correction_norms = {
            str(idx): float(deltas[idx].norm().item()) for idx in sorted(deltas.keys())
        }
        # Solver-specific sensitivity condition number, surfaced for diagnostics.
        candidate_condition_numbers = {
            str(idx): cand_info[idx]["cond"] for idx in sorted(cand_info.keys())
        }
        candidate_adjoint_norms = {
            str(idx): float(A_flat_by_idx[idx].norm().item()) for idx in sorted(A_flat_by_idx.keys())
        }
        candidate_predicted_residual_norms = {
            str(idx): float((r + A_flat_by_idx[idx] @ deltas[idx].reshape(-1)).norm().item())
            for idx in sorted(deltas.keys())
        }
        candidate_score_breakdown = {
            str(idx): {k: float(v) for k, v in score_breakdown[idx].items()}
            for idx in sorted(score_breakdown.keys())
        }

        candidate_realized_residual_norms = {}
        candidate_realized_residual_infs = {}
        candidate_diagnostics = []
        realized_candidates = {}
        if save_mathematical_diagnostics:
            for idx, delta in deltas.items():
                realized = reintegrate_euler_suffix(
                    self.model, traj[idx] + delta.unsqueeze(0), ts, idx,
                )
                realized_candidates[idx] = realized
                with torch.no_grad():
                    realized_r = hfunc(realized.reshape(-1))
                    if realized_r.ndim == 0:
                        realized_r = realized_r.unsqueeze(0)
                candidate_realized_residual_norms[str(idx)] = float(realized_r.norm().item())
                candidate_realized_residual_infs[str(idx)] = float(realized_r.abs().max().item())
                predicted_r = r + A_flat_by_idx[idx] @ delta.reshape(-1)
                candidate_diagnostics.append({
                    "candidate_index": int(idx),
                    "candidate_time": float(ts[idx].item()),
                    "correction_norm": float(delta.norm().item()),
                    "residual_before": r.detach().cpu().tolist(),
                    "predicted_linearized_residual_after": predicted_r.detach().cpu().tolist(),
                    "realized_residual_after": realized_r.detach().cpu().tolist(),
                    "score": float(all_scores[idx]),
                })

        # Boundary check: did selection collapse to the edge of the candidate set?
        sorted_idxs = sorted(deltas.keys())
        if len(sorted_idxs) > 1 and best_idx in (sorted_idxs[0], sorted_idxs[-1]):
            span = ts[sorted_idxs[-1]].item() - ts[sorted_idxs[0]].item()
            edge_dist = min(abs(s_star - ts[sorted_idxs[0]].item()), abs(s_star - ts[sorted_idxs[-1]].item()))
            if span <= 0 or edge_dist <= boundary_frac * span:
                warnings.append("selected_time_near_boundary")

        u_corrected_start = (traj[best_idx] + best_delta.unsqueeze(0)).detach()

        # Large-correction check: |delta| relative to the state norm it perturbs.
        state_norm = float(traj[best_idx].norm().item())
        delta_norm = float(best_delta.norm().item())
        if state_norm > 0 and (delta_norm / state_norm) > large_correction_ratio:
            warnings.append("large_correction_norm")

        corrected_path_snapshots = None
        if path_fractions:
            corrected_suffix = reintegrate_euler_suffix_with_trajectory(
                self.model, u_corrected_start, ts, best_idx,
            )
            u_T_corrected = corrected_suffix[-1]
            corrected_path_snapshots = []
            for frac in path_fractions:
                idx = min(max(int(round(float(frac) * forward_steps)), 0), forward_steps)
                state = traj[idx] if idx < best_idx else corrected_suffix[idx - best_idx]
                corrected_path_snapshots.append(state.squeeze(0).detach().cpu().numpy())
        else:
            u_T_corrected = realized_candidates.get(best_idx)
            if u_T_corrected is None:
                u_T_corrected = reintegrate_euler_suffix(
                    self.model, u_corrected_start, ts, best_idx,
                )
        if not torch.isfinite(u_T_corrected).all():
            warnings.append("nan_in_corrected_sample")

        with torch.no_grad():
            r_after = hfunc(u_T_corrected.reshape(-1))
            if r_after.ndim == 0:
                r_after = r_after.unsqueeze(0)
        if not torch.isfinite(r_after).all():
            warnings.append("nan_in_residual_after")

        linearization_convergence = []
        zero_restart_relative_field_error = float("nan")
        zero_restart_residual_vector_error = float("nan")
        zero_restart_R_inf_difference = float("nan")
        if save_mathematical_diagnostics:
            zero_restart = reintegrate_euler_suffix(
                self.model, traj[best_idx], ts, best_idx,
            )
            with torch.no_grad():
                zero_restart_r = hfunc(zero_restart.reshape(-1))
                if zero_restart_r.ndim == 0:
                    zero_restart_r = zero_restart_r.unsqueeze(0)
            terminal_scale = max(float(traj[-1].norm().item()), 1e-30)
            zero_restart_relative_field_error = float(
                (zero_restart - traj[-1]).norm().item() / terminal_scale
            )
            zero_restart_residual_vector_error = float((zero_restart_r - r).norm().item())
            zero_restart_R_inf_difference = float(
                abs(zero_restart_r.abs().max().item() - r.abs().max().item())
            )
            selected_A = A_flat_by_idx[best_idx]
            selected_delta_flat = best_delta.reshape(-1)
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                predicted_r = r + alpha * (selected_A @ selected_delta_flat)
                realized = reintegrate_euler_suffix(
                    self.model,
                    traj[best_idx] + alpha * best_delta.unsqueeze(0),
                    ts,
                    best_idx,
                )
                with torch.no_grad():
                    realized_r = hfunc(realized.reshape(-1))
                    if realized_r.ndim == 0:
                        realized_r = realized_r.unsqueeze(0)
                linearization_convergence.append({
                    "alpha": alpha,
                    "delta_norm": float((alpha * selected_delta_flat).norm().item()),
                    "predicted_residual_norm": float(predicted_r.norm().item()),
                    "predicted_residual_inf": float(predicted_r.abs().max().item()),
                    "realized_residual_norm": float(realized_r.norm().item()),
                    "realized_residual_inf": float(realized_r.abs().max().item()),
                    "prediction_error_norm": float((realized_r - predicted_r).norm().item()),
                })

        diagnostics = {
            "s_star":                 s_star,
            "best_idx":               int(best_idx),
            "num_candidates":         len(deltas),
            "residual_norm_before":   float(r.norm().item()),
            "residual_norm_after":    float(r_after.norm().item()),
            "terminal_residual_before": float(r.norm().item()),
            "terminal_residual_after":  float(r_after.norm().item()),
            "terminal_residual_inf_before": float(r.abs().max().item()),
            "terminal_residual_inf_after": float(r_after.abs().max().item()),
            "delta_norm":             delta_norm,
            "correction_norm":        delta_norm,
            "score":                  float(best_score),
            "candidate_scores":       all_scores,
            "candidate_times":        candidate_times,
            "candidate_correction_norms": candidate_correction_norms,
            "candidate_condition_numbers": candidate_condition_numbers,
            "candidate_adjoint_norms": candidate_adjoint_norms,
            "candidate_predicted_residual_norms": candidate_predicted_residual_norms,
            "candidate_realized_residual_norms": candidate_realized_residual_norms,
            "candidate_realized_residual_infs": candidate_realized_residual_infs,
            "candidate_diagnostics": candidate_diagnostics,
            "candidate_score_breakdown": candidate_score_breakdown,
            "linearization_convergence": linearization_convergence,
            "zero_restart_relative_field_error": zero_restart_relative_field_error,
            "zero_restart_residual_vector_error": zero_restart_residual_vector_error,
            "zero_restart_R_inf_difference": zero_restart_R_inf_difference,
            "terminal_jacobian_frobenius_norm": float(J.norm().item()),
            "terminal_jacobian_backend": jac_info["jacobian_backend"],
            "selected_adjoint_norm": candidate_adjoint_norms[str(best_idx)],
            "selected_predicted_residual_norm": candidate_predicted_residual_norms[str(best_idx)],
            "selected_predicted_residual_inf": float(
                (r + A_flat_by_idx[best_idx] @ best_delta.reshape(-1)).abs().max().item()
            ),
            "line_search_used": False,
            "adjoint_mode": "discrete_euler_exact",
            "suffix_reintegration": "original_remaining_euler_grid",
            "correction_mode": correction_mode,
            "correction_scale": float(correction_scale),
            "damping_ridge": float(ridge) if correction_mode == "damped" else 0.0,
            "pseudoinverse_rcond": pseudoinverse_rcond,
            "adjoint_chunk_size": int(adjoint_chunk_size),
            "mintflow_score_mode":         score_mode,
            "mintflow_time_penalty":       float(time_penalty),
            "mintflow_interior_penalty":   float(interior_penalty),
            "mintflow_correction_weight":  float(correction_weight),
            "mintflow_candidate_t_min":    float(candidate_t_min),
            "mintflow_candidate_t_max":    float(candidate_t_max),
            "mintflow_time_sampling":      time_sampling,
            "mintflow_end_bias_power":     float(end_bias_power),
            "warnings":               warnings,
            # Removed by sampling.methods before JSON serialization and
            # written to flow_paths.npz when path recording is enabled.
            "_corrected_path_snapshots": corrected_path_snapshots,
        }
        return u_T_corrected.detach(), diagnostics

    def guided_sample(self, u0, u1_true, mask, n_step, loss_fn, eta=2e2):
        """
        DiffusionPDE: takes an IC and PINN loss (if known) on the extrapolated sample and updates the vector field
        """
        device = u0.device
        u = u0.clone().to(device)
        u1_true = u1_true.to(device)
        mask = mask.to(device)
        ts = torch.linspace(0, 1, n_step + 1, device=device)

        for t in tqdm(ts[:-1], desc='DiffusionPDE sampling'):
            vf = self.model(t, u).detach()
            if t < ts[-2]:
                vf2 = self.model(t + 1 / n_step, u).detach()
                vf = (vf + vf2) / 2

            u.requires_grad_(True)
            u1_pred = u + vf * (1 - t)
            loss = loss_fn(u1_pred, u1_true, mask)
            loss.backward()
            grad = u.grad
            u = u.detach() + vf / n_step - eta * grad
        return u.detach()

    #
    def dflow_sample(self, u1_true, mask, n_sample, n_step, n_iter=20, lr=1e-1, loss_fn=None):
        """
        D-Flow: optimizes the noise by differentiating through the flow matching ODE steps
        """
        device = u1_true.device
        mask = mask.to(device)
        grid = make_grid(u1_true.size()[1:], device)

        noise = self.gp.sample(grid, u1_true.size()[1:], n_samples=n_sample).to(device)
        noise.requires_grad_(True)

        ts = torch.linspace(0, 1, n_step + 1, device=device)

        def default_loss_fn(u_pred, u_true, mask):
            return ((u_pred - u_true) * mask).square().sum()
        loss_fn = loss_fn or default_loss_fn

        def euler_ffm(u):
            print("DFlow sampling...")
            tspan = torch.tensor([0, 1.], device=device)
            u = odeint(self.model, u, tspan, method="euler", options = {"step_size":ts[1]-ts[0]})[-1]
            return u

        def closure():
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            u_pred = euler_ffm(noise)
            loss = loss_fn(u_pred, u1_true, mask)
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS([noise], max_iter=n_iter, lr=lr)
        optimizer.step(closure)

        with torch.no_grad():
            u_final = euler_ffm(noise)
        return u_final.detach()
