# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
"""Production sampling wrappers for Heat and RD1D constrained FFM.

Every constrained method consumes the same ``sampling.tasks.ConstraintTask``
(or lazy paired provider), ensuring Scenario 1 and Scenario 2 use identical
constraint definitions across PCFM, ECI, MintFlow, guided baselines, and final
projection. All methods consume the same production constraint-task interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from sampling.flow_paths import merge_path_arrays, save_flow_paths_npz
from sampling.noise import shared_noise_metadata_fields, slice_initial_noise

if TYPE_CHECKING:  # avoid a hard circular import at module load time
    from sampling.tasks import ConstraintTask


def _task_for_sample(task, sample_index: int):
    """Resolve a shared task or one lazily paired task."""
    return task.task_for(sample_index) if hasattr(task, "task_for") else task


def _task_batch(task, start: int, count: int):
    if hasattr(task, "task_for"):
        return [task.task_for(start + index) for index in range(count)]
    return task


def _representative_task(task):
    if hasattr(task, "task_for"):
        return task.task_for(0)
    if isinstance(task, (list, tuple)):
        return task[0]
    return task


def _record_indices(batch_start: int, batch_size: int, max_path_samples: int) -> Optional[list[int]]:
    """Indices within a batch whose flow paths should be recorded."""
    if max_path_samples <= 0:
        return None
    indices = []
    for i in range(batch_size):
        global_idx = batch_start + i
        if global_idx < max_path_samples:
            indices.append(i)
    return indices or None


def _finalize_flow_paths(
    run_dir: Optional[str],
    path_chunks: list[np.ndarray],
    path_indices: list[int],
    fractions: list[float],
) -> Optional[str]:
    if not path_chunks or run_dir is None:
        return None
    path_samples = merge_path_arrays(path_chunks)
    out = f"{run_dir}/flow_paths.npz" if not str(run_dir).endswith(".npz") else str(run_dir)
    save_flow_paths_npz(
        out,
        path_samples=path_samples,
        path_sample_indices=np.asarray(path_indices[: path_samples.shape[0]], dtype=np.int64),
        flow_path_fractions=fractions,
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_vanilla_sampling(
    ffm,
    config,
    num_samples: int,
    n_eval: int,
    batch_size: int,
    device: str,
    seed: int,
    initial_noise: Optional[torch.Tensor] = None,
    initial_noise_path: Optional[str] = None,
    save_flow_paths: bool = False,
    flow_path_fractions: Optional[list[float]] = None,
    max_path_samples: int = 32,
    run_dir: Optional[str] = None,
) -> tuple[torch.Tensor, dict]:
    """Vanilla FFM: dopri5 ODE via generate_samples_batched, or fixed-step Euler
    when ``save_flow_paths`` is enabled (paths require Euler snapshots)."""
    from sampling.ffm_sampler import FFM_sampler
    from visualization.sampling import generate_samples_batched

    meta = shared_noise_metadata_fields(initial_noise_path, initial_noise, seed)
    flow_path_fractions = flow_path_fractions or [0.0, 0.25, 0.5, 0.75, 1.0]
    path_chunks: list[np.ndarray] = []
    path_indices: list[int] = []

    if save_flow_paths and initial_noise is not None:
        sampler = FFM_sampler(model=ffm.model, gp=ffm.gp)
        all_samples: list[torch.Tensor] = []
        n_generated = 0
        while n_generated < num_samples:
            B = min(batch_size, num_samples - n_generated)
            u0_batch = slice_initial_noise(initial_noise, n_generated, n_generated + B, device)
            rec = _record_indices(n_generated, B, max_path_samples)
            u_out, snaps = sampler.vanilla_sample_with_paths(
                u0_batch, n_step=n_eval, path_fractions=flow_path_fractions, record_indices=rec,
            )
            all_samples.append(u_out.cpu())
            if rec is not None:
                idxs = [n_generated + i for i in rec]
                path_indices.extend(idxs)
                chunk = np.stack([snaps[f].numpy() for f in flow_path_fractions], axis=1)
                path_chunks.append(chunk)
            n_generated += B
            print(f"  Vanilla: {n_generated}/{num_samples} samples generated")
        samples = torch.cat(all_samples, dim=0)
        meta["flow_paths_file"] = _finalize_flow_paths(run_dir, path_chunks, path_indices, flow_path_fractions)
        meta["flow_path_fractions"] = flow_path_fractions
        meta["integrator_for_paths"] = f"euler_n_step={n_eval}"
    else:
        samples = generate_samples_batched(
            ffm, config,
            num_samples=num_samples,
            batch_size=batch_size,
            n_eval=n_eval,
            device=device,
            seed=seed,
            initial_noise=initial_noise,
        )

    meta.update({
        "sampler": "FFM.sample (torchdiffeq dopri5)" if not save_flow_paths else f"FFM_sampler.vanilla_sample (Euler n_step={n_eval})",
        "n_eval": n_eval,
    })
    return samples, meta


def _batched_constraint_residual(samples: torch.Tensor, task) -> torch.Tensor:
    """Stack normalized task residuals without assuming ``vmap`` compatibility."""
    tasks = task if isinstance(task, (list, tuple)) else None
    if tasks is not None and len(tasks) != len(samples):
        raise ValueError("Paired task batch length does not match sample batch length")
    return torch.stack([
        (tasks[index] if tasks is not None else task).normalized_residual(
            sample.reshape(-1)
        )
        for index, sample in enumerate(samples)
    ])


def run_diffusionpde_sampling(
    ffm, config, num_samples: int, n_step: int, eta: float,
    batch_size: int, device: str, seed: int,
    initial_noise: Optional[torch.Tensor] = None,
    initial_noise_path: Optional[str] = None,
    save_flow_paths: bool = False,
    flow_path_fractions: Optional[list[float]] = None,
    max_path_samples: int = 32,
    run_dir: Optional[str] = None,
    task: Optional["ConstraintTask"] = None,
) -> tuple[torch.Tensor, dict]:
    """DiffusionPDE-style endpoint guidance adapted to a pretrained FFM.

    At Euler time ``t_k``, the frozen velocity predicts the clean endpoint
    ``u_hat_1 = u_k + (1-t_k) v_theta(t_k,u_k)``. The state then takes its
    Euler flow step plus a negative constraint-loss gradient step. This is
    the adaptation shipped in the official PCFM code, with the production
    global ``ConstraintTask`` replacing the paper's IC/BC+PINN composite.
    """
    from models.functional import make_grid

    if task is None:
        raise ValueError("DiffusionPDE baseline requires a production ConstraintTask")
    if n_step < 1 or eta < 0:
        raise ValueError("n_step must be positive and eta must be non-negative")
    dims = list(config.sample_dims)
    grid = make_grid(dims, device)
    if initial_noise is None:
        torch.manual_seed(seed)
    flow_path_fractions = flow_path_fractions or [0.0, 0.25, 0.5, 0.75, 1.0]
    path_chunks: list[np.ndarray] = []
    path_indices: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    outputs: list[torch.Tensor] = []
    n_generated = 0
    dt = 1.0 / n_step
    ts = torch.linspace(0.0, 1.0, n_step + 1, device=device)

    while n_generated < num_samples:
        B = min(batch_size, num_samples - n_generated)
        batch_task = _task_batch(task, n_generated, B)
        u = (slice_initial_noise(initial_noise, n_generated, n_generated + B, device)
             if initial_noise is not None else ffm.gp.sample(grid, dims, n_samples=B).to(device))
        rec = _record_indices(n_generated, B, max_path_samples) if save_flow_paths else None
        record_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if rec:
            record_mask[rec] = True
        snapshots: dict[float, list[torch.Tensor]] = {f: [] for f in flow_path_fractions}
        with torch.no_grad():
            initial_residual = _batched_constraint_residual(
                u, batch_task
            ).abs().amax(dim=1)

        for step_index, t in enumerate(ts[:-1]):
            if rec:
                for fraction in flow_path_fractions:
                    if step_index == int(round(float(fraction) * n_step)):
                        snapshots[fraction].append(u[record_mask].detach().cpu())
            with torch.no_grad():
                velocity = ffm.model(t, u)
                if step_index < n_step - 1:
                    velocity = 0.5 * (velocity + ffm.model(ts[step_index + 1], u))
            state = u.detach().requires_grad_(True)
            predicted_terminal = state + (1.0 - t) * velocity
            residual = _batched_constraint_residual(predicted_terminal, batch_task)
            # Mean over residual components matches DiffusionPDE's averaged
            # guidance losses and keeps eta invariant to task output_dim.
            loss = 0.5 * residual.square().mean(dim=1).sum()
            gradient, = torch.autograd.grad(loss, state)
            u = (state + dt * velocity - eta * gradient).detach()

        if rec:
            for fraction in flow_path_fractions:
                if int(round(float(fraction) * n_step)) == n_step:
                    snapshots[fraction].append(u[record_mask].detach().cpu())
            path_chunks.append(np.stack([
                torch.cat(snapshots[fraction], dim=0).numpy()
                for fraction in flow_path_fractions
            ], axis=1))
            path_indices.extend([n_generated + index for index in rec])
        with torch.no_grad():
            final_residual = _batched_constraint_residual(
                u, batch_task
            ).abs().amax(dim=1)
        for index in range(B):
            diagnostics.append({
                "sample_index": n_generated + index,
                "initial_normalized_R_inf": float(initial_residual[index].item()),
                "final_normalized_R_inf": float(final_residual[index].item()),
                "finite_output": bool(torch.isfinite(u[index]).all().item()),
            })
        outputs.append(u.cpu())
        n_generated += B
        print(f"  DiffusionPDE: {n_generated}/{num_samples} samples generated")

    metadata = {
        "sampler": "DiffusionPDE-style FFM endpoint guidance (fixed-step Euler)",
        "adaptation": "soft guidance uses the production global ConstraintTask residual",
        "n_step": n_step, "eta": float(eta),
        "loss": "sum_samples(0.5 * mean_components(normalized residual ** 2))",
        "terminal_intervention": False,
        "diffusionpde_per_sample": diagnostics,
        **shared_noise_metadata_fields(initial_noise_path, initial_noise, seed),
        **_task_metadata(task),
    }
    if save_flow_paths:
        metadata["flow_paths_file"] = _finalize_flow_paths(
            run_dir, path_chunks, path_indices, flow_path_fractions)
        metadata["flow_path_fractions"] = flow_path_fractions
    return torch.cat(outputs, dim=0), metadata


def run_dflow_sampling(
    ffm, config, num_samples: int, n_step: int, n_iter: int, lr: float,
    batch_size: int, device: str, seed: int,
    initial_noise: Optional[torch.Tensor] = None,
    initial_noise_path: Optional[str] = None,
    save_flow_paths: bool = False,
    flow_path_fractions: Optional[list[float]] = None,
    max_path_samples: int = 32,
    run_dir: Optional[str] = None,
    task: Optional["ConstraintTask"] = None,
) -> tuple[torch.Tensor, dict]:
    """D-Flow source optimization with LBFGS and an adjoint Euler solve."""
    from torchdiffeq import odeint_adjoint

    from models.functional import make_grid
    from sampling.ffm_sampler import FFM_sampler

    if task is None:
        raise ValueError("D-Flow baseline requires a production ConstraintTask")
    if n_step < 1 or n_iter < 1 or lr <= 0:
        raise ValueError("n_step/n_iter must be positive and lr must be > 0")
    dims = list(config.sample_dims)
    grid = make_grid(dims, device)
    sampler = FFM_sampler(model=ffm.model, gp=ffm.gp)
    if initial_noise is None:
        torch.manual_seed(seed)
    flow_path_fractions = flow_path_fractions or [0.0, 0.25, 0.5, 0.75, 1.0]
    path_chunks: list[np.ndarray] = []
    path_indices: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    outputs: list[torch.Tensor] = []
    n_generated = 0
    dt = 1.0 / n_step
    tspan = torch.tensor([0.0, 1.0], device=device)

    def differentiable_flow(source: torch.Tensor) -> torch.Tensor:
        return odeint_adjoint(
            ffm.model, source, tspan, method="euler", options={"step_size": dt},
            adjoint_method="euler", adjoint_options={"step_size": dt}, adjoint_params=(),
        )[-1]

    while n_generated < num_samples:
        B = min(batch_size, num_samples - n_generated)
        batch_task = _task_batch(task, n_generated, B)
        original_source = (slice_initial_noise(initial_noise, n_generated, n_generated + B, device)
                           if initial_noise is not None else ffm.gp.sample(grid, dims, n_samples=B).to(device))
        source = original_source.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([source], max_iter=n_iter, lr=lr)
        closure_evaluations = 0

        def closure():
            nonlocal closure_evaluations
            optimizer.zero_grad(set_to_none=True)
            terminal = differentiable_flow(source)
            residual = _batched_constraint_residual(terminal, batch_task)
            # D-Flow/PCFM's reference implementation uses a summed terminal
            # constraint loss; retain that scaling for the published lr=1.
            loss = 0.5 * residual.square().sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("D-Flow constraint objective became non-finite")
            loss.backward()
            closure_evaluations += 1
            return loss

        optimizer.step(closure)
        rec = _record_indices(n_generated, B, max_path_samples) if save_flow_paths else None
        final, snapshots = sampler.vanilla_sample_with_paths(
            source.detach(), n_step=n_step, path_fractions=flow_path_fractions,
            record_indices=rec)
        if rec:
            path_chunks.append(np.stack([
                snapshots[fraction].numpy() for fraction in flow_path_fractions
            ], axis=1))
            path_indices.extend([n_generated + index for index in rec])
        with torch.no_grad():
            residual = _batched_constraint_residual(final, batch_task)
            per_sample_loss = 0.5 * residual.square().sum(dim=1)
            source_shift = (source.detach() - original_source).reshape(B, -1).norm(dim=1)
            source_norm = original_source.reshape(B, -1).norm(dim=1).clamp_min(1e-30)
        for index in range(B):
            diagnostics.append({
                "sample_index": n_generated + index,
                "final_normalized_R_inf": float(residual[index].abs().max().item()),
                "final_constraint_loss": float(per_sample_loss[index].item()),
                "source_relative_displacement": float((source_shift[index] / source_norm[index]).item()),
                "closure_evaluations": closure_evaluations,
                "finite_output": bool(torch.isfinite(final[index]).all().item()),
            })
        outputs.append(final.detach().cpu())
        n_generated += B
        print(f"  D-Flow: {n_generated}/{num_samples} samples generated")

    metadata = {
        "sampler": "D-Flow source optimization (LBFGS + torchdiffeq Euler adjoint)",
        "adaptation": "terminal objective uses the production global ConstraintTask residual",
        "n_step": n_step, "n_iter": n_iter, "lr": float(lr),
        "optimizer": "torch.optim.LBFGS",
        "loss": "0.5 * sum(normalized terminal ConstraintTask residual ** 2)",
        "terminal_intervention": False,
        "dflow_per_sample": diagnostics,
        **shared_noise_metadata_fields(initial_noise_path, initial_noise, seed),
        **_task_metadata(task),
    }
    if save_flow_paths:
        metadata["flow_paths_file"] = _finalize_flow_paths(
            run_dir, path_chunks, path_indices, flow_path_fractions)
        metadata["flow_path_fractions"] = flow_path_fractions
    return torch.cat(outputs, dim=0), metadata


def run_pcfm_sampling(
    ffm,
    config,
    num_samples: int,
    n_step: int,
    newton_steps: int,
    tol: float,
    batch_size: int,
    device: str,
    seed: int,
    initial_noise: Optional[torch.Tensor] = None,
    initial_noise_path: Optional[str] = None,
    save_flow_paths: bool = False,
    flow_path_fractions: Optional[list[float]] = None,
    max_path_samples: int = 32,
    save_sampler_diagnostics: bool = False,
    run_dir: Optional[str] = None,
    task: Optional["ConstraintTask"] = None,
    intermediate_only: bool = False,
) -> tuple[torch.Tensor, dict]:
    """PCFM: Physics-Constrained Flow Matching (Euler + Newton projection).

    Each generated sample is projected onto the constraint manifold at every
    Euler step via :func:`~sampling.pcfm_sampling.pcfm_batched`. With
    ``intermediate_only=True``, all projections are retained except the final
    one; the last Euler update uses the unmodified vector field.

    A shared Scenario-1 task or paired Scenario-2 task provider supplies the
    normalized residual. Samples are projected individually while model noise
    is still drawn in batches.
    """
    from models.functional import make_grid
    from sampling.ffm_sampler import FFM_sampler

    sampler  = FFM_sampler(model=ffm.model, gp=ffm.gp)
    dims     = list(config.sample_dims)
    grid     = make_grid(dims, device)       # reused across all batches
    if task is None:
        raise ValueError("PCFM requires a production ConstraintTask")

    if initial_noise is None:
        torch.manual_seed(seed)
    flow_path_fractions = flow_path_fractions or [0.0, 0.25, 0.5, 0.75, 1.0]
    path_chunks: list[np.ndarray] = []
    path_indices: list[int] = []
    pcfm_diagnostics: list[dict] = []
    all_samples: list[torch.Tensor] = []
    n_generated = 0

    while n_generated < num_samples:
        B = min(batch_size, num_samples - n_generated)

        if initial_noise is not None:
            u0_batch = slice_initial_noise(initial_noise, n_generated, n_generated + B, device)
        else:
            u0_batch = ffm.gp.sample(grid, dims, n_samples=B).to(device)

        batch_results: list[torch.Tensor] = []
        rec = _record_indices(n_generated, B, max_path_samples) if save_flow_paths else None
        batch_paths: list[np.ndarray] = []

        for i in range(B):
            sample_idx = n_generated + i

            this_task = _task_for_sample(task, sample_idx)
            # Scenario 2 normalizes heterogeneous residual components; unit
            # Scenario-1 scales preserve the conservation-only formulation.
            hfunc = this_task.normalized_residual

            if save_sampler_diagnostics or save_flow_paths:
                rec_i = [0] if (rec is not None and i in rec) else None
                result, diag, paths = sampler.pcfm_sample_with_diagnostics(
                    u0=u0_batch[i : i + 1],
                    n_step=n_step,
                    hfunc=hfunc,
                    mode="root",
                    newtonsteps=newton_steps,
                    eps=tol,
                    guided_interpolation=False,
                    use_vmap=False,
                    path_fractions=flow_path_fractions if rec_i else None,
                    record_indices=rec_i,
                    apply_terminal_projection=not intermediate_only,
                )
                if save_sampler_diagnostics and diag.get("pcfm_per_sample"):
                    row = dict(diag["pcfm_per_sample"][0])
                    row["sample_index"] = sample_idx
                    pcfm_diagnostics.append(row)
                if rec_i and paths:
                    batch_paths.append(np.stack([paths[f].numpy() for f in flow_path_fractions], axis=1))
            else:
                result = sampler.pcfm_sample(
                    u0=u0_batch[i : i + 1],
                    n_step=n_step,
                    hfunc=hfunc,
                    mode="root",
                    newtonsteps=newton_steps,
                    eps=tol,
                    guided_interpolation=False,
                    use_vmap=False,
                    apply_terminal_projection=not intermediate_only,
                )
            batch_results.append(result.detach().cpu())

        if rec is not None and batch_paths:
            path_chunks.extend(batch_paths)
            path_indices.extend([n_generated + j for j in rec])

        n_generated += B
        all_samples.append(torch.cat(batch_results, dim=0))
        print(f"  PCFM: {n_generated}/{num_samples} samples generated")

    meta = {
        "sampler":         "FFM_sampler.pcfm_sample (Euler + Newton projection)",
        "n_step":          n_step,
        "newton_steps":    newton_steps,
        "tol":             tol,
        "guided_interp":   False,
        "intermediate_only": bool(intermediate_only),
        "intrinsic_terminal_intervention_applied": not intermediate_only,
        "intrinsic_projection_steps": n_step - int(intermediate_only),
        "hfunc":           f"ConstraintTask.{task.name}",
        **shared_noise_metadata_fields(initial_noise_path, initial_noise, seed),
    }
    meta.update(_task_metadata(task))
    if save_sampler_diagnostics:
        meta["pcfm_per_sample"] = pcfm_diagnostics
    if save_flow_paths:
        meta["flow_paths_file"] = _finalize_flow_paths(run_dir, path_chunks, path_indices, flow_path_fractions)
        meta["flow_path_fractions"] = flow_path_fractions
    return torch.cat(all_samples, dim=0), meta


def run_eci_sampling(
    ffm,
    config,
    num_samples: int,
    n_step: int,
    n_mix: int,
    resample_every: int,
    batch_size: int,
    device: str,
    seed: int,
    initial_noise: Optional[torch.Tensor] = None,
    initial_noise_path: Optional[str] = None,
    save_flow_paths: bool = False,
    flow_path_fractions: Optional[list[float]] = None,
    max_path_samples: int = 32,
    save_sampler_diagnostics: bool = False,
    run_dir: Optional[str] = None,
    task: Optional["ConstraintTask"] = None,
    eci_mode: str = "native",
    projector_config=None,
    intermediate_only: bool = False,
) -> tuple[torch.Tensor, dict]:
    """ECI: Energy-Constrained Inference (Euler + correction).

    The endpoint correction projects ``u1`` onto the shared Scenario-1 task
    or the aligned per-sample Scenario-2 task. Two modes are retained:

    * ``eci_mode="native"`` — uses ``task.project()`` (native closed-form
      projector when available, else generic Gauss-Newton). "ECI-native".
    * ``eci_mode="gn"``     — always uses the generic damped Gauss-Newton
      projector, even if a native one exists (fair ablation). "ECI-GN".

    In conservation-only runs, all samples share one global task. In
    Scenario 2, each batch receives an aligned list of per-sample tasks from
    the paired target bank. Accumulated per-call diagnostics are surfaced in
    the returned metadata when ``save_sampler_diagnostics=True``.
    With ``intermediate_only=True``, only the final adjustment that directly
    produces the returned sample is skipped. Earlier mixes and pullbacks,
    including earlier mixes at the final flow-time step, remain unchanged.
    """
    from models.constraints import TaskProjectionConstraint
    from models.functional import make_grid
    from sampling.ffm_sampler import FFM_sampler
    from sampling.projectors import ProjectionConfig

    n_mix_orig = n_mix
    if n_mix < 1:
        print(f"  [ECI] WARNING: n_mix={n_mix} runs no mixing steps (output = "
              "initial noise). Overriding to n_mix=1.")
        n_mix = 1

    sampler = FFM_sampler(model=ffm.model, gp=ffm.gp)
    dims    = list(config.sample_dims)
    grid    = make_grid(dims, device)

    if initial_noise is None:
        torch.manual_seed(seed)
    flow_path_fractions = flow_path_fractions or [0.0, 0.25, 0.5, 0.75, 1.0]
    path_chunks: list[np.ndarray] = []
    path_indices: list[int] = []
    eci_diagnostics: list[dict] = []
    all_samples: list[torch.Tensor] = []
    n_generated = 0

    task_constraint = None
    resolved_projector_config = projector_config
    if task is None:
        raise ValueError("ECI requires a production ConstraintTask")
    if eci_mode not in ("native", "gn"):
        raise ValueError(f"eci_mode must be 'native' or 'gn', got {eci_mode!r}.")
    representative = _representative_task(task)
    resolved_projector_config = (
        projector_config or ProjectionConfig(scales=representative.scales)
    )

    while n_generated < num_samples:
        B  = min(batch_size, num_samples - n_generated)
        if initial_noise is not None:
            u0 = slice_initial_noise(initial_noise, n_generated, n_generated + B, device)
        else:
            u0 = ffm.gp.sample(grid, dims, n_samples=B).to(device)

        batch_task = _task_batch(task, n_generated, B)
        task_constraint = TaskProjectionConstraint(
            batch_task,
            config=resolved_projector_config,
            use_native=(eci_mode == "native"),
        )
        constraint = task_constraint
        resample_step = resample_every if resample_every > 0 else None

        rec = _record_indices(n_generated, B, max_path_samples) if save_flow_paths else None
        if save_sampler_diagnostics or save_flow_paths:
            diagnostics_start = 0
            result, diag, paths = sampler.eci_sample_with_diagnostics(
                u0=u0,
                n_step=n_step,
                n_mix=n_mix,
                resample_step=resample_step,
                constraint=constraint,
                path_fractions=flow_path_fractions if rec else None,
                record_indices=rec,
                apply_terminal_correction=not intermediate_only,
            )
            if save_sampler_diagnostics:
                source_rows = (task_constraint.diagnostics[diagnostics_start:]
                               if task_constraint is not None else diag.get("eci_per_sample", []))
                for row in source_rows:
                    row = dict(row)
                    row["sample_index"] = n_generated + row["sample_index"]
                    eci_diagnostics.append(row)
            if rec and paths:
                path_chunks.append(np.stack([paths[f].numpy() for f in flow_path_fractions], axis=1))
                path_indices.extend([n_generated + j for j in rec])
        else:
            result = sampler.eci_sample(
                u0=u0,
                n_step=n_step,
                n_mix=n_mix,
                resample_step=resample_step,
                constraint=constraint,
                apply_terminal_correction=not intermediate_only,
            )
        all_samples.append(result.detach().cpu())
        n_generated += B
        print(f"  ECI:  {n_generated}/{num_samples} samples generated")

    meta = {
        "sampler":         f"FFM_sampler.eci_sample (Euler + ECI-{eci_mode} task projection)",
        "n_step":          n_step,
        "n_mix":           n_mix,
        "n_mix_requested": n_mix_orig,
        "resample_every":  resample_every,
        "eci_mode":        eci_mode,
        "intermediate_only": bool(intermediate_only),
        "intrinsic_terminal_intervention_applied": not intermediate_only,
        "intrinsic_adjustments_per_sample": n_step * n_mix - int(intermediate_only),
        "eci_projector_max_iter": (
            int(resolved_projector_config.max_iter)
        ),
        "eci_projector_damping": (
            float(resolved_projector_config.damping)
        ),
        "eci_projector_tolerance": (
            float(resolved_projector_config.tol)
        ),
        "eci_projector_condition_number_diagnostics": (
            bool(resolved_projector_config.compute_condition_number)
        ),
        "eci_constraint":  f"TaskProjectionConstraint(use_native={eci_mode == 'native'}) — projects onto task manifold at every mixing step",
        **shared_noise_metadata_fields(initial_noise_path, initial_noise, seed),
    }
    meta.update(_task_metadata(task))
    if resample_every > 0 and initial_noise is not None:
        meta["eci_shared_noise_warning"] = (
            "ECI resample_every > 0 can break strict shared-noise terminal correspondence."
        )
    if save_sampler_diagnostics:
        meta["eci_per_sample"] = eci_diagnostics
    if save_flow_paths:
        meta["flow_paths_file"] = _finalize_flow_paths(run_dir, path_chunks, path_indices, flow_path_fractions)
        meta["flow_path_fractions"] = flow_path_fractions
    return torch.cat(all_samples, dim=0), meta


def run_mintflow_sampling(
    ffm,
    config,
    num_samples: int,
    num_candidates: int,
    ridge: float,
    time_penalty: float,
    forward_steps: int,
    batch_size: int,
    device: str,
    seed: int,
    initial_noise: Optional[torch.Tensor] = None,
    initial_noise_path: Optional[str] = None,
    save_flow_paths: bool = False,
    flow_path_fractions: Optional[list[float]] = None,
    max_path_samples: int = 32,
    save_sampler_diagnostics: bool = False,
    save_mathematical_diagnostics: bool = False,
    run_dir: Optional[str] = None,
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
    task: Optional["ConstraintTask"] = None,
) -> tuple[torch.Tensor, dict]:
    """MintFlow: Adjoint-Based Optimal Correction (see ``Constrained_FM.pdf``).

    For each sample: forward-solve the flow ODE with fixed-step Euler, compute
    the terminal residual ``r = H(u_T)`` and Jacobian ``J = dH/du_T``, solve the
    exact discrete-Euler sensitivity backward along the trajectory (never
    forming ``df/du``), evaluate the selected pseudoinverse or explicitly
    damped minimum-norm correction at ``num_candidates`` candidate
    times, select the time/perturbation minimising
    ``||delta||^2 + time_penalty * (T - s)^2``, and re-integrate the *original*
    flow from that point to ``T``.

    A shared Scenario-1 task or aligned per-sample Scenario-2 task provider
    defines ``H``. Samples are processed individually because the selected
    correction time may differ by sample.
    """
    from models.functional import make_grid
    from sampling.ffm_sampler import FFM_sampler

    sampler  = FFM_sampler(model=ffm.model, gp=ffm.gp)
    dims     = list(config.sample_dims)
    grid     = make_grid(dims, device)
    if task is None:
        raise ValueError("MintFlow requires a production ConstraintTask")

    if initial_noise is None:
        torch.manual_seed(seed)
    flow_path_fractions = flow_path_fractions or [0.0, 0.25, 0.5, 0.75, 1.0]
    path_chunks: list[np.ndarray] = []
    path_indices: list[int] = []
    all_samples: list[torch.Tensor] = []
    all_diagnostics: list[dict] = []
    n_generated = 0

    while n_generated < num_samples:
        B = min(batch_size, num_samples - n_generated)

        if initial_noise is not None:
            u0_batch = slice_initial_noise(initial_noise, n_generated, n_generated + B, device)
        else:
            u0_batch = ffm.gp.sample(grid, dims, n_samples=B).to(device)

        batch_results: list[torch.Tensor] = []
        for i in range(B):
            sample_idx = n_generated + i
            sample_task = _task_for_sample(task, sample_idx)
            if sample_task.enforcement_jacobian is None:
                raise RuntimeError(
                    f"MintFlow requires an analytic Jacobian for {sample_task.name!r}"
                )
            result, diag = sampler.mintflow_sample(
                u0=u0_batch[i : i + 1],
                hfunc=sample_task.normalized_residual,
                jacobian_func=sample_task.normalized_jacobian,
                forward_steps=forward_steps,
                num_candidates=num_candidates,
                ridge=ridge,
                time_penalty=time_penalty,
                score_mode=score_mode,
                interior_penalty=interior_penalty,
                correction_weight=correction_weight,
                candidate_t_min=candidate_t_min,
                candidate_t_max=candidate_t_max,
                time_sampling=time_sampling,
                end_bias_power=end_bias_power,
                correction_mode=correction_mode,
                correction_scale=correction_scale,
                pseudoinverse_rcond=pseudoinverse_rcond,
                adjoint_chunk_size=adjoint_chunk_size,
                save_mathematical_diagnostics=save_mathematical_diagnostics,
                path_fractions=(
                    flow_path_fractions
                    if save_flow_paths and sample_idx < max_path_samples
                    else None
                ),
            )
            corrected_path = diag.pop("_corrected_path_snapshots", None)
            batch_results.append(result.detach().cpu())
            all_diagnostics.append(diag)

            if save_flow_paths and sample_idx < max_path_samples:
                if corrected_path is None:
                    raise RuntimeError("MintFlow corrected path snapshots were not returned")
                path_chunks.append(np.stack(corrected_path, axis=0)[None, ...])
                path_indices.append(sample_idx)
        all_samples.append(torch.cat(batch_results, dim=0))

        n_generated += B
        print(f"  MintFlow:  {n_generated}/{num_samples} samples generated")

    import statistics as _stats

    s_stars     = [d["s_star"] for d in all_diagnostics]
    res_before  = [d["residual_norm_before"] for d in all_diagnostics]
    res_after   = [d["residual_norm_after"] for d in all_diagnostics]
    corr_norms  = [d["correction_norm"] for d in all_diagnostics]

    # Aggregate numerical-safety warnings across all samples (deduplicated,
    # with a per-warning-type sample count so isolated vs. systemic issues
    # are distinguishable at a glance).
    warning_counts: dict[str, int] = {}
    for d in all_diagnostics:
        for w in d.get("warnings", []):
            warning_counts[w] = warning_counts.get(w, 0) + 1
    warnings_summary = [
        f"{w} ({count}/{len(all_diagnostics)} samples)"
        for w, count in sorted(warning_counts.items())
    ]

    per_sample_diagnostics = [
        {
            "sample_index":             i,
            "s_star":                   d["s_star"],
            "best_idx":                 d["best_idx"],
            "terminal_residual_before": d["terminal_residual_before"],
            "terminal_residual_after":  d["terminal_residual_after"],
            "terminal_residual_inf_before": d.get("terminal_residual_inf_before"),
            "terminal_residual_inf_after": d.get("terminal_residual_inf_after"),
            "correction_norm":          d["correction_norm"],
            "candidate_scores":         d["candidate_scores"],
            "candidate_times":          d.get("candidate_times", {}),
            "candidate_correction_norms": d.get("candidate_correction_norms", {}),
            "candidate_condition_numbers": d.get("candidate_condition_numbers", {}),
            "candidate_adjoint_norms": d.get("candidate_adjoint_norms", {}),
            "candidate_predicted_residual_norms": d.get("candidate_predicted_residual_norms", {}),
            "candidate_realized_residual_norms": d.get("candidate_realized_residual_norms", {}),
            "candidate_realized_residual_infs": d.get("candidate_realized_residual_infs", {}),
            "candidate_diagnostics": d.get("candidate_diagnostics", []),
            "candidate_score_breakdown": d.get("candidate_score_breakdown", {}),
            "linearization_convergence": d.get("linearization_convergence", []),
            "zero_restart_relative_field_error": d.get("zero_restart_relative_field_error"),
            "zero_restart_residual_vector_error": d.get("zero_restart_residual_vector_error"),
            "zero_restart_R_inf_difference": d.get("zero_restart_R_inf_difference"),
            "terminal_jacobian_frobenius_norm": d.get("terminal_jacobian_frobenius_norm"),
            "terminal_jacobian_backend": d.get("terminal_jacobian_backend"),
            "selected_adjoint_norm": d.get("selected_adjoint_norm"),
            "selected_predicted_residual_norm": d.get("selected_predicted_residual_norm"),
            "selected_predicted_residual_inf": d.get("selected_predicted_residual_inf"),
            "line_search_used": d.get("line_search_used", False),
            "damping_ridge": d.get("damping_ridge", ridge),
            "adjoint_mode": d.get("adjoint_mode"),
            "suffix_reintegration": d.get("suffix_reintegration"),
            "correction_mode": d.get("correction_mode"),
            "correction_scale": d.get("correction_scale", correction_scale),
            "pseudoinverse_rcond": d.get("pseudoinverse_rcond"),
            "time_sampling": d.get("mintflow_time_sampling"),
            "end_bias_power": d.get("mintflow_end_bias_power"),
            "score":                    d.get("score"),
            "warnings":                 d["warnings"],
        }
        for i, d in enumerate(all_diagnostics)
    ]

    def _mstd(vals):
        return float(_stats.mean(vals)), (float(_stats.stdev(vals)) if len(vals) > 1 else 0.0)

    s_star_mean, s_star_std = _mstd(s_stars)
    corr_mean, corr_std = _mstd(corr_norms)

    return torch.cat(all_samples, dim=0), {
        "sampler":             "FFM_sampler.mintflow_sample (adjoint-based optimal correction)",
        "num_candidates":      num_candidates,
        "ridge":               ridge,
        "time_penalty":        time_penalty,
        "forward_steps":       forward_steps,
        "mintflow_adjoint_mode":    "discrete_euler_exact_autograd_batched",
        "mintflow_suffix_reintegration": "original_remaining_euler_grid",
        "mintflow_correction_mode": correction_mode,
        "mintflow_correction_scale": correction_scale,
        "mintflow_pseudoinverse_rcond": pseudoinverse_rcond,
        "mintflow_adjoint_chunk_size": adjoint_chunk_size,
        "mintflow_mathematical_diagnostics_enabled": save_mathematical_diagnostics,
        "mintflow_score_mode":      score_mode,
        "mintflow_interior_penalty": interior_penalty,
        "mintflow_correction_weight": correction_weight,
        "mintflow_candidate_t_min": candidate_t_min,
        "mintflow_candidate_t_max": candidate_t_max,
        "mintflow_time_sampling": time_sampling,
        "mintflow_end_bias_power": end_bias_power,
        "hfunc":               f"ConstraintTask.{task.name}",
        "mintflow_constraint_components": list(task.component_names),
        **_task_metadata(task),
        "mintflow_s_star_mean":     s_star_mean,
        "mintflow_s_star_std":      s_star_std,
        "mintflow_s_star_median":   float(_stats.median(s_stars)),
        "mintflow_s_star_min":      float(min(s_stars)),
        "mintflow_s_star_max":      float(max(s_stars)),
        "mintflow_residual_before_mean": float(sum(res_before) / len(res_before)),
        "mintflow_residual_after_mean":  float(sum(res_after) / len(res_after)),
        "terminal_residual_before_mean": float(sum(res_before) / len(res_before)),
        "terminal_residual_after_mean":  float(sum(res_after) / len(res_after)),
        "mintflow_correction_norm_mean": corr_mean,
        "mintflow_correction_norm_std":  corr_std,
        "mintflow_correction_norm_min":  float(min(corr_norms)),
        "mintflow_correction_norm_max":  float(max(corr_norms)),
        "mintflow_per_sample":      per_sample_diagnostics,
        "warnings":            warnings_summary,
        **shared_noise_metadata_fields(initial_noise_path, initial_noise, seed),
        **({"flow_paths_file": _finalize_flow_paths(run_dir, path_chunks, path_indices, flow_path_fractions),
            "flow_path_fractions": flow_path_fractions} if save_flow_paths else {}),
    }


def run_mintflow_project_sampling(
    ffm,
    config,
    num_samples: int,
    num_candidates: int,
    ridge: float,
    time_penalty: float,
    forward_steps: int,
    batch_size: int,
    device: str,
    seed: int,
    task: "ConstraintTask | list[ConstraintTask]",
    initial_noise: Optional[torch.Tensor] = None,
    initial_noise_path: Optional[str] = None,
    save_flow_paths: bool = False,
    flow_path_fractions: Optional[list[float]] = None,
    max_path_samples: int = 32,
    save_sampler_diagnostics: bool = False,
    run_dir: Optional[str] = None,
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
    projector_config=None,
) -> tuple[torch.Tensor, dict]:
    """Optional MintFlow variant (Part 5): MintFlow's adjoint-based correction followed
    by exactly one terminal projection, using the *same* shared projector
    library as the ``final_projection`` baseline (``ConstraintTask.project``).

    Motivation: MintFlow's damped minimum-norm correction at the selected
    candidate time ``s*`` is generally *not* an exact terminal projection
    (it is a linearized one-step correction propagated through the
    remaining flow, which is nonlinear) — so residual after MintFlow alone need
    not be at the projector's convergence tolerance. This variant answers
    "does one extra cheap terminal clean-up on top of MintFlow help?" and is
    directly comparable to the plain ``final_projection`` baseline (same
    projector, same task) and to plain MintFlow (same adjoint correction).

    ``task`` is required because a terminal projection has no meaning without
    a target manifold. It may be a shared Scenario-1 task or the lazy paired
    Scenario-2 task provider.
    """
    if task is None:
        raise ValueError(
            "run_mintflow_project_sampling requires `task` (a ConstraintTask, or a "
            "paired Scenario-2 provider) — a terminal projection has no "
            "manifold to project onto "
            "without one. Use run_mintflow_sampling directly if you don't want the "
            "extra terminal projection step."
        )

    samples, mintflow_meta = run_mintflow_sampling(
        ffm, config, num_samples,
        num_candidates, ridge, time_penalty, forward_steps,
        batch_size, device, seed,
        initial_noise=initial_noise, initial_noise_path=initial_noise_path,
        save_flow_paths=save_flow_paths, flow_path_fractions=flow_path_fractions,
        max_path_samples=max_path_samples, save_sampler_diagnostics=save_sampler_diagnostics,
        run_dir=run_dir, score_mode=score_mode, interior_penalty=interior_penalty,
        correction_weight=correction_weight, candidate_t_min=candidate_t_min,
        candidate_t_max=candidate_t_max, time_sampling=time_sampling,
        end_bias_power=end_bias_power, correction_mode=correction_mode,
        correction_scale=correction_scale,
        pseudoinverse_rcond=pseudoinverse_rcond,
        adjoint_chunk_size=adjoint_chunk_size,
        task=task,
    )

    projected, per_sample_proj, proj_summary = apply_final_projection(
        samples, task, config=projector_config,
    )

    task0 = _representative_task(task)
    # Promote shared-noise and flow-path fields so evaluation can use one
    # metadata schema across all methods.
    shared_fields = {
        k: mintflow_meta[k]
        for k in (
            "shared_initial_noise",
            "initial_noise_path",
            "initial_noise_shape",
            "initial_noise_seed",
            "flow_paths_file",
            "flow_path_fractions",
        )
        if k in mintflow_meta
    }
    meta = {
        "sampler": "FFM_sampler.mintflow_sample + single terminal projection (mintflow_project variant)",
        "mintflow_meta": mintflow_meta,
        "final_projection_summary": proj_summary,
        "final_projection_per_sample": per_sample_proj,
        **_task_metadata(task0),
        **shared_fields,
    }
    return projected, meta


def apply_final_projection(
    samples: torch.Tensor,
    task: "ConstraintTask | list[ConstraintTask]",
    config=None,
) -> tuple[torch.Tensor, list[dict], dict]:
    """Project every sample in *samples* exactly once onto its task's manifold.

    Pure function (no model, no I/O) so it is unit-testable in isolation from
    :func:`run_final_projection_sampling`, which supplies the vanilla-FFM
    samples via :func:`run_vanilla_sampling`.

    Parameters
    ----------
    samples : ``(N, *dims)`` CPU or device tensor (vanilla FFM output).
    task    : A single shared :class:`~sampling.tasks.ConstraintTask`, a list
              of ``N`` tasks, or a lazy Scenario-2 provider holding one task
              for each paired IC/BC target.
    config  : Optional :class:`~sampling.projectors.ProjectionConfig`.
              Defaults to ``ProjectionConfig(scales=task.scales)`` (using the
              first task's scales when *task* is a list).

    Returns
    -------
    projected      : ``(N, *dims)`` tensor, same device/dtype as *samples*.
    per_sample     : list of ``N`` dicts with pre/post residual norms,
                     convergence, iterations, condition number, method, and
                     warnings (see :class:`~sampling.projectors.ProjectionResult`).
    summary        : aggregate dict (mean/median/p95/max residual before and
                      after, convergence fraction, feasible fraction).
    """
    from sampling.projectors import ProjectionConfig

    tasks = task if isinstance(task, (list, tuple)) else None
    provider = task if hasattr(task, "task_for") else None
    task0 = _representative_task(task)
    config = config or ProjectionConfig(scales=task0.scales)
    dims = samples.shape[1:]
    n = samples.shape[0]
    if tasks is not None and len(tasks) != n:
        raise ValueError(f"len(task) ({len(tasks)}) must equal len(samples) ({n}) when task is a list.")
    if provider is not None and provider.num_tasks != n:
        raise ValueError(
            f"task provider has {provider.num_tasks} targets, expected {n}"
        )
    projected = torch.empty_like(samples)
    per_sample: list[dict] = []

    for i in range(n):
        this_task = (
            tasks[i] if tasks is not None
            else provider.task_for(i) if provider is not None
            else task
        )
        xi_flat = samples[i].reshape(-1)
        result = this_task.project(xi_flat, config)
        projected[i] = result.u.reshape(dims).to(samples.dtype)
        projection_delta = result.u.to(xi_flat) - xi_flat
        source_norm = xi_flat.norm().clamp_min(torch.finfo(xi_flat.dtype).tiny)
        per_sample.append({
            "sample_index":          i,
            "method":                result.method,
            "converged":             result.converged,
            "iterations":            result.iterations,
            "initial_residual_norm": result.initial_residual_norm,
            "final_residual_norm":   result.final_residual_norm,
            "feasible":              result.feasible,
            "condition_number":      result.condition_number,
            "projection_correction_l2": float(projection_delta.norm().item()),
            "projection_correction_relative_l2": float(
                (projection_delta.norm() / source_norm).item()
            ),
            "projection_correction_linf": float(
                projection_delta.abs().max().item()
            ),
            "warnings":              result.warnings,
        })

    before = np.array([d["initial_residual_norm"] for d in per_sample])
    after  = np.array([d["final_residual_norm"] for d in per_sample])
    correction_l2 = np.array(
        [d["projection_correction_l2"] for d in per_sample], dtype=np.float64
    )
    correction_relative_l2 = np.array(
        [d["projection_correction_relative_l2"] for d in per_sample],
        dtype=np.float64,
    )
    summary = {
        "residual_before_mean":   float(before.mean()),
        "residual_before_median": float(np.median(before)),
        "residual_before_p95":    float(np.percentile(before, 95)),
        "residual_before_max":    float(before.max()),
        "residual_after_mean":    float(after.mean()),
        "residual_after_median":  float(np.median(after)),
        "residual_after_p95":     float(np.percentile(after, 95)),
        "residual_after_max":     float(after.max()),
        "converged_fraction":     float(np.mean([d["converged"] for d in per_sample])),
        "feasible_fraction":      float(np.mean([d["feasible"] for d in per_sample])),
        "projector_method":       per_sample[0]["method"] if per_sample else "",
        "projection_correction_l2_mean": float(correction_l2.mean()),
        "projection_correction_l2_max": float(correction_l2.max()),
        "projection_correction_relative_l2_mean": float(
            correction_relative_l2.mean()
        ),
    }
    return projected, per_sample, summary


def apply_optional_final_projection(
    samples: torch.Tensor,
    metadata: dict[str, Any],
    task: "ConstraintTask | list[ConstraintTask]",
    *,
    enabled: bool,
    config=None,
    save_per_sample: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Uniform optional post-sampling projection for any sampler output.

    This wrapper deliberately describes the operation as *post-sampling*:
    PCFM and ECI retain their intrinsic in-loop endpoint corrections. When
    enabled, exactly one additional call to :func:`apply_final_projection`
    is made after the underlying sampler returns.
    """
    out_meta = dict(metadata)
    out_meta["post_sampling_final_projection_enabled"] = bool(enabled)
    out_meta["post_sampling_final_projection_count"] = 1 if enabled else 0
    if not enabled:
        return samples, out_meta

    projected, per_sample, summary = apply_final_projection(samples, task, config=config)
    out_meta["sampler_before_final_projection"] = metadata.get("sampler", "")
    out_meta["sampler"] = f"{metadata.get('sampler', 'sampler')} + optional terminal projection"
    out_meta["final_projection_summary"] = summary
    if save_per_sample:
        out_meta["final_projection_per_sample"] = per_sample

    # Flow paths describe the returned sampler trajectory. A post-hoc
    # projection is an instantaneous terminal operation, so replace only the
    # tau=1 snapshot while preserving every earlier state.
    path_value = out_meta.get("flow_paths_file")
    if path_value:
        path = Path(path_value)
        if path.exists():
            with np.load(path) as payload:
                path_samples = payload["path_samples"].copy()
                path_indices = payload["path_sample_indices"].copy()
                fractions = payload["flow_path_fractions"].copy()
            terminal = np.flatnonzero(np.isclose(fractions, 1.0, rtol=0.0, atol=1e-12))
            if len(terminal) == 1:
                for row, sample_index in enumerate(path_indices):
                    path_samples[row, terminal[0]] = projected[int(sample_index)].detach().cpu().numpy()
                save_flow_paths_npz(path, path_samples, path_indices, fractions.tolist())
                out_meta["flow_paths_terminal_projection_applied"] = True
            else:
                out_meta["flow_paths_terminal_projection_applied"] = False

    return projected, out_meta


def run_final_projection_sampling(
    ffm,
    config,
    num_samples: int,
    task: "ConstraintTask | list[ConstraintTask]",
    n_eval: int,
    batch_size: int,
    device: str,
    seed: int,
    initial_noise: Optional[torch.Tensor] = None,
    initial_noise_path: Optional[str] = None,
    save_flow_paths: bool = False,
    flow_path_fractions: Optional[list[float]] = None,
    max_path_samples: int = 32,
    save_sampler_diagnostics: bool = False,
    run_dir: Optional[str] = None,
    projector_config=None,
) -> tuple[torch.Tensor, dict]:
    """Final-projection baseline: vanilla FFM + exactly one terminal projection.

    Isolates "generation" from "correction" — unlike PCFM/MintFlow/ECI which
    intervene *during* sampling, this method generates completely
    unconstrained samples and then applies a single call to
    :func:`apply_final_projection` (which dispatches to each task's
    ``native_projector`` when available, else damped Gauss-Newton). This is
    the fairness baseline requested for comparing "in-the-loop" correction
    methods against "correct once at the end".

    ``task`` may be a shared :class:`~sampling.tasks.ConstraintTask`, a list
    of per-sample tasks, or the lazy paired Scenario-2 task provider built by
    ``scripts/run_sampling.py``.
    """
    samples, vanilla_meta = run_vanilla_sampling(
        ffm, config, num_samples, n_eval, batch_size, device, seed,
        initial_noise=initial_noise,
        initial_noise_path=initial_noise_path,
        save_flow_paths=save_flow_paths,
        flow_path_fractions=flow_path_fractions,
        max_path_samples=max_path_samples,
        run_dir=run_dir,
    )

    projected, projected_meta = apply_optional_final_projection(
        samples,
        vanilla_meta,
        task,
        enabled=True,
        config=projector_config,
        save_per_sample=save_sampler_diagnostics,
    )
    summary = projected_meta["final_projection_summary"]
    per_sample = projected_meta.get("final_projection_per_sample", [])

    task0 = _representative_task(task)
    # Promote shared-noise / flow-path fields from the nested vanilla_meta so
    # eval_pde trajectory-consistency can align same-seed vanilla without
    # requiring nested-metadata recovery.
    shared_fields = {
        k: vanilla_meta[k]
        for k in (
            "shared_initial_noise",
            "initial_noise_path",
            "initial_noise_shape",
            "initial_noise_seed",
            "flow_paths_file",
            "flow_path_fractions",
        )
        if k in vanilla_meta
    }
    meta = {
        "sampler":    "vanilla FFM + single terminal projection (final_projection baseline)",
        "vanilla_meta": vanilla_meta,
        "final_projection_summary": summary,
        "final_projection_per_sample_tasks": (
            isinstance(task, (list, tuple)) or hasattr(task, "task_for")
        ),
        **_task_metadata(task0),
        **shared_fields,
        "post_sampling_final_projection_enabled": True,
        "post_sampling_final_projection_count": 1,
        "flow_paths_terminal_projection_applied": projected_meta.get(
            "flow_paths_terminal_projection_applied", False,
        ),
    }
    if save_sampler_diagnostics:
        meta["final_projection_per_sample"] = per_sample
    return projected, meta


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _task_metadata(task: "ConstraintTask") -> dict:
    """Common ``ConstraintTask`` metadata block shared by every method's run metadata.

    Callers (``_resolve_task`` in ``scripts/run_sampling.py``) are expected to
    stash target-bank provenance (``target_id``, ``reference_pool_hash``,
    ``reference_pool_path``, ``quantile``/``cluster_label``) inside
    ``task.metadata`` before sampling, so it is captured here automatically.
    """
    target = task.target
    return {
        "task_name":            task.name,
        "task_dataset":         task.dataset,
        "task_display_name":    task.display_name,
        "task_output_dim":      task.output_dim,
        "task_component_names": list(task.component_names),
        "task_units":           list(task.units),
        "task_target_raw":      target.detach().cpu().tolist() if target is not None else None,
        "task_target_mode": (
            "paired_per_sample_bank" if hasattr(task, "task_for") else "shared"
        ),
        "task_num_paired_targets": (
            int(task.num_tasks) if hasattr(task, "task_for") else None
        ),
        "task_target_normalized": (
            (target / task.scales.clamp_min(1e-300)).detach().cpu().tolist()
            if target is not None else None
        ),
        "task_scales":          task.scales.detach().cpu().tolist(),
        "task_native_projector_available": task.native_projector is not None,
        "task_metadata":        task.metadata,
    }
