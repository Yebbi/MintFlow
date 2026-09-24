#!/usr/bin/env python3
# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
"""
run_sampling.py — unified FFM sampling + constraint-evaluation CLI (Step 3/4).

Supports configurable sampling methods:
  --method vanilla   Dopri5 ODE via FFM.sample() (default)
  --method pcfm      Physics-Constrained Flow Matching (Euler + Newton projection)
  --method eci       Energy-Constrained Inference (Euler + IC correction)
  --method mintflow       Adjoint-Based Optimal Correction (Euler + adjoint solve + single correction)

All methods write the same output structure:
    <output-dir>/<dataset>/<method>/ckpt_<stem>_seed<s>_n<N>_<ts>/
        samples.npy               — generated samples (N, *dims)
        metadata.json             — run parameters + method-specific config
        constraint_metrics.json   — aggregate residual statistics (if --evaluate-constraints)
        constraint_metrics.csv    — per-sample residual norms (if --evaluate-constraints)
        previews/
            sample_0000.png
            ...

Usage — vanilla Heat
--------------------
    uv run python scripts/run_sampling.py \\
        --dataset diffusion --method vanilla \\
        --config configs/heat.yml --checkpoint logs/diffusion/latest.pt \\
        --num-samples same-as-test --n-eval 10 --batch-size 16 --seed 0 \\
        --device cuda --output-dir outputs/sampling --num-preview 8 \\
        --evaluate-constraints

Usage — PCFM
------------
    uv run python scripts/run_sampling.py \\
        --dataset diffusion --method pcfm \\
        --config configs/heat.yml --checkpoint logs/diffusion/latest.pt \\
        --num-samples 256 --pcfm-n-step 100 --pcfm-newton-steps 10 \\
        --batch-size 16 --seed 0 --device cuda \\
        --output-dir outputs/sampling --evaluate-constraints

Usage — ECI
-----------
    uv run python scripts/run_sampling.py \\
        --dataset diffusion --method eci \\
        --config configs/heat.yml --checkpoint logs/diffusion/latest.pt \\
        --num-samples 256 --eci-n-step 100 --eci-n-mix 1 \\
        --batch-size 16 --seed 0 --device cuda \\
        --output-dir outputs/sampling --evaluate-constraints

Usage — MintFlow
-----------
    uv run python scripts/run_sampling.py \\
        --dataset diffusion --method mintflow \\
        --config configs/heat.yml --checkpoint logs/diffusion/latest.pt \\
        --num-samples 8 --mintflow-num-candidates 10 --mintflow-ridge 1e-5 \\
        --mintflow-time-penalty 0.01 --mintflow-forward-steps 200 \\
        --mintflow-candidate-range 0.1-0.98 --time-sampling end_biased \\
        --final-projection \\
        --batch-size 8 --seed 0 --device cuda \\
        --output-dir outputs/sampling --evaluate-constraints
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from sampling.io import (
    make_sampling_run_dir,
    save_metadata,
    save_metrics_csv,
    save_metrics_json,
    save_samples,
)
from visualization.datasets import (
    SUPPORTED_DATASETS,
    default_config_for_dataset,
    get_vis_kwargs,
    load_dataset_split,
)
from visualization.io import get_git_commit
from visualization.plotting import save_sample_image
from visualization.sampling import (
    default_checkpoint_for_dataset,
    load_ffm_model,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Global constraint task resolution (--task / --target-bank / --target-vector)
# ---------------------------------------------------------------------------

def _projector_config_from_args(args: argparse.Namespace, scales):
    """Build a sampling.projectors.ProjectionConfig from CLI flags."""
    from sampling.projectors import ProjectionConfig
    return ProjectionConfig(
        max_iter=args.constraint_max_iter,
        damping=args.constraint_ridge,
        tol=args.constraint_tolerance,
        scales=scales if args.constraint_normalization == "task_scales" else None,
    )


def _apply_projector_choice(task, args: argparse.Namespace):
    """Honor an explicit --constraint-projector choice by wrapping/validating
    *task* so downstream ``task.project()`` calls use the requested family.

    'auto' (default) leaves the task untouched. Explicit 'linear'/'affine'
    require a compatible native_projector (error otherwise, since forcing a
    linear/affine solve on a task that has none would silently be wrong).
    'gauss_newton' always drops the native projector (forces the generic
    fallback) even if one exists -- useful for a fair native-vs-generic
    projector ablation, matching the ECI-native/ECI-GN distinction.
    """
    import dataclasses as _dc

    choice = args.constraint_projector
    if choice in (None, "auto"):
        return task
    if choice == "gauss_newton":
        return _dc.replace(task, native_projector=None)
    # linear / affine: require a native projector and (best-effort) check its
    # reported method matches, by probing it once at the zero vector.
    if task.native_projector is None:
        raise ValueError(
            f"--constraint-projector {choice!r} requires task '{task.name}' to have a "
            f"native_projector, but it has none (only the generic Gauss-Newton fallback "
            f"is available for this task). Use --constraint-projector auto or gauss_newton."
        )
    return task


def _resolve_task(
    args: argparse.Namespace,
    config,
    config_path: str,
    device: str,
    reference_sample: torch.Tensor | None = None,
    boundary_target: torch.Tensor | None = None,
    scenario_target_metadata: dict | None = None,
):
    """Build a shared task or a lazy one-to-one Scenario-2 task provider."""
    expected = {
        "diffusion": "heat_mass_conservation",
        "rd1d": "global_balance",
    }
    task_name = args.task or expected[args.dataset]
    if task_name != expected[args.dataset]:
        raise ValueError(
            f"Production dataset {args.dataset!r} requires --task {expected[args.dataset]!r}."
        )
    from sampling.tasks import ConstraintTaskProvider, build_task

    target_id = (
        scenario_target_metadata["target_id"]
        if scenario_target_metadata is not None else "__intrinsic__"
    )
    common_metadata = {
        "target_source": (
            "unconstrained_vanilla_ffm"
            if reference_sample is not None else "task_intrinsic"
        ),
        "target_id": target_id,
        "scenario_target": scenario_target_metadata,
    }

    if reference_sample is not None and reference_sample.ndim == 3:
        count = len(reference_sample)
        if boundary_target is not None and len(boundary_target) != count:
            raise ValueError("Scenario-2 IC and BC target counts differ")

        def build_paired(index: int):
            this_boundary = (
                boundary_target[index] if boundary_target is not None else None
            )
            paired = build_task(
                args.dataset,
                task_name,
                config=config,
                config_path=config_path,
                reference_sample=reference_sample[index],
                boundary_target=this_boundary,
                scenario=args.constraint_scenario,
                device=device,
            )
            paired.metadata = {
                **paired.metadata,
                **common_metadata,
                "target_kind": "paired_per_sample_bank",
                "target_index": int(index + args.sample_offset),
            }
            return _apply_projector_choice(paired, args)

        prototype = build_paired(0)
        provider_metadata = {
            **prototype.metadata,
            **common_metadata,
            "target_kind": "paired_per_sample_bank",
            "paired_target_count": count,
            "paired_target_offset": int(args.sample_offset),
        }
        return ConstraintTaskProvider(
            num_tasks=count,
            task_builder=build_paired,
            metadata=provider_metadata,
        )

    task = build_task(
        args.dataset,
        task_name,
        config=config,
        config_path=config_path,
        reference_sample=reference_sample,
        boundary_target=boundary_target,
        scenario=args.constraint_scenario,
        device=device,
    )
    task.metadata = {**task.metadata, **common_metadata}
    return _apply_projector_choice(task, args)


def _projection_task_for_samples(task):
    """Return the production task used by the terminal projector."""
    if task is None:
        raise RuntimeError("Production sampling requires an explicit constraint task")
    return task


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "PCFM Step 3/4 — configurable FFM sampling + constraint evaluation.\n"
            "Supported methods: vanilla, pcfm, eci, mintflow, DiffusionPDE, and D-Flow."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- shared / required -----------------------------------------------
    p.add_argument("--dataset",     required=True, choices=SUPPORTED_DATASETS,
                   help="PDE dataset to sample from.")
    p.add_argument("--method",      default="vanilla",
                   choices=["vanilla", "pcfm", "eci", "mintflow", "diffusionpde", "dflow", "final_projection", "mintflow_project"],
                   help="Sampling method.")
    p.add_argument("--config",      default=None, metavar="PATH",
                   help="YAML config path.  Required for latest.pt (no embedded config).")
    p.add_argument("--checkpoint",  default=None, metavar="PATH",
                   help="Checkpoint path.  Defaults to logs/<dataset>/latest.pt.")
    p.add_argument("--num-samples", default="same-as-test",
                   help="Integer or 'same-as-test' (uses len(test_set)).")
    p.add_argument(
        "--sample-offset", type=int, default=0,
        help=("Zero-based offset into a reused shared-noise pool. This makes "
              "large deterministic runs safely shardable; default: 0."),
    )
    p.add_argument("--batch-size",  type=int, default=16,
                   help="Samples per model call / GP draw.  Reduce if GPU OOM.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--device",      default=_default_device())
    p.add_argument("--output-dir",  default="outputs/sampling")
    p.add_argument("--save-format", default="npy", choices=["npy", "h5"])
    p.add_argument("--num-preview", type=int, default=8,
                   help="Preview PNG images to save (0 = skip).")

    # ---- constraint evaluation -------------------------------------------
    p.add_argument("--evaluate-constraints", action="store_true",
                   help="Evaluate physical constraint residuals on generated samples.")
    p.add_argument("--max-eval-samples", type=int, default=None,
                   help="Limit constraint evaluation to this many samples.")

    # ---- production global constraint task -------------------------------
    task_g = p.add_argument_group(
        "Physical constraint options",
        "Heat uses heat_mass_conservation; RD1D uses global_balance.",
    )
    task_g.add_argument("--task", default=None,
                         choices=["heat_mass_conservation", "global_balance"],
                         help="Optional explicit task; inferred from --dataset by default.")
    task_g.add_argument(
        "--constraint-scenario",
        default="conservation_only",
        choices=["conservation_only", "conservation_ic_bc"],
        help=(
            "Constraint composition shared by every sampler. conservation_only preserves "
            "the current mass/global-balance study. conservation_ic_bc additionally fixes "
            "one paired Vanilla-FFM IC/BC target for every generated sample."
        ),
    )
    task_g.add_argument(
        "--constraint-target-file",
        default=None,
        metavar="PATH.npz",
        help=(
            "Existing Scenario-2 paired target-bank artifact shared by all methods."
        ),
    )
    task_g.add_argument(
        "--constraint-target-seed",
        type=int,
        default=None,
        help=(
            "Seed recorded by the paired Vanilla target bank."
        ),
    )
    task_g.add_argument("--constraint-projector", default="auto",
                         choices=["auto", "linear", "affine", "gauss_newton"],
                         help="'auto' uses the task's native_projector when available, else "
                              "damped Gauss-Newton. Explicit values force that family and error "
                              "if the task's native_projector is incompatible with it.")
    task_g.add_argument("--constraint-tolerance", type=float, default=1e-6,
                         help="Convergence / feasibility threshold on the scaled residual norm.")
    task_g.add_argument("--constraint-ridge", type=float, default=1e-6,
                         help="Levenberg-Marquardt damping added to (J J^T) before solving.")
    task_g.add_argument("--constraint-max-iter", type=int, default=10,
                         help="Max damped Gauss-Newton iterations (ignored by the exact "
                              "linear/affine one-shot projectors).")
    task_g.add_argument("--constraint-normalization", default="task_scales",
                         choices=["task_scales", "none"],
                         help="'task_scales' (default) normalizes each residual component by "
                              "the task's scales before computing norms/tolerances; 'none' uses "
                              "raw (unnormalized) residuals.")
    task_g.add_argument("--eci-mode", default="native", choices=["native", "gn"],
                         help="ECI correction mode: project onto the task manifold every "
                              "mixing step, using the "
                              "task's native closed-form projector ('native') or always the "
                              "generic damped Gauss-Newton projector ('gn', for a fair ablation).")
    task_g.add_argument("--save-constraint-diagnostics", action="store_true",
                         help="Save per-sample projection/task diagnostics in metadata.json "
                              "(final_projection_per_sample / eci_per_sample with exact "
                              "convergence, iterations, condition numbers). "
                              "Equivalent to --save-sampler-diagnostics for task-mode runs.")
    task_g.add_argument(
        "--final-projection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply one additional post-sampling ConstraintTask projection to the returned "
            "field. The production default is enabled for MintFlow and disabled for all "
            "other methods; use --no-final-projection for the MintFlow ablation. This is "
            "intentionally disallowed for PCFM/ECI because their standard algorithms "
            "already contain a terminal intervention."
        ),
    )
    task_g.add_argument(
        "--intermediate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "PCFM/ECI ablation: retain every intermediate projection and pullback, but "
            "skip the final intrinsic projection/correction that directly produces the "
            "returned sample. Only valid with --method pcfm or --method eci."
        ),
    )

    # ---- misc -----------------------------------------------------------
    p.add_argument("--wandb-mode",  default="disabled",
                   choices=["disabled", "offline", "online"])

    # ---- vanilla-specific -----------------------------------------------
    vanilla_g = p.add_argument_group("Vanilla FFM options")
    vanilla_g.add_argument("--n-eval", type=int, default=10,
                            help="ODE evaluation points for dopri5.")

    # ---- PCFM-specific --------------------------------------------------
    pcfm_g = p.add_argument_group("PCFM options (--method pcfm)")
    pcfm_g.add_argument("--pcfm-n-step",      type=int,   default=100,
                        help="Number of Euler integration steps.")
    pcfm_g.add_argument("--pcfm-newton-steps", type=int,  default=10,
                        help="Newton iterations per projection step.")
    pcfm_g.add_argument("--pcfm-tol",         type=float, default=1e-6,
                        help="Jacobian solve regularisation (eps in J@J^T + eps*I).")

    # ---- ECI-specific ---------------------------------------------------
    eci_g = p.add_argument_group("ECI options (--method eci)")
    eci_g.add_argument("--eci-n-step",       type=int, default=100,
                       help="Number of Euler integration steps.")
    eci_g.add_argument("--eci-n-mix",        type=int, default=1,
                       help="Mixing steps per Euler step (must be ≥ 1 for ECI to operate).")
    eci_g.add_argument("--eci-resample-every", type=int, default=0,
                       help="Resample initial noise every N mixing steps (0 = never).")
    eci_g.add_argument(
        "--eci-projector-max-iter", type=int, default=None,
        help=(
            "Maximum nonlinear projector iterations per ECI adjustment. "
            "Defaults to --constraint-max-iter. A value of 1 gives a "
            "single Gauss-Newton update per flow/mixing step."
        ),
    )

    # ---- gradient-guidance baselines ------------------------------------
    guided_g = p.add_argument_group("Gradient-guidance baselines")
    guided_g.add_argument("--guided-n-step", type=int, default=100,
                          help="Fixed Euler steps for DiffusionPDE and D-Flow.")
    guided_g.add_argument("--diffusionpde-eta", type=float, default=1.0,
                          help="DiffusionPDE constraint-gradient coefficient (PCFM paper: 1.0).")
    guided_g.add_argument("--dflow-n-iter", type=int, default=20,
                          help="D-Flow LBFGS iterations (PCFM/D-Flow setup: 20).")
    guided_g.add_argument("--dflow-lr", type=float, default=1.0,
                          help="D-Flow LBFGS learning rate (PCFM setup: 1.0).")

    # ---- MintFlow-specific -----------------------------------------------------
    mintflow_g = p.add_argument_group("MintFlow options (--method mintflow)")
    mintflow_g.add_argument("--mintflow-num-candidates", type=int, default=10,
                       help="Number of candidate correction times s_1 < ... < s_K.")
    mintflow_g.add_argument("--mintflow-ridge",         type=float, default=1e-5,
                       help="Ridge regularisation in (A A^T + ridge*I) for the "
                            "damped minimum-norm correction.")
    mintflow_g.add_argument("--mintflow-correction-mode", default="pseudoinverse",
                       choices=("pseudoinverse", "damped"),
                       help="'pseudoinverse' is document-faithful exact minimum norm; "
                            "'damped' uses the explicitly regularized historical solve.")
    mintflow_g.add_argument("--mintflow-correction-scale", type=float, default=1.0,
                       help="Scale gamma in delta=-gamma*A^dagger*H; must lie in [0, 1].")
    mintflow_g.add_argument("--mintflow-pseudoinverse-rcond", type=float, default=None,
                       help="Optional relative singular-value cutoff for pseudoinverse mode.")
    mintflow_g.add_argument("--mintflow-adjoint-chunk-size", type=int, default=64,
                       help="Constraint-Jacobian rows propagated together during the discrete "
                            "adjoint. Larger values improve GPU throughput but use more memory.")
    mintflow_g.add_argument("--mintflow-save-mathematical-diagnostics", action="store_true",
                       help="Run expensive candidate-realization and linearization checks. "
                            "Core per-sample sampler diagnostics are saved without this flag.")
    mintflow_g.add_argument("--mintflow-time-penalty",  type=float, default=0.01,
                       help="Weight on (T - s_k)^2 in the candidate score "
                            "||delta_k||^2 + time_penalty * (T - s_k)^2.")
    mintflow_g.add_argument("--mintflow-forward-steps", type=int, default=200,
                       help="Euler steps for the forward solve + backward adjoint grid.")
    mintflow_g.add_argument("--mintflow-score-mode", default="default",
                       choices=("default", "normalized", "interior_penalty", "residual_predictive"),
                       help="Candidate scoring mode.")
    mintflow_g.add_argument("--mintflow-interior-penalty", type=float, default=0.0,
                       help="Boundary penalty weight for interior_penalty score mode.")
    mintflow_g.add_argument("--mintflow-correction-weight", type=float, default=1.0,
                       help="Correction-norm weight for residual_predictive score mode.")
    mintflow_g.add_argument("--mintflow-candidate-range", default="0.1-0.98",
                       help="Candidate flow-time range t_min-t_max (e.g. 0.2-0.8). "
                            "The validated Heat production default is 0.1-0.98.")
    mintflow_g.add_argument(
        "--time-sampling", "--mintflow-time-sampling", dest="mintflow_time_sampling",
        choices=("end_biased", "uniform"), default="end_biased",
        help=("Candidate-time schedule. end_biased (default) applies the smooth "
              "power warp t=t_min+(t_max-t_min)*(1-(1-u)^p); uniform retains "
              "the historical evenly spaced grid."),
    )
    mintflow_g.add_argument(
        "--end-bias-power", "--mintflow-end-bias-power", dest="mintflow_end_bias_power",
        type=float, default=2.0,
        help="Power p>1 for --time-sampling end_biased (default: 2.0).",
    )

    # ---- Phase 3B: shared noise / flow paths / diagnostics ----------------
    p3b_g = p.add_argument_group("Phase 3B options (opt-in; default behavior unchanged)")
    p3b_g.add_argument("--initial-noise-path", default=None, metavar="PATH",
                       help="Path to save shared initial noise (with --save-initial-noise) "
                            "or directory containing initial_noise.npy.")
    p3b_g.add_argument("--save-initial-noise", action="store_true",
                       help="Generate and save one shared initial-noise pool before sampling.")
    p3b_g.add_argument("--reuse-initial-noise", default=None, metavar="PATH",
                       help="Load and reuse shared initial noise from PATH (file or directory).")
    p3b_g.add_argument("--save-flow-paths", action="store_true",
                       help="Save intermediate flow-path snapshots to flow_paths.npz.")
    p3b_g.add_argument("--flow-path-fractions", default="0.0,0.25,0.5,0.75,1.0",
                       help="Comma-separated flow-time fractions for path snapshots.")
    p3b_g.add_argument("--max-path-samples", type=int, default=32,
                       help="Maximum number of samples whose flow paths are recorded.")
    p3b_g.add_argument("--save-sampler-diagnostics", action="store_true",
                       help="Save per-sample PCFM/ECI diagnostics in metadata.json.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate_method_flags(args: argparse.Namespace) -> None:
    if args.intermediate_only and args.method not in {"pcfm", "eci"}:
        raise ValueError("--intermediate-only is only valid for --method pcfm or eci.")
    if args.final_projection and args.method in {"pcfm", "eci"}:
        raise ValueError(
            "--final-projection is redundant for standard PCFM/ECI and is no longer "
            "supported. Use the standard sampler, or --intermediate-only for the "
            "terminal-intervention ablation."
        )


def main(args: argparse.Namespace) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.final_projection is None:
        args.final_projection = args.method == "mintflow"
    _validate_method_flags(args)

    checkpoint_path = args.checkpoint or default_checkpoint_for_dataset(args.dataset)
    config_path     = args.config     or default_config_for_dataset(args.dataset)

    # Seed before loading data because Heat samples are synthesized on demand
    # with Python's stdlib RNG; RD1D loads fixed HDF5 arrays.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    _print_banner(args, checkpoint_path, config_path)

    # ------------------------------------------------------------------ W&B
    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import os

            import wandb
            os.environ["WANDB_MODE"] = args.wandb_mode
            wandb_run = wandb.init(
                project="pcfm_sampling",
                name=f"{args.dataset}_{args.method}_{ts}",
                config=vars(args),
                mode=args.wandb_mode,
            )
        except Exception as exc:
            print(f"  [W&B] init failed ({exc}), continuing without logging.")

    # ------------------------------------------------------------------ model
    print("Loading model ...")
    model, config, ckpt_meta = load_ffm_model(
        args.dataset,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=args.device,
    )
    sample_dims = list(config.sample_dims)
    vis_kwargs  = get_vis_kwargs(config)
    print(f"  sample_dims : {sample_dims}")
    print(f"  ckpt step   : {ckpt_meta.get('checkpoint_step')}\n")

    # ------------------------------------------------------------------ test set
    # Test data are needed only when they determine the requested sample count.
    # Production global-task evaluation uses the frozen task target directly.
    need_test_set = args.num_samples == "same-as-test"
    test_set = None
    if need_test_set:
        print("Loading test split ...")
        test_set, _, _ = load_dataset_split(args.dataset, "test", config_path)
        print(f"  test set size: {len(test_set)}\n")

    if args.num_samples == "same-as-test":
        if test_set is None:
            raise RuntimeError("--num-samples same-as-test requires the test split.")
        num_samples = len(test_set)
    else:
        try:
            num_samples = int(args.num_samples)
        except ValueError:
            raise ValueError(
                f"--num-samples must be a positive integer or 'same-as-test', "
                f"got '{args.num_samples}'"
            )
    if args.sample_offset < 0:
        raise ValueError("--sample-offset must be non-negative")
    if args.sample_offset and not args.reuse_initial_noise:
        raise ValueError("--sample-offset requires --reuse-initial-noise")

    # ------------------------------------------------------- Scenario-2 target bank
    scenario_target = None
    scenario_target_path = None
    if args.constraint_scenario == "conservation_ic_bc":
        from sampling.scenario_targets import load_scenario_target_bank

        if not args.constraint_target_file:
            raise ValueError(
                "Scenario 2 requires --constraint-target-file pointing to a "
                "paired Vanilla-FFM target bank"
            )
        scenario_target_path = Path(args.constraint_target_file).resolve()
        scenario_target = load_scenario_target_bank(
            scenario_target_path, args.dataset
        )
        scenario_target.validate_slice(args.sample_offset, num_samples)
        target_checkpoint = Path(
            scenario_target.metadata.get("checkpoint_path", "")
        ).resolve()
        if target_checkpoint != Path(checkpoint_path).resolve():
            raise ValueError(
                "Scenario target bank was generated by a different checkpoint: "
                f"{target_checkpoint} != {Path(checkpoint_path).resolve()}"
            )
        if int(scenario_target.metadata.get("source_seed", -1)) != int(
            args.constraint_target_seed
        ):
            raise ValueError(
                "Scenario target seed does not match --constraint-target-seed"
            )
        print(f"Loaded paired Scenario-2 target bank: {scenario_target_path}")
        print(f"  target_id   : {scenario_target.target_id}")
        print(f"  target slice: [{args.sample_offset}:{args.sample_offset + num_samples}]")
        print(
            "  BC type     : "
            f"{scenario_target.metadata['physical_specification']['boundary_type']}\n"
        )
    # The test split can determine ``num_samples`` but is never a constraint
    # target. Scenario 2 obtains its paired IC/BC targets exclusively from
    # the persisted Vanilla-FFM target bank.

    # ------------------------------------------------------------------ global constraint task
    target_sample = None
    boundary_target = None
    if args.constraint_scenario == "conservation_ic_bc":
        start, stop = args.sample_offset, args.sample_offset + num_samples
        # Only the initial column is physically constrained. Constructing this
        # sparse reference tensor avoids duplicating the full Vanilla bank.
        target_sample = torch.zeros(
            (num_samples, *sample_dims), dtype=torch.float32
        )
        target_sample[:, :, 0] = torch.from_numpy(
            scenario_target.initial_conditions[start:stop]
        )
        if scenario_target.boundary_values.shape[1]:
            boundary_target = torch.from_numpy(
                scenario_target.boundary_values[start:stop]
            )

    task = _resolve_task(
        args,
        config,
        config_path,
        args.device,
        reference_sample=target_sample,
        boundary_target=boundary_target,
        scenario_target_metadata=(scenario_target.metadata if scenario_target else None),
    )
    print(f"Constraint task: '{task.name}' (dataset={task.dataset}, "
          f"scenario={args.constraint_scenario}, output_dim={task.output_dim})")
    print(f"  target_id   : {task.metadata.get('target_id')}")
    print(f"  target size : {task.target.numel() if task.target is not None else 0}")
    print(f"  native proj.: {task.native_projector is not None}\n")
    # Run-dir "extra" slug: prevents different task/target ids from colliding
    # when every other naming component (checkpoint/seed/n_eval/num_samples)
    # is identical.
    run_dir_extra = None
    if task is not None:
        target_id = str(task.metadata.get("target_id", "notarget")).replace("/", "_")
        run_dir_extra = f"task_{task.name}_{target_id}"
    if args.sample_offset:
        offset_slug = f"offset_{args.sample_offset}"
        run_dir_extra = (
            f"{run_dir_extra}_{offset_slug}" if run_dir_extra else offset_slug
        )

    # ------------------------------------------------------------------ run dir (early when Phase 3B outputs needed)
    from sampling.flow_paths import parse_flow_path_fractions
    from sampling.noise import generate_initial_noise, load_initial_noise, save_initial_noise

    flow_fractions = parse_flow_path_fractions(args.flow_path_fractions)
    need_early_run_dir = args.save_flow_paths or args.save_initial_noise
    run_dir = None
    if need_early_run_dir:
        run_dir = make_sampling_run_dir(
            output_dir=args.output_dir,
            dataset=args.dataset,
            method=args.method,
            checkpoint=checkpoint_path,
            seed=args.seed,
            n_eval=args.n_eval if args.method == "vanilla" else None,
            num_samples=num_samples,
            timestamp=ts,
            extra=run_dir_extra,
        )

    # ------------------------------------------------------------------ shared initial noise
    initial_noise = None
    initial_noise_path_str = None
    if args.reuse_initial_noise:
        initial_noise, _noise_meta = load_initial_noise(args.reuse_initial_noise)
        initial_noise_path_str = str(Path(args.reuse_initial_noise).resolve())
        stop = args.sample_offset + num_samples
        if stop > len(initial_noise):
            raise ValueError(
                "Requested shared-noise slice "
                f"[{args.sample_offset}:{stop}] exceeds pool size {len(initial_noise)}"
            )
        initial_noise = initial_noise[args.sample_offset:stop].clone()
        print(f"Reusing shared initial noise from {initial_noise_path_str} "
              f"slice [{args.sample_offset}:{stop}] "
              f"(shape {tuple(initial_noise.shape)})\n")
    elif args.save_initial_noise:
        if args.sample_offset:
            raise ValueError("Cannot combine --save-initial-noise with a nonzero offset")
        initial_noise = generate_initial_noise(
            model, num_samples, sample_dims, args.device, args.seed,
        )
        noise_save_path = args.initial_noise_path or (run_dir or Path(args.output_dir) / args.dataset)
        saved = save_initial_noise(
            noise_save_path,
            initial_noise,
            metadata={
                "dataset": args.dataset,
                "num_samples": num_samples,
                "sample_dims": sample_dims,
                "seed": args.seed,
            },
        )
        initial_noise_path_str = str(saved.resolve())
        print(f"Saved shared initial noise to {initial_noise_path_str}\n")

    if args.save_flow_paths and initial_noise is None:
        raise ValueError("--save-flow-paths requires shared initial noise "
                         "(use --save-initial-noise or --reuse-initial-noise).")

    instrumentation_kwargs = dict(
        initial_noise=initial_noise,
        initial_noise_path=initial_noise_path_str,
        save_flow_paths=args.save_flow_paths,
        flow_path_fractions=flow_fractions,
        max_path_samples=args.max_path_samples,
        run_dir=str(run_dir) if run_dir is not None else None,
    )
    pcfm_eci_kwargs = {
        **instrumentation_kwargs,
        "save_sampler_diagnostics": (args.save_sampler_diagnostics or args.save_constraint_diagnostics),
    }

    # ------------------------------------------------------------------ generate
    from sampling.methods import (
        run_dflow_sampling,
        run_diffusionpde_sampling,
        run_eci_sampling,
        run_final_projection_sampling,
        run_mintflow_sampling,
        run_pcfm_sampling,
        run_vanilla_sampling,
    )

    print(f"Generating {num_samples} samples via --method {args.method} ...")
    cuda_timing = args.device.startswith("cuda") and torch.cuda.is_available()
    if cuda_timing:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    vector_field_forward_calls = [0]
    def _count_vector_field_call(_module, _inputs, _output):
        vector_field_forward_calls[0] += 1
    vector_field_hook = model.model.register_forward_hook(_count_vector_field_call)
    t_gen_start = time.perf_counter()

    if args.method == "vanilla":
        samples, method_meta = run_vanilla_sampling(
            model, config,
            num_samples=num_samples,
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            **instrumentation_kwargs,
        )

    elif args.method == "pcfm":
        samples, method_meta = run_pcfm_sampling(
            model, config,
            num_samples=num_samples,
            n_step=args.pcfm_n_step,
            newton_steps=args.pcfm_newton_steps,
            tol=args.pcfm_tol,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            task=task,
            intermediate_only=args.intermediate_only,
            **pcfm_eci_kwargs,
        )

    elif args.method == "eci":
        eci_projector_config = None
        if task is not None:
            eci_projector_config = _projector_config_from_args(args, task.scales)
            if args.eci_projector_max_iter is not None:
                eci_projector_config.max_iter = int(args.eci_projector_max_iter)
            # ECI calls this projector at every flow/mixing step. A second
            # full Jacobian plus SVD solely for per-step conditioning
            # diagnostics dominates nonlinear RD1D runtime but does not
            # affect the projection update itself.
            eci_projector_config.compute_condition_number = False
        samples, method_meta = run_eci_sampling(
            model, config,
            num_samples=num_samples,
            n_step=args.eci_n_step,
            n_mix=args.eci_n_mix,
            resample_every=args.eci_resample_every,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            task=task,
            eci_mode=args.eci_mode,
            projector_config=eci_projector_config,
            intermediate_only=args.intermediate_only,
            **pcfm_eci_kwargs,
        )

    elif args.method == "mintflow":
        from sampling.adjoint_correction import parse_candidate_time_range
        cand_t_min, cand_t_max = parse_candidate_time_range(args.mintflow_candidate_range)
        samples, method_meta = run_mintflow_sampling(
            model, config,
            num_samples=num_samples,
            num_candidates=args.mintflow_num_candidates,
            ridge=args.mintflow_ridge,
            time_penalty=args.mintflow_time_penalty,
            forward_steps=args.mintflow_forward_steps,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            score_mode=args.mintflow_score_mode,
            interior_penalty=args.mintflow_interior_penalty,
            correction_weight=args.mintflow_correction_weight,
            candidate_t_min=cand_t_min,
            candidate_t_max=cand_t_max,
            time_sampling=args.mintflow_time_sampling,
            end_bias_power=args.mintflow_end_bias_power,
            correction_mode=args.mintflow_correction_mode,
            correction_scale=args.mintflow_correction_scale,
            pseudoinverse_rcond=args.mintflow_pseudoinverse_rcond,
            adjoint_chunk_size=args.mintflow_adjoint_chunk_size,
            save_sampler_diagnostics=(args.save_sampler_diagnostics or args.save_constraint_diagnostics),
            save_mathematical_diagnostics=args.mintflow_save_mathematical_diagnostics,
            task=task,
            **instrumentation_kwargs,
        )

    elif args.method == "diffusionpde":
        samples, method_meta = run_diffusionpde_sampling(
            model, config, num_samples=num_samples,
            n_step=args.guided_n_step, eta=args.diffusionpde_eta,
            batch_size=args.batch_size, device=args.device, seed=args.seed,
            task=task, **instrumentation_kwargs,
        )

    elif args.method == "dflow":
        samples, method_meta = run_dflow_sampling(
            model, config, num_samples=num_samples,
            n_step=args.guided_n_step, n_iter=args.dflow_n_iter, lr=args.dflow_lr,
            batch_size=args.batch_size, device=args.device, seed=args.seed,
            task=task, **instrumentation_kwargs,
        )

    elif args.method == "mintflow_project":
        if task is None:
            raise RuntimeError(
                "--method mintflow_project requires --task (a terminal projection needs an "
                "explicit target manifold). Use --method mintflow without terminal projection, "
                "or --method final_projection for a projection-only baseline."
            )
        from sampling.adjoint_correction import parse_candidate_time_range
        from sampling.methods import run_mintflow_project_sampling
        cand_t_min, cand_t_max = parse_candidate_time_range(args.mintflow_candidate_range)
        samples, method_meta = run_mintflow_project_sampling(
            model, config,
            num_samples=num_samples,
            num_candidates=args.mintflow_num_candidates,
            ridge=args.mintflow_ridge,
            time_penalty=args.mintflow_time_penalty,
            forward_steps=args.mintflow_forward_steps,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            score_mode=args.mintflow_score_mode,
            interior_penalty=args.mintflow_interior_penalty,
            correction_weight=args.mintflow_correction_weight,
            candidate_t_min=cand_t_min,
            candidate_t_max=cand_t_max,
            time_sampling=args.mintflow_time_sampling,
            end_bias_power=args.mintflow_end_bias_power,
            correction_mode=args.mintflow_correction_mode,
            correction_scale=args.mintflow_correction_scale,
            pseudoinverse_rcond=args.mintflow_pseudoinverse_rcond,
            adjoint_chunk_size=args.mintflow_adjoint_chunk_size,
            save_sampler_diagnostics=(args.save_sampler_diagnostics or args.save_constraint_diagnostics),
            task=task,
            projector_config=_projector_config_from_args(args, task.scales),
            **instrumentation_kwargs,
        )

    elif args.method == "final_projection":
        projector_config = _projector_config_from_args(
            args, task.scales if task is not None else None,
        )
        task_for_run = _projection_task_for_samples(task)
        samples, method_meta = run_final_projection_sampling(
            model, config,
            num_samples=num_samples,
            task=task_for_run,
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            projector_config=projector_config,
            save_sampler_diagnostics=(args.save_sampler_diagnostics or args.save_constraint_diagnostics),
            **instrumentation_kwargs,
        )

    else:
        raise ValueError(f"Unknown method '{args.method}'")

    # Uniform optional post-sampling projection. Dedicated compatibility
    # methods already perform this operation internally and are not projected
    # a second time.
    if args.method not in {"final_projection", "mintflow_project"}:
        from sampling.methods import apply_optional_final_projection
        projection_task = (
            _projection_task_for_samples(task)
            if args.final_projection
            else task
        )
        samples, method_meta = apply_optional_final_projection(
            samples,
            method_meta,
            projection_task,
            enabled=args.final_projection,
            config=(
                _projector_config_from_args(
                    args,
                    projection_task[0].scales
                    if isinstance(projection_task, (list, tuple))
                    else projection_task.scales,
                )
                if args.final_projection
                else None
            ),
            save_per_sample=(args.save_sampler_diagnostics or args.save_constraint_diagnostics),
        )
    else:
        method_meta["post_sampling_final_projection_enabled"] = True
        method_meta["post_sampling_final_projection_count"] = 1

    vector_field_hook.remove()
    if cuda_timing:
        torch.cuda.synchronize(args.device)
    gen_time = time.perf_counter() - t_gen_start
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(args.device)) if cuda_timing else None
    )
    peak_gpu_memory_reserved_bytes = (
        int(torch.cuda.max_memory_reserved(args.device)) if cuda_timing else None
    )
    print(f"  Generated shape={tuple(samples.shape)}  time={gen_time:.1f}s\n")

    # ------------------------------------------------------------------ output dir
    if run_dir is None:
        run_dir = make_sampling_run_dir(
            output_dir=args.output_dir,
            dataset=args.dataset,
            method=args.method,
            checkpoint=checkpoint_path,
            seed=args.seed,
            n_eval=args.n_eval if args.method == "vanilla" else None,
            num_samples=num_samples,
            timestamp=ts,
            extra=run_dir_extra,
        )
    print(f"Output directory: {run_dir}\n")

    # ------------------------------------------------------------------ save samples
    samples_fname = "samples.npy" if args.save_format == "npy" else "samples.h5"
    samples_path  = run_dir / samples_fname
    save_samples(samples_path, samples, format=args.save_format)
    print(f"Saved {samples_path.name}  ({_fmt_size(samples_path)})")

    # ------------------------------------------------------------------ metadata
    base_meta = {
        "task":                 "ffm_sampling",
        "dataset":              args.dataset,
        "method":               args.method,
        "checkpoint_path":      str(checkpoint_path),
        "checkpoint_step":      ckpt_meta.get("checkpoint_step"),
        "checkpoint_type":      ckpt_meta.get("checkpoint_type"),
        "config_path":          str(config_path),
        "num_samples":          num_samples,
        "sample_offset":        args.sample_offset,
        "sample_ids":           list(range(args.sample_offset, args.sample_offset + num_samples)),
        "sample_shape":         list(samples.shape),
        "sample_dims":          sample_dims,
        "batch_size":           args.batch_size,
        "seed":                 args.seed,
        "device":               args.device,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "generation_time_s":    round(gen_time, 2),
        "timing_cuda_synchronized": cuda_timing,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_memory_reserved_bytes": peak_gpu_memory_reserved_bytes,
        "vector_field_forward_calls_total": vector_field_forward_calls[0],
        "vector_field_forward_calls_per_sample_amortized": (
            vector_field_forward_calls[0] / num_samples if num_samples else None
        ),
        "vector_field_forward_calls_per_batch_amortized": (
            vector_field_forward_calls[0] / ((num_samples + args.batch_size - 1) // args.batch_size)
            if num_samples else None
        ),
        "samples_file":         samples_fname,
        "save_format":          args.save_format,
        "constraint_scenario":  args.constraint_scenario,
        "constraint_target_file": (
            str(scenario_target_path) if scenario_target_path is not None else None
        ),
        "constraint_target_seed": (
            int(scenario_target.metadata["source_seed"])
            if args.constraint_scenario == "conservation_ic_bc" else None
        ),
        "constraint_target_id": (
            scenario_target.target_id if scenario_target is not None else None
        ),
        "evaluate_constraints": args.evaluate_constraints,
        "output_dir":           str(run_dir),
        "timestamp":            ts,
        "git_commit":           get_git_commit(),
        # Global constraint task CLI provenance.
        "task_arg":                    args.task,
        "constraint_projector_arg":    args.constraint_projector,
        "constraint_tolerance_arg":    args.constraint_tolerance,
        "constraint_ridge_arg":        args.constraint_ridge,
        "constraint_max_iter_arg":     args.constraint_max_iter,
        "constraint_normalization_arg": args.constraint_normalization,
        "eci_mode_arg":                args.eci_mode,
        "final_projection_arg":        args.final_projection,
        "intermediate_only_arg":       args.intermediate_only,
    }
    # Merge method-specific metadata
    if args.method == "pcfm":
        base_meta["pcfm_n_step"]      = args.pcfm_n_step
        base_meta["pcfm_newton_steps"]= args.pcfm_newton_steps
        base_meta["pcfm_tol"]         = args.pcfm_tol
    elif args.method == "eci":
        base_meta["eci_n_step"]        = args.eci_n_step
        base_meta["eci_n_mix"]         = args.eci_n_mix
        base_meta["eci_resample_every"]= args.eci_resample_every
        base_meta["eci_constraints"]   = [f"TaskProjectionConstraint({args.eci_mode})"]
    elif args.method in ("mintflow", "mintflow_project"):
        base_meta["mintflow_num_candidates"]  = args.mintflow_num_candidates
        base_meta["mintflow_ridge"]           = args.mintflow_ridge
        base_meta["mintflow_time_penalty"]    = args.mintflow_time_penalty
        base_meta["mintflow_forward_steps"]   = args.mintflow_forward_steps
        base_meta["mintflow_score_mode"]      = args.mintflow_score_mode
        base_meta["mintflow_interior_penalty"] = args.mintflow_interior_penalty
        base_meta["mintflow_correction_weight"] = args.mintflow_correction_weight
        base_meta["mintflow_candidate_range"] = args.mintflow_candidate_range
        base_meta["mintflow_time_sampling"] = args.mintflow_time_sampling
        base_meta["mintflow_end_bias_power"] = args.mintflow_end_bias_power
        base_meta["mintflow_correction_mode"] = args.mintflow_correction_mode
        base_meta["mintflow_correction_scale"] = args.mintflow_correction_scale
        base_meta["mintflow_pseudoinverse_rcond"] = args.mintflow_pseudoinverse_rcond
    elif args.method == "diffusionpde":
        base_meta["guided_n_step"] = args.guided_n_step
        base_meta["diffusionpde_eta"] = args.diffusionpde_eta
    elif args.method == "dflow":
        base_meta["guided_n_step"] = args.guided_n_step
        base_meta["dflow_n_iter"] = args.dflow_n_iter
        base_meta["dflow_lr"] = args.dflow_lr
    elif args.method in ("vanilla", "final_projection"):
        base_meta["n_eval"] = args.n_eval

    # Full ConstraintTask provenance (raw/normalized target, scales, native-projector
    # availability, reference-pool hash/target_id via task.metadata) for *every*
    # task-mode run, including plain vanilla+task (used as the "no correction" row
    # in comparisons). pcfm/eci/mintflow/final_projection already put the same block in
    # method_meta; the merge below is then just a harmless no-op overwrite for those.
    if task is not None:
        from sampling.methods import _task_metadata
        base_meta.update(_task_metadata(task))

    metadata = {**base_meta, **method_meta}
    save_metadata(run_dir / "metadata.json", metadata)
    print("Saved metadata.json")

    # ------------------------------------------------------------------ previews
    preview_dir = run_dir / "previews"
    if args.num_preview > 0:
        preview_dir.mkdir(exist_ok=True)
        n_prev = min(args.num_preview, num_samples)
        print(f"\nSaving {n_prev} preview image(s) ...")
        for i in range(n_prev):
            fname = preview_dir / f"sample_{i:04d}.png"
            save_sample_image(samples[i], fname, vis_kwargs=vis_kwargs)
        print(f"  Saved to previews/ ({n_prev} PNG files)")

    # ------------------------------------------------------------------ constraint eval
    agg_metrics = None
    if args.evaluate_constraints:
        agg_metrics = _run_global_task_evaluation(
            args, samples, task, metadata, run_dir
        )

    # ------------------------------------------------------------------ W&B
    if wandb_run is not None:
        _log_to_wandb(wandb_run, agg_metrics, preview_dir, args.num_preview, num_samples)
        wandb_run.finish()

    # ------------------------------------------------------------------ summary
    _print_summary(args, samples, gen_time, run_dir, agg_metrics, method_meta=method_meta)


def _run_global_task_evaluation(args, samples, task, metadata, run_dir):
    """Write aggregate and per-sample metrics for every task component."""
    print(f"\nEvaluating global constraint task '{task.name}' (target_id="
          f"{task.metadata.get('target_id')!r}) ...")
    n_to_eval = len(samples) if args.max_eval_samples is None else min(args.max_eval_samples, len(samples))
    eval_samples = samples[:n_to_eval].numpy() if hasattr(samples, "numpy") else np.asarray(samples[:n_to_eval])
    t_eval = time.time()
    residual_rows = []
    raw_components: dict[str, list[np.ndarray]] = {
        name: [] for name in task.component_names
    }
    for index, sample in enumerate(eval_samples):
        this_task = task.task_for(index) if hasattr(task, "task_for") else task
        value = torch.as_tensor(
            sample, device=this_task.device, dtype=this_task.scales.dtype
        )
        residual_rows.append(
            this_task.normalized_residual(value.reshape(-1)).detach().cpu().numpy()
        )
        if this_task.eval_residual is not None:
            raw = this_task.eval_residual(sample[None, ...])
            for name in raw_components:
                raw_components[name].append(np.asarray(raw[name])[0])
    normalized = np.stack(residual_rows).astype(np.float64)
    stacked_raw = {
        name: np.stack(values) for name, values in raw_components.items() if values
    }
    norms = np.linalg.norm(normalized, axis=1)
    infs = np.max(np.abs(normalized), axis=1)
    per_sample_rows = []
    for index in range(n_to_eval):
        per_sample_rows.append({
            "sample_index": index,
            "normalized_total_L2": float(norms[index]),
            "normalized_total_Linf": float(infs[index]),
        })

    component_results = {}
    offset = 0
    for name, dimension in zip(task.component_names, task.component_dims):
        block = normalized[:, offset : offset + dimension]
        block_l2 = np.linalg.norm(block, axis=1)
        block_linf = np.max(np.abs(block), axis=1)
        raw_block = np.asarray(stacked_raw.get(name), dtype=np.float64)
        if raw_block.shape != (n_to_eval, dimension):
            raise RuntimeError(
                f"Task component {name!r} has raw shape {raw_block.shape}, "
                f"expected {(n_to_eval, dimension)}"
            )
        raw_l2 = np.linalg.norm(raw_block, axis=1)
        raw_linf = np.max(np.abs(raw_block), axis=1)
        component_results[name] = {
            "dimension": int(dimension),
            "raw_L2": {
                "mean": float(np.mean(raw_l2)),
                "median": float(np.median(raw_l2)),
                "p95": float(np.quantile(raw_l2, 0.95)),
                "max": float(np.max(raw_l2)),
            },
            "raw_Linf": {
                "mean": float(np.mean(raw_linf)),
                "median": float(np.median(raw_linf)),
                "p95": float(np.quantile(raw_linf, 0.95)),
                "max": float(np.max(raw_linf)),
            },
            "normalized_L2": {
                "mean": float(np.mean(block_l2)),
                "median": float(np.median(block_l2)),
                "p95": float(np.quantile(block_l2, 0.95)),
                "max": float(np.max(block_l2)),
            },
            "normalized_Linf": {
                "mean": float(np.mean(block_linf)),
                "median": float(np.median(block_linf)),
                "p95": float(np.quantile(block_linf, 0.95)),
                "max": float(np.max(block_linf)),
            },
        }
        for index, row in enumerate(per_sample_rows):
            row[f"{name}_normalized_L2"] = float(block_l2[index])
            row[f"{name}_normalized_Linf"] = float(block_linf[index])
            row[f"{name}_raw_L2"] = float(raw_l2[index])
            row[f"{name}_raw_Linf"] = float(raw_linf[index])
        offset += dimension
    if offset != task.output_dim:
        raise RuntimeError(
            f"Task component dimensions sum to {offset}, expected output_dim={task.output_dim}"
        )
    result = {
        "task": task.name,
        "constraint_scenario": task.metadata.get("constraint_scenario"),
        "n_samples": n_to_eval,
        "components": component_results,
        "residuals": {
            "normalized_vector_residual_norm": {
                "mean": float(np.mean(norms)),
                "median": float(np.median(norms)),
                "p95": float(np.quantile(norms, 0.95)),
                "max": float(np.max(norms)),
            },
            "valid_at": {
                str(args.constraint_tolerance): float(np.mean(norms <= args.constraint_tolerance)),
            },
        },
    }
    result["evaluation_time_s"] = round(time.time() - t_eval, 2)
    result["method"] = args.method
    result["target_id"] = task.metadata.get("target_id")

    save_metrics_json(run_dir / "global_task_metrics.json", result)
    save_metrics_csv(run_dir / "global_task_metrics.csv", per_sample_rows)
    print("  Saved global_task_metrics.json + global_task_metrics.csv")
    print(f"  normalized_vector_residual_norm mean : {result['residuals']['normalized_vector_residual_norm']['mean']:.4g}")
    print(f"  valid_at                              : {result['residuals']['valid_at']}")
    return result


# ---------------------------------------------------------------------------
# W&B helper
# ---------------------------------------------------------------------------

def _log_to_wandb(wandb_run, agg_metrics, preview_dir, num_preview, num_samples):
    try:
        import wandb
        payload = {}
        if agg_metrics is not None:
            flat = {k: v for k, v in agg_metrics.items() if isinstance(v, (int, float))}
            payload.update(flat)
        if num_preview > 0 and preview_dir.exists():
            imgs = [
                wandb.Image(str(preview_dir / f"sample_{i:04d}.png"))
                for i in range(min(num_preview, num_samples))
                if (preview_dir / f"sample_{i:04d}.png").exists()
            ]
            if imgs:
                payload["preview_samples"] = imgs
        if payload:
            wandb_run.log(payload)
    except Exception as exc:
        print(f"  [W&B] logging failed ({exc}), skipping.")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_banner(args, checkpoint_path, config_path):
    print(f"\n{'='*60}")
    print(f"  PCFM Step 3/4 — FFM Sampling ({args.method.upper()})")
    print(f"{'='*60}")
    print(f"  dataset      : {args.dataset}")
    print(f"  method       : {args.method}")
    print(f"  checkpoint   : {checkpoint_path}")
    print(f"  config       : {config_path}")
    print(f"  num_samples  : {args.num_samples}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  seed         : {args.seed}")
    print(f"  device       : {args.device}")
    if args.method == "vanilla":
        print(f"  n_eval       : {args.n_eval}")
    elif args.method == "pcfm":
        print(f"  pcfm_n_step  : {args.pcfm_n_step}")
        print(f"  newton_steps : {args.pcfm_newton_steps}")
        print(f"  pcfm_tol     : {args.pcfm_tol}")
    elif args.method == "eci":
        print(f"  eci_n_step   : {args.eci_n_step}")
        print(f"  eci_n_mix    : {args.eci_n_mix}")
        print(f"  resample_ev  : {args.eci_resample_every}")
    elif args.method in ("mintflow", "mintflow_project"):
        print(f"  num_candid.  : {args.mintflow_num_candidates}")
        print(f"  ridge        : {args.mintflow_ridge}")
        print(f"  time_penalty : {args.mintflow_time_penalty}")
        print(f"  time_sampling: {args.mintflow_time_sampling}")
        if args.mintflow_time_sampling == "end_biased":
            print(f"  end_bias_pow.: {args.mintflow_end_bias_power}")
        print(f"  forward_steps: {args.mintflow_forward_steps}")
    elif args.method == "diffusionpde":
        print(f"  guided_steps : {args.guided_n_step}")
        print(f"  eta          : {args.diffusionpde_eta}")
    elif args.method == "dflow":
        print(f"  guided_steps : {args.guided_n_step}")
        print(f"  LBFGS iters  : {args.dflow_n_iter}")
        print(f"  LBFGS lr     : {args.dflow_lr}")
    print(f"  eval_constr  : {args.evaluate_constraints}")
    print(f"  wandb_mode   : {args.wandb_mode}")
    print(f"{'='*60}\n")


def _print_summary(args, samples, gen_time, run_dir, agg_metrics, method_meta=None):
    print(f"\n{'='*60}")
    print("  FFM Sampling — COMPLETE")
    print(f"{'='*60}")
    print(f"  dataset          : {args.dataset}")
    print(f"  method           : {args.method}")
    print(f"  samples shape    : {tuple(samples.shape)}")
    print(f"  generation time  : {gen_time:.1f}s")
    if args.method == "pcfm":
        print(f"  PCFM n_step      : {args.pcfm_n_step}")
        print(f"  PCFM newton steps: {args.pcfm_newton_steps}")
    elif args.method == "eci":
        print(f"  ECI n_step       : {args.eci_n_step}")
        print(f"  ECI n_mix        : {args.eci_n_mix}")
    elif args.method in ("mintflow", "mintflow_project"):
        print(f"  MintFlow num_candid.  : {args.mintflow_num_candidates}")
        print(f"  MintFlow forward_steps: {args.mintflow_forward_steps}")
        # mintflow_project nests MintFlow's own metadata under "mintflow_meta" (its top level
        # is the terminal-projection summary) -- read stats from there instead.
        mintflow_stats = method_meta.get("mintflow_meta", method_meta) if method_meta is not None else None
        if mintflow_stats is not None:
            print(f"  MintFlow mean s*      : {mintflow_stats.get('mintflow_s_star_mean', float('nan')):.4f}"
                  f"  (std={mintflow_stats.get('mintflow_s_star_std', float('nan')):.4f})")
            print(f"  MintFlow residual before -> after (raw H-norm, pre-eval): "
                  f"{mintflow_stats.get('mintflow_residual_before_mean', float('nan')):.4f} -> "
                  f"{mintflow_stats.get('mintflow_residual_after_mean', float('nan')):.4f}")
            print(f"  MintFlow correction norm mean: "
                  f"{mintflow_stats.get('mintflow_correction_norm_mean', float('nan')):.4f}"
                  f"  (std={mintflow_stats.get('mintflow_correction_norm_std', float('nan')):.4f})")
            mintflow_warnings = mintflow_stats.get("warnings", [])
            if mintflow_warnings:
                print(f"  MintFlow warnings     : {mintflow_warnings}")
        if args.method == "mintflow_project" and method_meta is not None:
            proj = method_meta.get("final_projection_summary", {})
            print(f"  Terminal projection residual after -> {proj.get('projector_method', '?')}: "
                  f"{proj.get('residual_before_mean', float('nan')):.4g} -> "
                  f"{proj.get('residual_after_mean', float('nan')):.4g}")
    if agg_metrics is not None:
        def _pfmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        scenario = agg_metrics.get("constraint_scenario", "?")
        normalized = agg_metrics.get("residuals", {}).get(
            "normalized_vector_residual_norm", {}
        )
        print(f"  --- Constraint residuals ({scenario}) ---")
        print(f"  normalized L2 mean: {_pfmt(normalized.get('mean', 'N/A'))}")
        print(f"  normalized L2 p95 : {_pfmt(normalized.get('p95', 'N/A'))}")
        for comp, comp_stats in agg_metrics.get("components", {}).items():
            raw_linf = comp_stats.get("raw_Linf", {})
            print(
                f"  {comp:20s} raw Linf mean: "
                f"{_pfmt(raw_linf.get('mean', 'N/A'))}"
            )
    print(f"  output dir       : {run_dir}")
    print(f"{'='*60}\n")


def _fmt_size(path: Path) -> str:
    sz = path.stat().st_size
    if sz >= 1_000_000:
        return f"{sz / 1e6:.1f} MB"
    if sz >= 1_000:
        return f"{sz / 1e3:.1f} KB"
    return f"{sz} B"


if __name__ == "__main__":
    main(_build_parser().parse_args())
