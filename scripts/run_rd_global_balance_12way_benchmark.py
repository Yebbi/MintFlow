#!/usr/bin/env python3
"""Run the paired 14-configuration RD1D/global_balance benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.pipeline_common import (
    benchmark_configurations,
    find_completed_run,
    run_logged,
)

RUN = REPO / "scripts" / "run_sampling.py"
EVALUATE = REPO / "scripts" / "evaluate_type1_benchmark.py"
ANALYZE = REPO / "scripts" / "analyze_type1_benchmark.py"
PHYSICAL = REPO / "scripts" / "evaluate_physical_consistency.py"
PROJECT_COMPLETED = REPO / "scripts" / "project_completed_sampling_run.py"

CONFIGURATIONS = benchmark_configurations(vanilla_n_eval=200)


def _ensure_target_bank(path: Path, vanilla_run: Path, args) -> None:
    from sampling.scenario_targets import (
        derive_scenario_target_bank,
        load_scenario_target_bank,
        save_scenario_target_bank,
    )

    if path.exists():
        bank = load_scenario_target_bank(path, "rd1d")
        if bank.num_targets != args.num_samples:
            raise ValueError("Existing Scenario-2 target-bank size is incompatible")
        return
    from scripts.training.utils import load_config

    samples_path = vanilla_run / "samples.npy"
    samples = np.load(samples_path, mmap_mode="r")
    bank = derive_scenario_target_bank(
        "rd1d", samples, load_config("configs/rd1d.yml"), seed=args.seed,
        checkpoint_path="logs/rd1d/latest.pt", config_path="configs/rd1d.yml",
        n_eval=200, source_samples_path=str(samples_path),
    )
    save_scenario_target_bank(path, bank)


def main(args: argparse.Namespace) -> None:
    if args.diffusionpde_eta is None:
        # The IC and Neumann blocks make the Scenario-2 guidance gradient much
        # stiffer than the conservation-only objective.  eta=1 diverges on the
        # paired benchmark; 1e-3 is the validated stable RD1D setting.
        args.diffusionpde_eta = (
            1e-3 if args.constraint_scenario == "conservation_ic_bc" else 1.0
        )
    default_output = (
        "results/rd_global_balance_scenario2_benchmark"
        if args.constraint_scenario == "conservation_ic_bc"
        else "results/rd_global_balance_12way_benchmark"
    )
    root = Path(args.output_dir or default_output).resolve()
    raw, logs, noise = root / "raw", root / "logs", root / "shared_noise"
    root.mkdir(parents=True, exist_ok=True)
    if args.reference_samples != args.num_samples:
        raise ValueError("Production protocol requires reference_samples == num_samples")
    runs: dict[str, str] = {}
    target_seed = args.seed
    if args.constraint_target_seed not in (None, args.seed):
        raise ValueError("Paired Scenario-2 targets must use the benchmark Vanilla seed")
    target_file = root / "scenario_target" / f"rd1d_vanilla_bank_seed{target_seed}_n{args.num_samples}.npz"
    numerical_reference_path = root / "reference" / "numerical_reference_samples.npy"
    if not args.dry_run and args.constraint_scenario == "conservation_only":
        from eval_pde.benchmark_distributions import load_or_generate_ground_truth_reference
        from eval_pde.config import build_rd1d_config
        reference, _ = load_or_generate_ground_truth_reference(
            root / "reference",
            build_rd1d_config(task="global_balance", config_path="configs/rd1d.yml"),
            num_samples=args.reference_samples,
            seed=args.reference_seed,
        )
        if len(reference) != args.num_samples:
            raise ValueError("Production protocol requires reference_samples == num_samples")

    for position, spec in enumerate(CONFIGURATIONS):
        if args.configuration and spec.label not in args.configuration:
            continue
        label_root = raw / spec.label
        run_scenario = (
            "conservation_only" if spec.method == "vanilla"
            else args.constraint_scenario
        )
        if (
            not args.dry_run
            and args.constraint_scenario == "conservation_ic_bc"
            and spec.method != "vanilla" and not target_file.exists()
        ):
            vanilla_run = find_completed_run(
                raw / "vanilla_ffm", args.seed, args.num_samples,
                {"constraint_scenario": "conservation_only"},
            )
            if vanilla_run is None:
                raise RuntimeError("Run Vanilla FFM before Scenario-2 constrained methods")
            _ensure_target_bank(target_file, vanilla_run, args)
        required_metadata = {
            "constraint_scenario": run_scenario,
            "constraint_target_file": (
                str(target_file.resolve())
                if run_scenario == "conservation_ic_bc" else None
            ),
            "constraint_target_seed": (
                target_seed if run_scenario == "conservation_ic_bc" else None
            ),
        }
        if run_scenario == "conservation_ic_bc":
            # 99 external-flux balance + 128 IC + 2*99 positive-time BC.
            # This rejects artifacts from the former incompatible 427-D
            # formulation even when every other run identifier matches.
            required_metadata["task_output_dim"] = 425
        required_metadata.update(
            {
                "mintflow_time_sampling": args.time_sampling,
                "mintflow_end_bias_power": args.end_bias_power,
                "mintflow_candidate_range": f"0.1-{args.mintflow_terminal_cutoff}",
                "mintflow_candidate_t_min": 0.1,
                "mintflow_candidate_t_max": args.mintflow_terminal_cutoff,
                "mintflow_forward_steps": args.mintflow_steps,
                "mintflow_num_candidates": args.mintflow_candidates,
                "mintflow_time_penalty": args.mintflow_time_penalty,
                "mintflow_correction_scale": args.mintflow_correction_scale,
                "mintflow_correction_mode": (
                    "damped" if "damped" in spec.label else "pseudoinverse"
                ),
                "mintflow_adjoint_chunk_size": args.mintflow_adjoint_chunk_size,
                "final_projection_arg": spec.label.endswith("_with_final_projection"),
                "post_sampling_final_projection_enabled": spec.label.endswith(
                    "_with_final_projection"
                ),
            }
            if spec.method == "mintflow" else
            {"guided_n_step": args.euler_steps, "diffusionpde_eta": args.diffusionpde_eta}
            if spec.method == "diffusionpde" else
            {"guided_n_step": args.euler_steps, "dflow_n_iter": args.dflow_n_iter, "dflow_lr": args.dflow_lr}
            if spec.method == "dflow" else {}
        )
        complete = find_completed_run(
            label_root, args.seed, args.num_samples, required_metadata,
        )
        if complete is not None and not args.rerun:
            print(f"SKIP complete {spec.label}: {complete}", flush=True)
            runs[spec.label] = str(complete)
            if args.constraint_scenario == "conservation_ic_bc" and spec.method == "vanilla":
                _ensure_target_bank(target_file, complete, args)
            continue
        if spec.label.endswith("_with_final_projection"):
            source_label = spec.label.replace("_with_final_projection", "_no_final_projection")
            source_required_metadata = dict(required_metadata)
            source_required_metadata.update({
                "final_projection_arg": False,
                "post_sampling_final_projection_enabled": False,
            })
            source = find_completed_run(
                raw / source_label, args.seed, args.num_samples, source_required_metadata,
            )
            if source is not None:
                run_logged([
                    args.python, str(PROJECT_COMPLETED),
                    "--source-run", str(source), "--output-root", str(label_root),
                    "--dataset", "rd1d", "--task", "global_balance",
                    "--config", "configs/rd1d.yml", "--device", args.device,
                    "--max-iter", str(args.final_projector_max_iter),
                    "--ridge", str(args.projector_ridge),
                    "--tolerance", str(args.projector_tolerance),
                ], logs / f"{spec.label}.log", args.dry_run)
                if not args.dry_run:
                    complete = find_completed_run(
                        label_root, args.seed, args.num_samples, required_metadata,
                    )
                    if complete is None:
                        raise RuntimeError(f"Derived final-projection run missing for {spec.label}")
                    runs[spec.label] = str(complete)
                continue
        common = [
            args.python, str(RUN), "--dataset", "rd1d", "--method", spec.method,
            "--config", "configs/rd1d.yml", "--checkpoint", "logs/rd1d/latest.pt",
            "--num-samples", str(args.num_samples), "--batch-size", str(args.batch_size),
            "--seed", str(args.seed), "--device", args.device, "--task", "global_balance",
            "--constraint-scenario", run_scenario,
            "--evaluate-constraints", "--save-constraint-diagnostics", "--save-sampler-diagnostics",
            "--output-dir", str(label_root), "--num-preview", "0", "--wandb-mode", "disabled",
            "--pcfm-n-step", str(args.euler_steps), "--pcfm-newton-steps", "1",
            "--pcfm-tol", str(args.projector_ridge),
            "--eci-n-step", str(args.euler_steps), "--eci-n-mix", "1", "--eci-resample-every", "0",
            "--eci-projector-max-iter", "1",
            "--guided-n-step", str(args.euler_steps),
            "--diffusionpde-eta", str(args.diffusionpde_eta),
            "--dflow-n-iter", str(args.dflow_n_iter),
            "--dflow-lr", str(args.dflow_lr),
            "--mintflow-forward-steps", str(args.mintflow_steps),
            "--mintflow-num-candidates", str(args.mintflow_candidates), "--mintflow-ridge", str(args.mintflow_ridge),
            "--mintflow-adjoint-chunk-size", str(args.mintflow_adjoint_chunk_size),
            "--mintflow-time-penalty", str(args.mintflow_time_penalty),
            "--mintflow-correction-scale", str(args.mintflow_correction_scale),
            "--mintflow-score-mode", "default",
            "--mintflow-candidate-range", f"0.1-{args.mintflow_terminal_cutoff}",
            "--time-sampling", args.time_sampling,
            "--end-bias-power", str(args.end_bias_power),
            "--constraint-tolerance", str(args.projector_tolerance),
            "--constraint-ridge", str(args.projector_ridge), "--constraint-max-iter", str(args.final_projector_max_iter),
        ]
        if run_scenario == "conservation_ic_bc":
            common += [
                "--constraint-target-file", str(target_file),
                "--constraint-target-seed", str(target_seed),
            ]
        if position == 0 and not (noise / "initial_noise.npy").exists():
            common += ["--save-initial-noise", "--initial-noise-path", str(noise)]
        else:
            if not args.dry_run and not (noise / "initial_noise.npy").exists():
                raise FileNotFoundError("Shared noise was not created")
            common += ["--reuse-initial-noise", str(noise)]
        run_logged([*common, *spec.extra], logs / f"{spec.label}.log", args.dry_run)
        if not args.dry_run:
            complete = find_completed_run(
                label_root, args.seed, args.num_samples, required_metadata,
            )
            if complete is None:
                raise RuntimeError(f"No completed run found for {spec.label}")
            runs[spec.label] = str(complete)
            if args.constraint_scenario == "conservation_ic_bc" and spec.method == "vanilla":
                _ensure_target_bank(target_file, complete, args)

    if args.dry_run or args.sampling_only:
        return

    if args.constraint_scenario == "conservation_ic_bc":
        from eval_pde.benchmark_distributions import load_or_generate_paired_scenario_reference
        from eval_pde.config import build_rd1d_config
        from sampling.scenario_targets import load_scenario_target_bank

        bank = load_scenario_target_bank(target_file, "rd1d")
        reference, _ = load_or_generate_paired_scenario_reference(
            root / "reference",
            build_rd1d_config(task="global_balance", config_path="configs/rd1d.yml"),
            bank,
            workers=args.physical_workers,
        )
        if len(reference) != args.num_samples:
            raise ValueError("Scenario-2 numerical reference size mismatch")

    manifest = {
        "study": "RD1D nonlinear global_balance 14-configuration benchmark",
        "seed": args.seed, "num_samples": args.num_samples,
        "reference_samples": args.reference_samples,
        "shared_noise": str(noise / "initial_noise.npy"),
        "numerical_reference": str(numerical_reference_path),
        "reference_seed": args.reference_seed,
        "reference_contains_ffm_samples": False,
        "paper_alignment": {
            "grid": [128, 100], "euler_steps": args.euler_steps,
            "pcfm_newton_steps_per_flow_step": 1, "guided_interpolation_lambda": 0,
            "adaptation": "ECI nonlinear extension uses one GN update, one mix, and no resampling for paired-noise fidelity.",
        },
        "eci_native_equals_gn_expected": True,
        "mintflow_time_sampling": args.time_sampling,
        "mintflow_end_bias_power": args.end_bias_power,
        "mintflow_production_config": {
            "steps": args.mintflow_steps,
            "num_candidates": args.mintflow_candidates,
            "candidate_t_min": 0.1,
            "candidate_t_max": args.mintflow_terminal_cutoff,
            "time_sampling": args.time_sampling,
            "end_bias_power": args.end_bias_power,
            "time_penalty": args.mintflow_time_penalty,
            "correction_scale": args.mintflow_correction_scale,
            "correction_mode": "pseudoinverse",
            "final_projection": True,
            "projection_tolerance": args.projector_tolerance,
            "evaluation_pass_threshold": 1e-5,
        },
        "diffusionpde_eta": args.diffusionpde_eta,
        "constraint_scenario": args.constraint_scenario,
        "constraint_target_file": (
            str(target_file) if args.constraint_scenario == "conservation_ic_bc" else None
        ),
        "constraint_target_seed": (
            target_seed if args.constraint_scenario == "conservation_ic_bc" else None
        ),
        "intermediate_only_semantics": (
            "retain every in-loop projection/pullback except the final intrinsic "
            "projection or correction producing the returned sample"
        ),
        "strict_validity": "returned-field R_inf <= 1e-5",
        "runs": runs,
    }
    (root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))

    evaluation = [
        args.python, str(EVALUATE), "--benchmark-root", str(root),
        "--dataset", "rd1d", "--task", "global_balance",
        "--config", "configs/rd1d.yml", "--seed", str(args.seed),
    ]
    run_logged(evaluation, logs / "evaluate.log")
    run_logged([
        args.python, str(PHYSICAL), "--benchmark-root", str(root),
        "--dataset", "rd1d", "--task", "global_balance",
        "--config", "configs/rd1d.yml", "--workers", str(args.physical_workers),
    ], logs / "physical_consistency.log", False)
    run_logged([
        args.python, str(ANALYZE), "--benchmark-root", str(root),
        "--dataset", "rd1d", "--task", "global_balance",
        "--config", "configs/rd1d.yml", "--figure-seed", str(args.figure_seed),
    ], logs / "analyze.log", False)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--reference-samples", type=int, default=1000)
    p.add_argument("--reference-seed", type=int, default=20260901)
    p.add_argument("--physical-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument(
        "--constraint-scenario",
        choices=("conservation_only", "conservation_ic_bc"),
        default="conservation_only",
    )
    p.add_argument("--constraint-target-seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--euler-steps", type=int, default=200)
    p.add_argument("--mintflow-steps", type=int, default=200)
    p.add_argument("--mintflow-candidates", type=int, default=10)
    p.add_argument(
        "--diffusionpde-eta", type=float, default=None,
        help="Guidance strength (default: 1.0 for Scenario 1, 1e-3 for Scenario 2).",
    )
    p.add_argument("--dflow-n-iter", type=int, default=20)
    p.add_argument("--dflow-lr", type=float, default=1.0)
    p.add_argument("--mintflow-ridge", type=float, default=1e-5)
    p.add_argument("--mintflow-adjoint-chunk-size", type=int, default=64)
    p.add_argument("--mintflow-time-penalty", type=float, default=1e-2)
    p.add_argument("--mintflow-correction-scale", type=float, default=1.0)
    p.add_argument("--mintflow-terminal-cutoff", type=float, default=0.98)
    p.add_argument("--time-sampling", choices=("end_biased", "uniform"), default="end_biased")
    p.add_argument("--end-bias-power", type=float, default=2.0)
    p.add_argument("--projector-ridge", type=float, default=1e-6)
    p.add_argument("--projector-tolerance", type=float, default=1e-6)
    p.add_argument("--final-projector-max-iter", type=int, default=10)
    p.add_argument("--figure-seed", type=int, default=20260902)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument(
        "--configuration", action="append", choices=[spec.label for spec in CONFIGURATIONS],
        help="Run only this configuration (repeatable); default runs the full suite.",
    )
    p.add_argument(
        "--sampling-only", action="store_true",
        help="Stop after selected sampling runs; useful for concurrent GPU workers.",
    )
    return p


if __name__ == "__main__":
    main(parser().parse_args())
