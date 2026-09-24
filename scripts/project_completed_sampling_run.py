#!/usr/bin/env python3
"""Apply the benchmark's terminal projector to an existing completed run.

This avoids rerunning an expensive deterministic sampler merely to evaluate
its ``+FP`` ablation. The resulting artifact has the same schema and total
cost accounting as an inline ``--final-projection`` run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sampling.methods import apply_optional_final_projection
from sampling.projectors import ProjectionConfig
from sampling.tasks import ConstraintTaskProvider, build_task
from scripts.training.utils import load_config


def main(args: argparse.Namespace) -> None:
    source = Path(args.source_run).resolve()
    source_meta = json.loads((source / "metadata.json").read_text())
    samples_np = np.load(source / "samples.npy")
    if source_meta.get("final_projection_arg"):
        raise ValueError("Source run already has final projection enabled")

    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    scenario = source_meta.get("constraint_scenario", "conservation_only")
    if scenario == "conservation_ic_bc":
        from sampling.scenario_targets import load_scenario_target_bank

        bank = load_scenario_target_bank(
            source_meta["constraint_target_file"], args.dataset
        )
        offset = int(source_meta.get("sample_offset", 0))
        bank.validate_slice(offset, len(samples_np))

        def build_paired(local_index: int):
            global_index = offset + local_index
            reference = torch.zeros(config.sample_dims, dtype=torch.float32)
            reference[:, 0] = torch.from_numpy(
                bank.initial_conditions[global_index]
            )
            boundary = (
                torch.from_numpy(bank.boundary_values[global_index])
                if bank.boundary_values.shape[1] else None
            )
            task = build_task(
                args.dataset, args.task, config=config,
                config_path=str(config_path), reference_sample=reference,
                boundary_target=boundary, scenario=scenario, device=args.device,
            )
            task.metadata.update({
                "target_id": bank.target_id,
                "target_kind": "paired_per_sample_bank",
                "target_index": global_index,
                "scenario_target": bank.metadata,
            })
            return task

        first = build_paired(0)
        task = ConstraintTaskProvider(
            num_tasks=len(samples_np), task_builder=build_paired,
            metadata={
                **first.metadata,
                "paired_target_count": len(samples_np),
                "paired_target_offset": offset,
            },
        )
    else:
        task = build_task(
            args.dataset, args.task, config=config,
            config_path=str(config_path), scenario="conservation_only",
            device=args.device,
        )
    projection_config = ProjectionConfig(
        max_iter=args.max_iter, damping=args.ridge, tol=args.tolerance,
        scales=task.scales,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        Path(args.output_root).resolve() / args.dataset / source_meta["method"]
        / f"derived_final_projection_seed{source_meta['seed']}_n{len(samples_np)}_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(source_meta)
    flow_value = source_meta.get("flow_paths_file")
    if flow_value:
        flow_source = Path(flow_value)
        if not flow_source.is_absolute():
            cwd_relative = flow_source.resolve()
            flow_source = cwd_relative if cwd_relative.exists() else source / flow_source
        if flow_source.exists():
            shutil.copy2(flow_source, output / "flow_paths.npz")
            metadata["flow_paths_file"] = str((output / "flow_paths.npz").resolve())

    samples = torch.from_numpy(samples_np).to(args.device)
    cuda_timing = args.device.startswith("cuda") and torch.cuda.is_available()
    if cuda_timing:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    projected, metadata = apply_optional_final_projection(
        samples, metadata, task, enabled=True, config=projection_config,
        save_per_sample=True,
    )
    if cuda_timing:
        torch.cuda.synchronize(args.device)
    projection_time = time.perf_counter() - started
    projection_peak = (
        int(torch.cuda.max_memory_allocated(args.device)) if cuda_timing else 0
    )

    projected_np = projected.detach().cpu().numpy()
    np.save(output / "samples.npy", projected_np)
    metadata.update({
        "num_samples": len(projected_np),
        "sample_shape": list(projected_np.shape),
        "samples_file": "samples.npy",
        "output_dir": str(output),
        "timestamp": timestamp,
        "final_projection_arg": True,
        "derived_final_projection": True,
        "source_run_before_final_projection": str(source),
        "source_generation_time_s": float(source_meta.get("generation_time_s", 0.0)),
        "final_projection_time_s": round(projection_time, 2),
        "generation_time_s": round(
            float(source_meta.get("generation_time_s", 0.0)) + projection_time, 2
        ),
        "peak_gpu_memory_bytes": max(
            int(source_meta.get("peak_gpu_memory_bytes") or 0), projection_peak,
        ),
    })
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(output)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--dataset", required=True, choices=("diffusion", "rd1d"))
    p.add_argument("--task", required=True, choices=("heat_mass_conservation", "global_balance"))
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-iter", type=int, default=10)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--tolerance", type=float, default=1e-6)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
