#!/usr/bin/env python3
"""Run/resume the primary N=1000 MintFlow configuration in deterministic shards."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_SAMPLING = REPO / "scripts" / "run_sampling.py"


@dataclass(frozen=True)
class Scenario:
    name: str
    root: Path
    dataset: str
    task: str
    config: str
    checkpoint: str
    seed: int
    constraint_scenario: str
    target_file: Path | None = None


SCENARIOS = {
    "heat2": Scenario(
        "heat2", Path("results/heat_mass_scenario2_benchmark"), "diffusion",
        "heat_mass_conservation", "configs/heat.yml", "logs/diffusion/latest.pt",
        20260811, "conservation_ic_bc",
        Path("results/heat_mass_scenario2_benchmark/scenario_target/"
             "diffusion_vanilla_bank_seed20260811_n1000.npz"),
    ),
    "rd1": Scenario(
        "rd1", Path("results/rd_global_balance_12way_benchmark"), "rd1d",
        "global_balance", "configs/rd1d.yml", "logs/rd1d/latest.pt",
        20260820, "conservation_only",
    ),
    "rd2": Scenario(
        "rd2", Path("results/rd_global_balance_scenario2_benchmark"), "rd1d",
        "global_balance", "configs/rd1d.yml", "logs/rd1d/latest.pt",
        20260820, "conservation_ic_bc",
        Path("results/rd_global_balance_scenario2_benchmark/scenario_target/"
             "rd1d_vanilla_bank_seed20260820_n1000.npz"),
    ),
}


def _completed_run(directory: Path, offset: int, count: int) -> Path | None:
    for metadata_path in directory.rglob("metadata.json") if directory.exists() else ():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            continue
        signature = (
            int(metadata.get("sample_offset", -1)),
            int(metadata.get("num_samples", -1)),
            int(metadata.get("mintflow_forward_steps", -1)),
            int(metadata.get("mintflow_num_candidates", -1)),
            float(metadata.get("mintflow_candidate_t_max", -1.0)),
            float(metadata.get("mintflow_time_penalty", -1.0)),
            float(metadata.get("mintflow_correction_scale", -1.0)),
            metadata.get("mintflow_time_sampling"),
            metadata.get("mintflow_correction_mode"),
            bool(metadata.get("post_sampling_final_projection_enabled", False)),
        )
        if signature == (offset, count, 200, 10, 0.98, 0.01, 1.0,
                         "end_biased", "pseudoinverse", True):
            run = metadata_path.parent
            if (run / "samples.npy").is_file():
                return run
    return None


def _completed_intervals(root: Path, total_samples: int) -> list[tuple[int, int]]:
    """Return the union of validated production shard intervals.

    Shard size is deliberately not part of the persisted experiment protocol.
    This lets a wall-time recovery use smaller shards without recomputing any
    sample IDs that were already completed by a larger shard.
    """
    intervals: list[tuple[int, int]] = []
    shard_root = root / "production_primary_shards"
    for metadata_path in shard_root.rglob("metadata.json") if shard_root.exists() else ():
        try:
            metadata = json.loads(metadata_path.read_text())
            offset = int(metadata.get("sample_offset", -1))
            count = int(metadata.get("num_samples", -1))
        except (OSError, TypeError, ValueError):
            continue
        if offset < 0 or count < 1 or offset + count > total_samples:
            continue
        if _completed_run(metadata_path.parent, offset, count) is not None:
            intervals.append((offset, offset + count))

    merged: list[list[int]] = []
    for start, stop in sorted(set(intervals)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(start, stop) for start, stop in merged]


def _pending_shards(
    scenario: Scenario, total_samples: int, shard_size: int,
) -> list[tuple[Scenario, int, int]]:
    completed = _completed_intervals(scenario.root, total_samples)
    tasks: list[tuple[Scenario, int, int]] = []
    position = 0
    for start, stop in completed + [(total_samples, total_samples)]:
        while position < start:
            count = min(shard_size, start - position)
            tasks.append((scenario, position, count))
            position += count
        position = max(position, stop)
    return tasks


def _command(scenario: Scenario, offset: int, count: int, use_srun: bool) -> list[str]:
    output = scenario.root / "production_primary_shards" / f"offset_{offset:04d}"
    command = [
        str(REPO / ".venv" / "bin" / "python"), str(RUN_SAMPLING),
        "--dataset", scenario.dataset, "--method", "mintflow",
        "--config", scenario.config, "--checkpoint", scenario.checkpoint,
        "--num-samples", str(count), "--sample-offset", str(offset),
        "--batch-size", "16", "--seed", str(scenario.seed), "--device", "cuda",
        "--task", scenario.task, "--constraint-scenario", scenario.constraint_scenario,
        "--reuse-initial-noise", str(scenario.root / "shared_noise"),
        "--output-dir", str(output), "--num-preview", "0", "--wandb-mode", "disabled",
        "--evaluate-constraints", "--save-constraint-diagnostics",
        "--save-sampler-diagnostics", "--mintflow-forward-steps", "200",
        "--mintflow-num-candidates", "10", "--mintflow-ridge", "1e-5",
        "--mintflow-adjoint-chunk-size", "64", "--mintflow-time-penalty", "0.01",
        "--mintflow-correction-scale", "1.0", "--mintflow-score-mode", "default",
        "--mintflow-candidate-range", "0.1-0.98", "--time-sampling", "end_biased",
        "--end-bias-power", "2.0", "--mintflow-correction-mode", "pseudoinverse",
        "--constraint-tolerance", "1e-6", "--constraint-ridge", "1e-6",
        "--constraint-max-iter", "10", "--final-projection",
    ]
    if scenario.target_file is not None:
        command.extend([
            "--constraint-target-file", str(scenario.target_file),
            "--constraint-target-seed", str(scenario.seed),
        ])
    if use_srun:
        command = [
            "srun", "--exact", "--nodes=1", "--ntasks=1",
            "--gpus-per-task=1", "--cpus-per-task=2", "--gpu-bind=single:1", *command,
        ]
    return command


def _run_one(scenario: Scenario, offset: int, count: int, use_srun: bool) -> Path:
    shard_root = scenario.root / "production_primary_shards" / f"offset_{offset:04d}"
    complete = _completed_run(shard_root, offset, count)
    if complete is not None:
        print(f"SKIP {scenario.name} [{offset}:{offset + count}]", flush=True)
        return complete
    shard_root.mkdir(parents=True, exist_ok=True)
    log = shard_root / "sampling.log"
    with log.open("w") as handle:
        subprocess.run(
            _command(scenario, offset, count, use_srun), cwd=REPO,
            stdout=handle, stderr=subprocess.STDOUT, check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"},
        )
    complete = _completed_run(shard_root, offset, count)
    if complete is None:
        raise RuntimeError(f"Completed shard not found for {scenario.name} offset {offset}")
    print(f"DONE {scenario.name} [{offset}:{offset + count}]", flush=True)
    return complete


def main(args: argparse.Namespace) -> None:
    selected = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in selected:
        scenario = SCENARIOS[name]
        if scenario.target_file is not None and not scenario.target_file.is_file():
            raise FileNotFoundError(scenario.target_file)
        if not (scenario.root / "shared_noise" / "initial_noise.npy").is_file():
            raise FileNotFoundError(scenario.root / "shared_noise" / "initial_noise.npy")

    tasks = []
    for name in selected:
        scenario = SCENARIOS[name]
        tasks.extend(_pending_shards(scenario, args.num_samples, args.shard_size))
    # Longest scenario first gives a near-optimal dynamic GPU schedule.
    priority = {"rd2": 3, "heat2": 2, "rd1": 1}
    tasks.sort(key=lambda item: (-priority[item[0].name], item[1]))

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one, scenario, offset, count, args.srun):
            (scenario.name, offset)
            for scenario, offset, count in tasks
        }
        for future in as_completed(futures):
            name, offset = futures[future]
            try:
                future.result()
            except Exception as exc:  # Continue so other GPU shards remain recoverable.
                failures.append((name, offset, repr(exc)))
                print(f"FAILED {name} offset={offset}: {exc}", file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} shard(s) failed: {failures}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--shard-size", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--srun", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
