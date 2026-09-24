#!/usr/bin/env python3
"""Resumable, shared-noise Heat MintFlow ablation runner for both scenarios."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "results" / "ablation"
PYTHON = REPO / ".venv" / "bin" / "python"
RUN_SAMPLING = REPO / "scripts" / "run_sampling.py"
MERGER = REPO / "scripts" / "merge_sampling_shards.py"
REFERENCE = REPO / "results/heat_mass_12way_benchmark/reference/numerical_reference_samples.npy"
NOISE = REPO / "results/heat_mass_12way_benchmark/shared_noise/initial_noise.npy"
POD = REPO / "results/heat_mass_12way_benchmark/evaluation/frozen_pod_embedding.npz"
DEFAULT_RUN = (
    REPO
    / "results/heat_mass_12way_benchmark/raw/"
    "mintflow_pseudoinverse_with_final_projection/diffusion/mintflow/"
    "production_steps200_k10_smax0p98_n1000"
)
SEED = 20260811


@dataclass(frozen=True)
class AblationStudy:
    """Immutable inputs and output namespace for one Heat constraint scenario."""

    key: str
    constraint_scenario: str
    root: Path
    reference: Path
    noise: Path
    pod: Path
    default_run: Path | None
    target_file: Path | None = None
    target_seed: int | None = None
    reuse_default: bool = True


def study_for(scenario: str) -> AblationStudy:
    if scenario == "type1":
        return AblationStudy(
            key="type1",
            constraint_scenario="conservation_only",
            root=ROOT,
            reference=REFERENCE,
            noise=NOISE,
            pod=POD,
            default_run=DEFAULT_RUN,
        )
    if scenario == "type2":
        benchmark = REPO / "results" / "heat_mass_scenario2_benchmark"
        return AblationStudy(
            key="type2",
            constraint_scenario="conservation_ic_bc",
            root=ROOT / "heat_type2",
            reference=benchmark / "reference" / "numerical_reference_samples.npy",
            noise=benchmark / "shared_noise" / "initial_noise.npy",
            pod=benchmark / "evaluation" / "frozen_pod_embedding.npz",
            # Re-sample the default as part of the requested Type-2 rerun so
            # its dynamic diagnostics are produced by the audited code path.
            default_run=None,
            target_file=(
                benchmark / "scenario_target" /
                "diffusion_vanilla_bank_seed20260811_n1000.npz"
            ),
            target_seed=SEED,
            reuse_default=False,
        )
    raise ValueError(f"Unknown ablation scenario {scenario!r}")


@dataclass(frozen=True)
class AblationConfig:
    config_id: str
    steps: int = 200
    num_candidates: int = 10
    spacing: str = "end_biased"
    terminal_cutoff: float = 0.98
    time_penalty: float = 0.01
    correction_scale: float = 1.0
    final_projection: bool = True
    end_bias_power: float = 2.0
    correction_mode: str = "pseudoinverse"
    ridge: float = 1e-5
    candidate_t_min: float = 0.1
    score_mode: str = "default"
    interior_penalty: float = 0.0
    correction_weight: float = 1.0
    pseudoinverse_rcond: float | None = None
    adjoint_chunk_size: int = 64


def ablation_design(
    *, include_projection_interactions: bool = False,
) -> tuple[list[AblationConfig], dict[str, list[str]]]:
    base = AblationConfig("default")
    configs: dict[str, AblationConfig] = {base.config_id: base}
    groups: dict[str, list[str]] = {}

    groups["integration_resolution"] = []
    for value in (50, 100, 200, 400):
        item = base if value == 200 else replace(base, config_id=f"steps_{value}", steps=value)
        configs[item.config_id] = item
        groups["integration_resolution"].append(item.config_id)

    groups["candidate_grid"] = []
    for spacing in ("end_biased", "uniform"):
        for count in (1, 5, 10, 20, 40):
            item = (
                base if (spacing, count) == ("end_biased", 10)
                else replace(base, config_id=f"grid_{spacing}_k{count}",
                             spacing=spacing, num_candidates=count)
            )
            configs[item.config_id] = item
            groups["candidate_grid"].append(item.config_id)

    groups["time_regularization"] = []
    for value, label in ((0.0, "0"), (1e-4, "1em4"), (1e-3, "1em3"),
                         (1e-2, "1em2"), (1e-1, "1em1"), (1.0, "1")):
        item = base if value == 1e-2 else replace(
            base, config_id=f"lambda_{label}", time_penalty=value
        )
        configs[item.config_id] = item
        groups["time_regularization"].append(item.config_id)

    groups["final_projection"] = ["default", "final_projection_off"]
    configs["final_projection_off"] = replace(
        base, config_id="final_projection_off", final_projection=False
    )

    groups["correction_damping"] = []
    for value, label in ((0.5, "0p5"), (0.8, "0p8"), (1.0, "1p0")):
        item = base if value == 1.0 else replace(
            base, config_id=f"gamma_{label}", correction_scale=value
        )
        configs[item.config_id] = item
        groups["correction_damping"].append(item.config_id)

    if include_projection_interactions:
        groups["projection_damping_interaction"] = [
            "gamma_0p5_no_fp", "gamma_0p8_no_fp", "final_projection_off"
        ]
        configs["gamma_0p5_no_fp"] = replace(
            base, config_id="gamma_0p5_no_fp", correction_scale=0.5,
            final_projection=False,
        )
        configs["gamma_0p8_no_fp"] = replace(
            base, config_id="gamma_0p8_no_fp", correction_scale=0.8,
            final_projection=False,
        )
        groups["projection_regularization_interaction"] = [
            "lambda_0_no_fp", "final_projection_off", "lambda_1_no_fp"
        ]
        configs["lambda_0_no_fp"] = replace(
            base, config_id="lambda_0_no_fp", time_penalty=0.0,
            final_projection=False,
        )
        configs["lambda_1_no_fp"] = replace(
            base, config_id="lambda_1_no_fp", time_penalty=1.0,
            final_projection=False,
        )
    return list(configs.values()), groups


def _canonical_noise_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    return path / "initial_noise.npy" if path.is_dir() else path


def _same_optional_float(actual: Any, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    try:
        return abs(float(actual) - expected) < 1e-12
    except (TypeError, ValueError):
        return False


def _metadata_signature(
    metadata: dict[str, Any], config: AblationConfig, study: AblationStudy | None = None,
) -> bool:
    study = study or study_for("type1")
    target_matches = True
    if study.target_file is not None:
        target_matches = (
            Path(str(metadata.get("constraint_target_file", ""))).resolve()
            == study.target_file.resolve()
            and int(metadata.get("constraint_target_seed", -1)) == study.target_seed
        )
    else:
        target_matches = metadata.get("constraint_target_file") in (None, "")
    return (
        metadata.get("dataset") == "diffusion"
        and metadata.get("method") == "mintflow"
        and int(metadata.get("seed", -1)) == SEED
        and int(metadata.get("mintflow_forward_steps", -1)) == config.steps
        and int(metadata.get("mintflow_num_candidates", -1)) == config.num_candidates
        and metadata.get("mintflow_time_sampling") == config.spacing
        and abs(float(metadata.get("mintflow_candidate_t_min", -1)) - config.candidate_t_min) < 1e-12
        and abs(float(metadata.get("mintflow_candidate_t_max", -1)) - config.terminal_cutoff) < 1e-12
        and abs(float(metadata.get("mintflow_end_bias_power", -1)) - config.end_bias_power) < 1e-12
        and abs(float(metadata.get("mintflow_time_penalty", -1)) - config.time_penalty) < 1e-12
        and abs(float(metadata.get("mintflow_correction_scale", -1)) - config.correction_scale) < 1e-12
        and metadata.get("mintflow_score_mode") == config.score_mode
        and abs(float(metadata.get("mintflow_interior_penalty", -1)) - config.interior_penalty) < 1e-12
        and abs(float(metadata.get("mintflow_correction_weight", -1)) - config.correction_weight) < 1e-12
        and metadata.get("mintflow_correction_mode") == config.correction_mode
        and abs(float(metadata.get("mintflow_ridge", -1)) - config.ridge) < 1e-12
        and _same_optional_float(
            metadata.get("mintflow_pseudoinverse_rcond"), config.pseudoinverse_rcond
        )
        and int(metadata.get("mintflow_adjoint_chunk_size", -1)) == config.adjoint_chunk_size
        and bool(metadata.get("post_sampling_final_projection_enabled", False)) == config.final_projection
        and metadata.get("constraint_scenario") == study.constraint_scenario
        and _canonical_noise_path(str(metadata.get("initial_noise_path", "")))
        == study.noise.resolve()
        and target_matches
    )


def _valid_run(
    path: Path, config: AblationConfig, offset: int, count: int,
    study: AblationStudy | None = None,
) -> bool:
    try:
        metadata = json.loads((path / "metadata.json").read_text())
        samples = np.load(path / "samples.npy", mmap_mode="r")
        sample_ids = np.asarray(metadata.get("sample_ids", []), dtype=np.int64)
        return (
            int(metadata.get("sample_offset", -1)) == offset
            and int(metadata.get("num_samples", -1)) == count
            and samples.shape == (count, 100, 100)
            and np.array_equal(sample_ids, np.arange(offset, offset + count))
            and _metadata_signature(metadata, config, study)
        )
    except (OSError, TypeError, ValueError, KeyError):
        return False


def _find_valid_run(
    root: Path, config: AblationConfig, offset: int, count: int,
    study: AblationStudy,
) -> Path | None:
    for metadata in root.rglob("metadata.json") if root.exists() else ():
        if _valid_run(metadata.parent, config, offset, count, study):
            return metadata.parent
    return None


def _valid_shard_runs(
    config: AblationConfig, num_samples: int, study: AblationStudy,
) -> list[tuple[int, int, Path]]:
    """Return every protocol-valid shard, independent of historical shard size."""
    root = study.root / "runs" / config.config_id / "shards"
    intervals: dict[tuple[int, int], Path] = {}
    for metadata_path in root.rglob("metadata.json") if root.exists() else ():
        try:
            metadata = json.loads(metadata_path.read_text())
            offset = int(metadata.get("sample_offset", -1))
            count = int(metadata.get("num_samples", -1))
        except (OSError, TypeError, ValueError):
            continue
        if offset < 0 or count < 1 or offset + count > num_samples:
            continue
        if _valid_run(metadata_path.parent, config, offset, count, study):
            key = (offset, offset + count)
            previous = intervals.get(key)
            if previous is None or metadata_path.stat().st_mtime > (
                previous / "metadata.json"
            ).stat().st_mtime:
                intervals[key] = metadata_path.parent
    return [(start, stop, path) for (start, stop), path in sorted(intervals.items())]


def _select_exact_tiling(
    runs: list[tuple[int, int, Path]], num_samples: int,
) -> list[Path] | None:
    """Select non-overlapping shards that exactly tile ``[0, num_samples)``."""
    by_start: dict[int, list[tuple[int, Path]]] = {}
    for start, stop, path in runs:
        by_start.setdefault(start, []).append((stop, path))
    reachable: dict[int, list[Path] | None] = {num_samples: []}
    for start in range(num_samples - 1, -1, -1):
        reachable[start] = None
        for stop, path in sorted(by_start.get(start, []), reverse=True):
            suffix = reachable.get(stop)
            if suffix is not None:
                reachable[start] = [path, *suffix]
                break
    return reachable[0]


def _uncovered_intervals(
    runs: list[tuple[int, int, Path]], num_samples: int,
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, stop, _ in runs:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    gaps = []
    position = 0
    for start, stop in merged + [[num_samples, num_samples]]:
        if position < start:
            gaps.append((position, start))
        position = max(position, stop)
    return gaps


def _tasks(
    num_samples: int, shard_size: int, study: AblationStudy | None = None,
) -> list[tuple[AblationConfig, int, int]]:
    study = study or study_for("type1")
    configs, _ = ablation_design(
        include_projection_interactions=study.key == "type2"
    )
    tasks = []
    for config in configs:
        if config.config_id == "default" and study.reuse_default:
            continue
        merged = study.root / "runs" / config.config_id / "merged"
        if _valid_run(merged, config, 0, num_samples, study):
            continue
        runs = _valid_shard_runs(config, num_samples, study)
        if _select_exact_tiling(runs, num_samples) is not None:
            continue
        for start, stop in _uncovered_intervals(runs, num_samples):
            for offset in range(start, stop, shard_size):
                tasks.append((config, offset, min(shard_size, stop - offset)))
    # Longest jobs first keeps a dynamic multi-GPU allocation balanced.
    tasks.sort(key=lambda item: (-item[0].steps, item[0].config_id, item[1]))
    return tasks


def _command(
    config: AblationConfig, offset: int, count: int, output: Path,
    study: AblationStudy | None = None,
) -> list[str]:
    study = study or study_for("type1")
    command = [
        str(PYTHON), str(RUN_SAMPLING), "--dataset", "diffusion",
        "--method", "mintflow", "--config", "configs/heat.yml",
        "--checkpoint", "logs/diffusion/latest.pt", "--num-samples", str(count),
        "--sample-offset", str(offset), "--batch-size", "16", "--seed", str(SEED),
        "--device", "cuda", "--task", "heat_mass_conservation",
        "--constraint-scenario", study.constraint_scenario,
        "--reuse-initial-noise", str(study.noise),
        "--output-dir", str(output), "--num-preview", "0", "--wandb-mode", "disabled",
        "--save-sampler-diagnostics", "--mintflow-forward-steps", str(config.steps),
        "--mintflow-num-candidates", str(config.num_candidates),
        "--mintflow-ridge", str(config.ridge), "--mintflow-adjoint-chunk-size",
        str(config.adjoint_chunk_size),
        "--mintflow-time-penalty", str(config.time_penalty),
        "--mintflow-correction-scale", str(config.correction_scale),
        "--mintflow-score-mode", config.score_mode,
        "--mintflow-interior-penalty", str(config.interior_penalty),
        "--mintflow-correction-weight", str(config.correction_weight),
        "--mintflow-candidate-range",
        f"{config.candidate_t_min}-{config.terminal_cutoff}", "--time-sampling", config.spacing,
        "--end-bias-power", str(config.end_bias_power), "--mintflow-correction-mode",
        config.correction_mode, "--constraint-tolerance", "1e-6",
        "--constraint-ridge", "1e-6", "--constraint-max-iter", "10",
        "--final-projection" if config.final_projection else "--no-final-projection",
    ]
    if config.pseudoinverse_rcond is not None:
        command += ["--mintflow-pseudoinverse-rcond", str(config.pseudoinverse_rcond)]
    if study.target_file is not None:
        command += [
            "--constraint-target-file", str(study.target_file),
            "--constraint-target-seed", str(study.target_seed),
        ]
    return command


def prepare(args: argparse.Namespace) -> None:
    study = study_for(args.scenario)
    study.root.mkdir(parents=True, exist_ok=True)
    required_inputs = [study.reference, study.noise, study.pod]
    if study.target_file is not None:
        required_inputs.append(study.target_file)
    if study.reuse_default and study.default_run is not None:
        required_inputs.extend([
            study.default_run / "samples.npy", study.default_run / "metadata.json"
        ])
    for required in required_inputs:
        if not required.is_file():
            raise FileNotFoundError(required)
    configs, groups = ablation_design(
        include_projection_interactions=study.key == "type2"
    )
    default = configs[0]
    if study.reuse_default and study.default_run is not None:
        metadata = json.loads((study.default_run / "metadata.json").read_text())
        if not (_metadata_signature(metadata, default, study)
                and int(metadata.get("num_samples", -1)) == args.num_samples):
            raise RuntimeError(
                "Existing production default does not match the ablation protocol"
            )
    manifest = {
        "study": f"Heat {study.key.title()} MintFlow architectural ablation",
        "scenario_key": study.key,
        "constraint_scenario": study.constraint_scenario,
        "num_samples_per_configuration": args.num_samples,
        "seed": SEED,
        "reference": str(study.reference.resolve()),
        "shared_noise": str(study.noise.resolve()),
        "frozen_pod_embedding": str(study.pod.resolve()),
        "scenario_target_file": (
            str(study.target_file.resolve()) if study.target_file else None
        ),
        "scenario_target_seed": study.target_seed,
        "reused_validated_default_run": (
            str(study.default_run.resolve())
            if study.reuse_default and study.default_run else None
        ),
        "defaults": asdict(default),
        "configurations": [asdict(item) for item in configs],
        "factor_groups": groups,
        "unique_configuration_count": len(configs),
        "new_configuration_count": len(configs) - int(study.reuse_default),
        "shard_size": args.shard_size,
        "metrics": ["FID", "KID", "MMD", "W2", "conservation", "PDE residual",
                    "simulation error", "time", "NFE", "peak GPU memory"],
    }
    (study.root / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if args.clear_claims:
        shutil.rmtree(study.root / "claims", ignore_errors=True)
    (study.root / "claims").mkdir(exist_ok=True)
    print(json.dumps({"unique_configs": len(configs), "pending_shards": len(_tasks(
        args.num_samples, args.shard_size, study))}, indent=2))


def worker(args: argparse.Namespace) -> None:
    study = study_for(args.scenario)
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    gpu_count = int(os.environ.get("ABLATION_GPUS_PER_NODE", "4"))
    # With --gpus-per-task Slurm has already restricted each task to one GPU.
    # Preserve that binding; use local-rank mapping only outside such a step.
    assigned_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = assigned_gpu if assigned_gpu else str(local_rank % gpu_count)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "OMP_NUM_THREADS": "1",
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "UV_CACHE_DIR": str(REPO / ".uv-cache"),
    }
    completed = 0
    while True:
        claimed = None
        for config, offset, count in _tasks(
            args.num_samples, args.shard_size, study
        ):
            claim = study.root / "claims" / f"{config.config_id}__{offset:04d}_n{count}"
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            claimed = config, offset, count, claim
            break
        if claimed is None:
            print(f"worker {os.getpid()}: no unclaimed pending shards", flush=True)
            return
        config, offset, count, claim = claimed
        output = (
            study.root / "runs" / config.config_id / "shards" /
            f"offset_{offset:04d}"
        )
        output.mkdir(parents=True, exist_ok=True)
        log = output / "sampling.log"
        execution = {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_proc_id": os.environ.get("SLURM_PROCID"),
            "slurm_local_id": os.environ.get("SLURM_LOCALID"),
            "physical_gpu_id": gpu_id,
        }
        try:
            with log.open("w") as handle:
                subprocess.run(_command(config, offset, count, output, study), cwd=REPO,
                               env=env, stdout=handle, stderr=subprocess.STDOUT,
                               check=True)
            run = _find_valid_run(output, config, offset, count, study)
            if run is None:
                raise RuntimeError("sampler exited without a protocol-valid run")
            metadata_path = run / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["ablation_execution"] = execution
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            (claim / "completed.json").write_text(json.dumps({
                **execution, "run": str(run.resolve())
            }, indent=2) + "\n")
            completed += 1
            print(f"DONE {config.config_id} [{offset}:{offset + count}]", flush=True)
        except Exception as exc:
            (claim / "failed.json").write_text(json.dumps({
                **execution, "error": repr(exc)
            }, indent=2) + "\n")
            print(f"FAILED {config.config_id} [{offset}:{offset + count}]: {exc}",
                  file=sys.stderr, flush=True)
        if args.max_shards and completed >= args.max_shards:
            return


def merge(args: argparse.Namespace) -> None:
    study = study_for(args.scenario)
    configs, _ = ablation_design(
        include_projection_interactions=study.key == "type2"
    )
    failures = []
    for config in configs:
        if config.config_id == "default" and study.reuse_default:
            continue
        output = study.root / "runs" / config.config_id / "merged"
        if _valid_run(output, config, 0, args.num_samples, study):
            print(f"SKIP merged {config.config_id}")
            continue
        shard_runs = _select_exact_tiling(
            _valid_shard_runs(config, args.num_samples, study), args.num_samples
        )
        if shard_runs is None:
            failures.append(config.config_id)
            continue
        if output.exists():
            shutil.rmtree(output)
        command = [str(PYTHON), str(MERGER)]
        for run in shard_runs:
            command += ["--shard-run", str(run)]
        command += ["--output-dir", str(output), "--total-samples", str(args.num_samples)]
        subprocess.run(command, cwd=REPO, check=True)
        if not _valid_run(output, config, 0, args.num_samples, study):
            raise RuntimeError(f"Merged run failed validation: {config.config_id}")
        print(f"MERGED {config.config_id}")
    if failures:
        print(f"Pending shard count: {len(failures)}")
        sys.exit(2)


def status(args: argparse.Namespace) -> None:
    study = study_for(args.scenario)
    configs, _ = ablation_design(
        include_projection_interactions=study.key == "type2"
    )
    pending = _tasks(args.num_samples, args.shard_size, study)
    by_config = {item.config_id: 0 for item in configs}
    for config, _, _ in pending:
        by_config[config.config_id] += 1
    print(json.dumps({
        "total_configs": len(configs),
        "scenario": study.key,
        "default_reused": study.reuse_default,
        "pending_shards": len(pending),
        "pending_by_config": {key: value for key, value in by_config.items() if value},
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "merge", "status"))
    p.add_argument("--scenario", choices=("type1", "type2"), default="type1")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--shard-size", type=int, default=50)
    p.add_argument("--clear-claims", action="store_true")
    p.add_argument("--max-shards", type=int, default=0)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    {"prepare": prepare, "worker": worker, "merge": merge, "status": status}[args.action](args)
