#!/usr/bin/env python3
"""Evaluate every complete run in an isolated Heat MintFlow ablation."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eval_pde.artifacts import file_sha256
from eval_pde.config import build_diffusion_config
from eval_pde.embeddings import FrozenPODEmbedding, fit_or_load_frozen_pod
from eval_pde.heat_ablation import (
    evaluate_heat_ablation_run,
    load_frozen_heat_diffusivities,
)
from sampling.adjoint_correction import select_candidate_indices
from sampling.scenario_targets import load_scenario_target_bank
from scripts.run_heat_mintflow_ablation import (
    SEED,
    AblationConfig,
    AblationStudy,
    _canonical_noise_path,
    _metadata_signature,
    ablation_design,
    study_for,
)


def _run_path(config: AblationConfig, study: AblationStudy) -> Path:
    if config.config_id == "default" and study.reuse_default:
        if study.default_run is None:
            raise RuntimeError("Default reuse requested without a default run")
        return study.default_run
    return study.root / "runs" / config.config_id / "merged"


def _validate_metadata(
    metadata: dict, config: AblationConfig, n: int, study: AblationStudy,
) -> None:
    actual_noise = _canonical_noise_path(
        str(metadata.get("initial_noise_path", ""))
    )
    expected = {
        "num_samples": (int(metadata.get("num_samples", -1)), n),
        "sample_offset": (int(metadata.get("sample_offset", -1)), 0),
        "steps": (int(metadata.get("mintflow_forward_steps", -1)), config.steps),
        "K": (int(metadata.get("mintflow_num_candidates", -1)), config.num_candidates),
        "spacing": (metadata.get("mintflow_time_sampling"), config.spacing),
        "candidate_t_min": (
            float(metadata.get("mintflow_candidate_t_min", -1)),
            config.candidate_t_min,
        ),
        "candidate_t_max": (
            float(metadata.get("mintflow_candidate_t_max", -1)),
            config.terminal_cutoff,
        ),
        "end_bias_power": (
            float(metadata.get("mintflow_end_bias_power", -1)),
            config.end_bias_power,
        ),
        "lambda": (float(metadata.get("mintflow_time_penalty", -1)), config.time_penalty),
        "gamma": (float(metadata.get("mintflow_correction_scale", -1)), config.correction_scale),
        "final_projection": (
            bool(metadata.get("post_sampling_final_projection_enabled", False)),
            config.final_projection,
        ),
        "seed": (int(metadata.get("seed", -1)), SEED),
        "noise": (actual_noise, study.noise.resolve()),
        "scenario": (
            metadata.get("constraint_scenario"), study.constraint_scenario
        ),
    }
    mismatches = {key: pair for key, pair in expected.items() if pair[0] != pair[1]}
    if mismatches:
        raise ValueError(f"Run metadata mismatch for {config.config_id}: {mismatches}")
    if not _metadata_signature(metadata, config, study):
        raise ValueError(f"Full configuration signature mismatch for {config.config_id}")

    expected_indices = select_candidate_indices(
        config.steps,
        config.num_candidates,
        t_min=config.candidate_t_min,
        t_max=config.terminal_cutoff,
        time_sampling=config.spacing,
        end_bias_power=config.end_bias_power,
    )
    expected_times = np.asarray(expected_indices, dtype=np.float64) / config.steps
    diagnostics = metadata.get("mintflow_per_sample", [])
    if len(diagnostics) != n:
        raise ValueError(f"Missing per-sample diagnostics for {config.config_id}")
    for sample_index, row in enumerate(diagnostics):
        actual_times = np.asarray([
            float(value)
            for _, value in sorted(
                row["candidate_times"].items(), key=lambda item: int(item[0])
            )
        ])
        if not np.allclose(actual_times, expected_times, rtol=0.0, atol=2e-7):
            raise ValueError(
                f"Candidate grid mismatch at sample {sample_index}: "
                f"{actual_times} != {expected_times}"
            )
        if float(row.get("correction_scale", np.nan)) != config.correction_scale:
            raise ValueError(f"Correction scale mismatch at sample {sample_index}")


def _write_summary(rows: list[dict], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _compute_environments(metadata: dict) -> list[dict]:
    environments = []
    for raw_path in metadata.get("merged_from_shards", []):
        metadata_path = Path(raw_path) / "metadata.json"
        if not metadata_path.is_file():
            continue
        environments.append(
            json.loads(metadata_path.read_text()).get("ablation_execution", {})
        )
    return environments


def main(args: argparse.Namespace) -> None:
    study = study_for(args.scenario)
    manifest_path = study.root / "ablation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run the ablation prepare action first")
    manifest = json.loads(manifest_path.read_text())
    reference = np.load(study.reference, mmap_mode="r")
    if len(reference) != args.num_samples:
        raise ValueError(f"Reference size {len(reference)} != {args.num_samples}")
    cfg = build_diffusion_config(
        task="heat_mass_conservation", config_path="configs/heat.yml"
    )
    # Reuse only if the cache proves it was fitted on this exact reference.
    embedding = fit_or_load_frozen_pod(study.pod, reference, cfg)
    if not isinstance(embedding, FrozenPODEmbedding):
        raise TypeError("Unexpected POD embedding type")
    scenario_target_bank = None
    if study.target_file is not None:
        scenario_target_bank = load_scenario_target_bank(
            study.target_file, "diffusion"
        )
        diffusivities = np.asarray(
            scenario_target_bank.physical_parameters[:, 0], dtype=np.float64
        )
        if len(diffusivities) != args.num_samples:
            raise ValueError("Scenario target bank does not match evaluation size")
    else:
        diffusivities = load_frozen_heat_diffusivities(
            Path(
                "results/heat_mass_12way_benchmark/evaluation/"
                "physical_consistency_metadata.json"
            ),
            args.num_samples,
        )
    rows = []
    reference_hash = file_sha256(study.reference)
    noise_hash = file_sha256(study.noise)
    configs = ablation_design(
        include_projection_interactions=study.key == "type2"
    )[0]
    for config in configs:
        run = _run_path(config, study)
        samples_path, metadata_path = run / "samples.npy", run / "metadata.json"
        if not samples_path.is_file() or not metadata_path.is_file():
            if args.allow_partial:
                print(f"PENDING {config.config_id}")
                continue
            raise FileNotFoundError(f"Incomplete merged run: {run}")
        metadata = json.loads(metadata_path.read_text())
        _validate_metadata(metadata, config, args.num_samples, study)
        samples = np.load(samples_path, mmap_mode="r")
        metrics, per_sample = evaluate_heat_ablation_run(
            samples, reference, embedding, diffusivities, cfg, metadata,
            seed=SEED, scenario_target_bank=scenario_target_bank,
        )
        output = study.root / "runs" / config.config_id
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_id": config.config_id,
            "config": asdict(config),
            "run_path": str(run.resolve()),
            "samples_sha256": file_sha256(samples_path),
            "reference_sha256": reference_hash,
            "shared_noise_sha256": noise_hash,
            "pod_embedding_metadata": embedding.metadata(),
            "compute_environments": _compute_environments(metadata),
            **metrics,
        }
        (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        np.savez_compressed(output / "per_sample_metrics.npz", **per_sample)
        row = {"config_id": config.config_id, **asdict(config), **metrics}
        rows.append(row)
        _write_summary(rows, study.root / "ablation_summary.csv")
        print(
            f"EVALUATED {config.config_id}: FID={metrics['FID']:.6g}, "
            f"KID={metrics['KID']:.6g}, pass={metrics['conservation_pass_rate_at_1e-5']:.3f}"
        )
    manifest["evaluated_configuration_count"] = len(rows)
    manifest["evaluation_complete"] = len(rows) == len(configs)
    manifest["metric_precision"] = "float64"
    manifest["pod_embedding_metadata"] = embedding.metadata()
    manifest["reference_sha256"] = reference_hash
    manifest["shared_noise_sha256"] = noise_hash
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--scenario", choices=("type1", "type2"), default="type1")
    p.add_argument("--allow-partial", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
