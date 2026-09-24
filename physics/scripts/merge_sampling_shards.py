#!/usr/bin/env python3
"""Merge deterministic shared-noise sampling shards into one canonical run."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

PER_SAMPLE_KEYS = (
    "diffusionpde_per_sample",
    "dflow_per_sample",
    "pcfm_per_sample",
    "eci_per_sample",
    "mintflow_per_sample",
    "final_projection_per_sample",
)


def _load_run(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    samples_path = path / "samples.npy"
    metadata_path = path / "metadata.json"
    if not samples_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete shard run: {path}")
    return np.load(samples_path), json.loads(metadata_path.read_text())


def _offset_records(records: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    result = []
    for position, record in enumerate(records):
        row = dict(record)
        row["sample_index"] = offset + int(row.get("sample_index", position))
        result.append(row)
    return result


def merge(shard_runs: list[Path], output_dir: Path, total_samples: int) -> Path:
    loaded = []
    common_signature = None
    for run in shard_runs:
        samples, metadata = _load_run(run.resolve())
        ids = np.asarray(
            metadata.get(
                "sample_ids",
                np.arange(int(metadata.get("sample_offset", 0)),
                          int(metadata.get("sample_offset", 0)) + len(samples)),
            ),
            dtype=np.int64,
        )
        if len(ids) != len(samples):
            raise ValueError(f"sample_ids length mismatch in {run}")
        signature = (
            metadata.get("dataset"), metadata.get("method"), metadata.get("seed"),
            metadata.get("checkpoint_path"), metadata.get("constraint_scenario"),
            metadata.get("constraint_target_id"), metadata.get("constraint_target_file"),
            metadata.get("constraint_target_seed"), metadata.get("initial_noise_path"),
            metadata.get("mintflow_correction_mode"), metadata.get("mintflow_correction_scale"),
            metadata.get("mintflow_forward_steps"), metadata.get("mintflow_num_candidates"),
            metadata.get("mintflow_candidate_t_min"), metadata.get("mintflow_candidate_t_max"),
            metadata.get("mintflow_time_sampling"), metadata.get("mintflow_end_bias_power"),
            metadata.get("mintflow_time_penalty"), metadata.get("mintflow_ridge"),
            metadata.get("mintflow_score_mode"),
            metadata.get("mintflow_interior_penalty"),
            metadata.get("mintflow_correction_weight"),
            metadata.get("mintflow_pseudoinverse_rcond"),
            metadata.get("mintflow_adjoint_chunk_size"),
            metadata.get("final_projection_arg"), metadata.get("intermediate_only_arg"),
            metadata.get("eci_mode_arg"),
        )
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError(f"Shard configuration mismatch: {run}")
        loaded.append((int(ids.min()), ids, samples, metadata, run.resolve()))

    loaded.sort(key=lambda item: item[0])
    all_ids = np.concatenate([item[1] for item in loaded])
    expected = np.arange(total_samples, dtype=np.int64)
    if not np.array_equal(all_ids, expected):
        missing = np.setdiff1d(expected, all_ids)
        duplicates = all_ids[np.flatnonzero(np.diff(np.sort(all_ids)) == 0)]
        raise ValueError(
            f"Shards do not exactly cover [0, {total_samples}); "
            f"missing={missing[:20].tolist()}, duplicates={duplicates[:20].tolist()}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    merged_samples = np.concatenate([item[2] for item in loaded], axis=0)
    np.save(output_dir / "samples.npy", merged_samples)

    # The offset-zero shard owns the publication flow paths (sample IDs 0..K-1).
    first_meta, first_run = loaded[0][3], loaded[0][4]
    metadata = dict(first_meta)
    flow_file = first_meta.get("flow_paths_file")
    if flow_file:
        source = Path(flow_file)
        if not source.is_absolute():
            cwd_relative = source.resolve()
            source = cwd_relative if cwd_relative.exists() else first_run / source
        if source.exists():
            shutil.copy2(source, output_dir / "flow_paths.npz")
            metadata["flow_paths_file"] = str((output_dir / "flow_paths.npz").resolve())

    metadata.update({
        "num_samples": total_samples,
        "sample_offset": 0,
        "sample_ids": expected.tolist(),
        "sample_shape": list(merged_samples.shape),
        "samples_file": "samples.npy",
        "output_dir": str(output_dir.resolve()),
        "merged_from_shards": [str(item[4]) for item in loaded],
        "num_merged_shards": len(loaded),
        "generation_time_s": round(sum(float(item[3].get("generation_time_s", 0.0)) for item in loaded), 2),
        "parallel_wall_time_s": round(max(float(item[3].get("generation_time_s", 0.0)) for item in loaded), 2),
        "peak_gpu_memory_bytes": max((int(item[3].get("peak_gpu_memory_bytes") or 0) for item in loaded), default=0),
        "peak_gpu_memory_reserved_bytes": max((int(item[3].get("peak_gpu_memory_reserved_bytes") or 0) for item in loaded), default=0),
    })
    vf_total = sum(int(item[3].get("vector_field_forward_calls_total") or 0) for item in loaded)
    metadata["vector_field_forward_calls_total"] = vf_total
    metadata["vector_field_forward_calls_per_sample_amortized"] = vf_total / total_samples
    batches = sum(
        (len(item[2]) + int(item[3].get("batch_size", 1)) - 1)
        // int(item[3].get("batch_size", 1))
        for item in loaded
    )
    metadata["vector_field_forward_calls_per_batch_amortized"] = vf_total / batches
    if isinstance(metadata.get("initial_noise_shape"), list):
        metadata["initial_noise_shape"][0] = total_samples

    for key in PER_SAMPLE_KEYS:
        records = []
        for _, ids, _, shard_meta, _ in loaded:
            values = shard_meta.get(key)
            if isinstance(values, list):
                records.extend(_offset_records(values, int(ids.min())))
        if records:
            metadata[key] = records
        else:
            metadata.pop(key, None)

    for key in ("failed_sample_indices", "failure_indices"):
        indices = []
        for _, ids, _, shard_meta, _ in loaded:
            offset = int(ids.min())
            indices.extend(offset + int(value) for value in shard_meta.get(key, []))
        if indices:
            metadata[key] = sorted(indices)
        else:
            metadata.pop(key, None)

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return output_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-run", action="append", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--total-samples", required=True, type=int)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    print(merge(args.shard_run, args.output_dir.resolve(), args.total_samples))
