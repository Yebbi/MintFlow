# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
"""
sampling/io.py
==============
I/O helpers for structured sampling experiment outputs.

Default output layout::

    outputs/sampling/<dataset>/<method>/
        ckpt_<stem>_seed<s>_neval<n>_n<N>_<timestamp>/
            samples.npy
            metadata.json
            constraint_metrics.json
            constraint_metrics.csv
            previews/
                sample_0000.png
                ...
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from visualization.io import save_metadata_json


def make_sampling_run_dir(
    output_dir: str | Path,
    dataset: str,
    method: str,
    checkpoint: str | None = None,
    seed: int | None = None,
    n_eval: int | None = None,
    num_samples: int | None = None,
    timestamp: str | None = None,
    extra: str | None = None,
) -> Path:
    """Create and return a unique sub-directory for one sampling run.

    Layout::

        <output_dir>/<dataset>/<method>/<run_name>/

    where ``run_name`` is assembled from the non-None keyword arguments::

        ckpt_latest_seed0_neval10_n256_20260630_103000

    Parameters
    ----------
    output_dir   : Root output directory (e.g. ``outputs/sampling``).
    dataset      : Dataset name (``diffusion`` or ``rd1d``).
    method       : Sampling method label (``vanilla``, ``pcfm``, …).
    checkpoint   : Checkpoint file path — stem is embedded in the run name.
    seed         : Random seed.
    n_eval       : ODE evaluation steps.
    num_samples  : Number of generated samples.
    timestamp    : ISO-like timestamp string.  Auto-generated if *None*.
    extra        : Optional extra path component (e.g. a task/target-id
                   slug) inserted before the timestamp, so different global
                   constraint tasks/targets cannot collide even when every
                   other naming component (checkpoint/seed/n_eval/num_samples)
                   is identical.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts: list[str] = []
    if checkpoint is not None:
        parts.append(f"ckpt_{Path(checkpoint).stem}")
    if seed is not None:
        parts.append(f"seed{seed}")
    if n_eval is not None:
        parts.append(f"neval{n_eval}")
    if num_samples is not None:
        parts.append(f"n{num_samples}")
    if extra:
        parts.append(extra)
    parts.append(timestamp)
    run_dir = Path(output_dir) / dataset / method / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_samples(
    path: str | Path,
    samples,
    format: str = "npy",
) -> Path:
    """Save generated samples to *path*.

    Parameters
    ----------
    path    : Destination file (extension is added or normalised based on *format*).
    samples : ``torch.Tensor`` or ``np.ndarray`` of shape ``(N, *dims)``.
    format  : ``'npy'`` (default) or ``'h5'`` / ``'hdf5'``.

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = samples.numpy() if hasattr(samples, "numpy") else np.asarray(samples)

    if format == "npy":
        if path.suffix != ".npy":
            path = path.with_suffix(".npy")
        np.save(str(path), arr)
    elif format in ("h5", "hdf5"):
        if path.suffix not in (".h5", ".hdf5"):
            path = path.with_suffix(".h5")
        import h5py
        with h5py.File(str(path), "w") as f:
            f.create_dataset("samples", data=arr, compression="gzip", compression_opts=4)
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'npy' or 'h5'.")
    return path


def save_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    """Serialise *metadata* to a pretty-printed JSON file.

    Delegates to :func:`visualization.io.save_metadata_json`.
    """
    return save_metadata_json(path, metadata)


def save_metrics_json(path: str | Path, metrics: dict[str, Any]) -> Path:
    """Save aggregate constraint metrics as a pretty-printed JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    return path


def save_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Save per-sample constraint metrics as a CSV file.

    Parameters
    ----------
    rows : List of dicts with consistent keys, one dict per generated sample.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# (empty)\n")
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
