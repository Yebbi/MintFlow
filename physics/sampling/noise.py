"""
sampling/noise.py
===================
Shared initial-noise utilities for Phase 3B same-seed comparisons.

When ``--reuse-initial-noise`` is supplied, every method must integrate from
exactly the same ``u0[i]`` regardless of ``--batch-size`` — batching only
slices a pre-generated ``(N, *sample_dims)`` tensor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from models.functional import make_grid


def generate_initial_noise(
    ffm,
    num_samples: int,
    sample_dims: list[int] | tuple[int, ...],
    device: str,
    seed: int,
) -> torch.Tensor:
    """Generate ``num_samples`` GP-prior initial noises using the same
    ``ffm.gp.sample`` call as the method wrappers (single draw, batch-size
    independent). Returns CPU float tensor ``(N, *sample_dims)``."""
    dims = list(sample_dims)
    torch.manual_seed(seed)
    grid = make_grid(tuple(dims), device)
    z0 = ffm.gp.sample(grid, dims, n_samples=num_samples).detach().cpu().float()
    return z0


def save_initial_noise(path: Union[str, Path], z0: torch.Tensor, metadata: dict[str, Any]) -> Path:
    """Save ``initial_noise.npy`` + ``initial_noise_metadata.json`` beside *path*."""
    path = Path(path)
    if path.suffix == ".npy":
        noise_path = path
        meta_path = path.with_name("initial_noise_metadata.json")
    else:
        path.mkdir(parents=True, exist_ok=True)
        noise_path = path / "initial_noise.npy"
        meta_path = path / "initial_noise_metadata.json"

    arr = z0.detach().cpu().numpy() if isinstance(z0, torch.Tensor) else np.asarray(z0)
    np.save(str(noise_path), arr)
    meta = dict(metadata)
    meta.setdefault("initial_noise_shape", list(arr.shape))
    meta.setdefault("dtype", str(arr.dtype))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return noise_path


def load_initial_noise(
    path: Union[str, Path],
    device: Optional[str] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load shared initial noise from ``initial_noise.npy`` (or a directory
    containing it) and optional sidecar metadata."""
    path = Path(path)
    if path.is_dir():
        noise_path = path / "initial_noise.npy"
        meta_path = path / "initial_noise_metadata.json"
    else:
        noise_path = path
        meta_path = path.with_name("initial_noise_metadata.json")

    arr = np.load(str(noise_path))
    z0 = torch.from_numpy(arr).float()
    if device is not None:
        z0 = z0.to(device)

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    metadata.setdefault("initial_noise_path", str(noise_path))
    metadata.setdefault("initial_noise_shape", list(arr.shape))
    return z0, metadata


def slice_initial_noise(
    initial_noise: torch.Tensor,
    start: int,
    end: int,
    device: str,
) -> torch.Tensor:
    """Return ``initial_noise[start:end]`` on *device* (preserves sample order)."""
    return initial_noise[start:end].to(device)


def shared_noise_metadata_fields(
    initial_noise_path: Optional[str],
    initial_noise: Optional[torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    """Standard metadata block for runs using shared initial noise."""
    if initial_noise is None:
        return {"shared_initial_noise": False}
    shape = list(initial_noise.shape) if hasattr(initial_noise, "shape") else None
    return {
        "shared_initial_noise": True,
        "initial_noise_path": initial_noise_path,
        "initial_noise_shape": shape,
        "initial_noise_seed": seed,
    }
