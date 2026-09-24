"""
sampling/flow_paths.py
========================
Opt-in flow-path snapshot helpers (Phase 3B).

Snapshots are taken at fixed flow-time fractions ``tau in [0, 1]`` using the
same fixed-step Euler grid as the underlying sampler (not PDE physical time).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def parse_flow_path_fractions(raw: str) -> list[float]:
    fracs = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not fracs:
        raise ValueError("flow_path_fractions must contain at least one value")
    for f in fracs:
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"flow path fraction must be in [0, 1], got {f}")
    return sorted(set(fracs))


def fraction_to_step_index(frac: float, n_step: int) -> int:
    """Map flow fraction ``tau`` to an Euler step index in ``[0, n_step]``."""
    return int(round(float(frac) * n_step))


def init_path_recorder(
    fractions: list[float],
    n_step: int,
    sample_shape: tuple[int, ...],
    max_path_samples: int,
    batch_size: int,
) -> tuple[dict[float, list[torch.Tensor]], list[int]]:
    """Per-batch path recorder. Returns (fraction->list of tensors, sample_indices)."""
    n_record = min(batch_size, max_path_samples)
    indices = list(range(n_record))
    store: dict[float, list[torch.Tensor]] = {f: [] for f in fractions}
    return store, indices


def maybe_record_state(
    store: dict[float, list[torch.Tensor]],
    fractions: list[float],
    step_idx: int,
    n_step: int,
    u: torch.Tensor,
    record_mask: Optional[list[bool]] = None,
) -> None:
    """Record ``u[i]`` when ``step_idx`` matches a requested fraction."""
    for frac in fractions:
        if step_idx != fraction_to_step_index(frac, n_step):
            continue
        if record_mask is None:
            store[frac].append(u.detach().cpu())
        else:
            sel = u[record_mask].detach().cpu()
            if sel.shape[0]:
                store[frac].append(sel)


def finalize_path_batch(
    store: dict[float, list[torch.Tensor]],
    fractions: list[float],
    sample_shape: tuple[int, ...],
) -> np.ndarray:
    """Stack one batch into ``(B, n_fractions, *sample_shape)``."""
    if not store[fractions[0]]:
        return np.empty((0, len(fractions), *sample_shape), dtype=np.float32)
    b = store[fractions[0]][0].shape[0]
    out = np.empty((b, len(fractions), *sample_shape), dtype=np.float32)
    for j, frac in enumerate(fractions):
        out[:, j] = torch.cat(store[frac], dim=0).numpy()
    return out


def merge_path_arrays(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def save_flow_paths_npz(
    path: str,
    path_samples: np.ndarray,
    path_sample_indices: np.ndarray,
    flow_path_fractions: list[float],
) -> None:
    np.savez(
        path,
        path_samples=path_samples,
        path_sample_indices=path_sample_indices,
        flow_path_fractions=np.asarray(flow_path_fractions, dtype=np.float64),
    )
