# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
"""
visualization/sampling.py
==========================
Helpers for loading pre-trained FFM checkpoints and generating samples.

Sampling here uses vanilla ``FFM.sample()`` (Dopri5); constrained methods are
orchestrated by ``sampling.methods``.

Checkpoint conventions
-----------------------
1D datasets (.pt)
    latest.pt  : {'model': state_dict, 'step': int}
                 weights_only=True safe.  Config must be supplied externally.
    20000.pt   : {'config': EasyDict, 'model': ..., 'optimizer': ...,
                  'scheduler': ..., 'avg_val_loss': float}
                 weights_only=True fails (EasyDict); fall back to
                 weights_only=False with a warning.

"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from visualization.datasets import (
    SUPPORTED_DATASETS,
    default_config_for_dataset,
)

# ---------------------------------------------------------------------------
# Default checkpoint paths
# ---------------------------------------------------------------------------

_CKPT_MAP: dict[str, str] = {
    "diffusion": "logs/diffusion/latest.pt",
    "rd1d":      "logs/rd1d/latest.pt",
}


def default_checkpoint_for_dataset(dataset_name: str) -> str:
    """Return the default checkpoint path for *dataset_name*.

    Raises ValueError for unknown datasets.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {', '.join(SUPPORTED_DATASETS)}"
        )
    return _CKPT_MAP[dataset_name]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _safe_torch_load(path: str, device: str) -> dict:
    """Load a .pt checkpoint, trying weights_only=True first.

    Numbered checkpoints embed EasyDict objects, so weights_only=True fails
    for them.  We fall back to weights_only=False with a warning in that case.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # EasyDict in numbered checkpoints triggers UnpicklingError with
        # weights_only=True — fall back silently (only trusted local files).
        return torch.load(path, map_location=device, weights_only=False)


def load_ffm_model(
    dataset_name: str,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
    device: str = "cuda",
) -> tuple[Any, Any, dict]:
    """Load a pre-trained FFM model from a checkpoint.

    Parameters
    ----------
    dataset_name:
        ``diffusion`` or ``rd1d``.
    checkpoint_path:
        Path to the checkpoint file.  If *None*, the default for
        *dataset_name* is used.
    config_path:
        Path to the YAML config.  Required when *checkpoint_path* is a
        ``latest.pt`` (which does not embed a config).  If *None* and the
        checkpoint does not embed a config, the default config for
        *dataset_name* is used.
    device:
        Target device string, e.g. ``'cuda'`` or ``'cpu'``.

    Returns
    -------
    model : FFM
        The loaded model in eval mode on *device*.
    config : EasyDict
        The resolved training config.
    meta : dict
        Metadata: checkpoint path, step, checkpoint type, device.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    if checkpoint_path is None:
        checkpoint_path = default_checkpoint_for_dataset(dataset_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"For dataset '{dataset_name}', run pre-training first or supply "
            f"--checkpoint with a valid path."
        )

    return _load_1d_pt(dataset_name, checkpoint_path, config_path, device)


def _load_1d_pt(
    dataset_name: str,
    checkpoint_path: str,
    config_path: str | None,
    device: str,
) -> tuple[Any, Any, dict]:
    from models import get_flow_model
    from scripts.training.utils import load_config

    ck = _safe_torch_load(checkpoint_path, device)

    # Resolve config: prefer embedded, then explicit, then default
    if "config" in ck:
        config = ck["config"]
        resolved_config_path = f"<embedded in {Path(checkpoint_path).name}>"
    else:
        if config_path is None:
            config_path = default_config_for_dataset(dataset_name)
        config = load_config(config_path)
        resolved_config_path = config_path

    model = get_flow_model(config.model, config.encoder)
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    meta = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_type":  "1d_pt",
        "checkpoint_step":  ck.get("step"),
        "config_source":    resolved_config_path,
        "device":           device,
    }
    return model, config, meta




# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_samples_batched(
    model,
    config,
    num_samples: int,
    batch_size: int = 16,
    n_eval: int = 10,
    device: str = "cuda",
    seed: int = 0,
    initial_noise: "torch.Tensor | None" = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> torch.Tensor:
    """Generate *num_samples* samples from *model* in batches.

    Uses vanilla ``FFM.sample()`` (dopri5 ODE solver) when *initial_noise* is
    ``None``.  When *initial_noise* is provided ``(N, *sample_dims)``, each
    batch integrates from the corresponding slice (batch-size independent).
    """
    import torch
    from torchdiffeq import odeint


    dims = list(config.sample_dims)
    if initial_noise is None:
        torch.manual_seed(seed)
    else:
        if initial_noise.shape[0] < num_samples:
            raise ValueError(
                f"initial_noise has {initial_noise.shape[0]} samples but "
                f"{num_samples} requested"
            )

    model.eval()
    all_samples: list[torch.Tensor] = []
    remaining = num_samples
    start = 0

    with torch.no_grad():
        while remaining > 0:
            cur = min(batch_size, remaining)
            if initial_noise is None:
                batch = model.sample(
                    n_sample=cur,
                    n_eval=n_eval,
                    dims=dims,
                    device=device,
                    rtol=rtol,
                    atol=atol,
                    return_traj=False,
                )
            else:
                u0 = initial_noise[start : start + cur].to(device)
                ts = torch.linspace(0, 1, n_eval + 1, device=device)
                xs = odeint(model.model, u0, ts, method="dopri5", rtol=rtol, atol=atol)
                batch = xs[-1].detach()
            all_samples.append(batch.cpu())
            remaining -= cur
            start += cur
            generated_so_far = num_samples - remaining
            print(
                f"  generated {generated_so_far}/{num_samples} samples "
                f"(batch {cur})",
                flush=True,
            )

    return torch.cat(all_samples, dim=0)    # (num_samples, *dims)
