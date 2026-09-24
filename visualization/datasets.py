# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
"""
visualization/datasets.py
==========================
Helpers for loading PCFM dataset splits by name and returning the vis kwargs
embedded in each training config.

All dataset loading is delegated to the existing datasets.get_dataset()
factory — no dataset classes are duplicated here.
"""
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ("diffusion", "rd1d")

_CONFIG_MAP: dict[str, str] = {
    "diffusion": "configs/heat.yml",
    "rd1d":      "configs/rd1d.yml",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def default_config_for_dataset(dataset_name: str) -> str:
    """Return the default config YAML path for *dataset_name*.

    Raises
    ------
    ValueError
        If *dataset_name* is not in SUPPORTED_DATASETS.
    """
    _check_dataset_name(dataset_name)
    return _CONFIG_MAP[dataset_name]


def load_dataset_split(
    dataset_name: str,
    split: str,
    config_path: str | None = None,
) -> tuple[Dataset, Any, str]:
    """Load one split of a PCFM dataset.

    Uses the existing ``datasets.get_dataset()`` factory, which builds both
    train and test sets from a single config.  Only the requested split is
    returned.

    Parameters
    ----------
    dataset_name:
        ``diffusion`` or ``rd1d``.
    split:
        ``'train'`` or ``'test'``.
    config_path:
        Path to the YAML config.  If *None*, the default for *dataset_name*
        is used.

    Returns
    -------
    dataset : torch.utils.data.Dataset
    config  : EasyDict — the full loaded config
    config_path : str — resolved path that was actually used
    """
    from datasets import get_dataset
    from scripts.training.utils import load_config

    _check_dataset_name(dataset_name)
    _check_split(split)

    if config_path is None:
        config_path = default_config_for_dataset(dataset_name)

    config = load_config(config_path)

    train_set, test_set = get_dataset(config.datasets)

    if split == "train":
        return train_set, config, config_path
    else:
        return test_set, config, config_path


def get_vis_kwargs(config: Any) -> dict[str, Any]:
    """Extract the ``vis:`` block from a config as a plain dict.

    Returns an empty dict if the config has no ``vis`` key.
    """
    vis = getattr(config, "vis", None)
    if vis is None:
        return {}
    # EasyDict supports .items(); convert to plain dict
    try:
        return dict(vis.items())
    except AttributeError:
        return dict(vis)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check_dataset_name(name: str) -> None:
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported: {', '.join(SUPPORTED_DATASETS)}"
        )


def _check_split(split: str) -> None:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
