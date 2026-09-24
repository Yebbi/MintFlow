# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

"""ExperimentLogger: W&B-only experiment logging abstraction.

Wraps wandb.init and provides a simple interface for logging scalars,
images, config, and checkpoint artifacts from the 1D training loop.

Usage::

    logger = ExperimentLogger(config, config_path='configs/heat.yml')
    logger.log_config(build_run_metadata(config, train_set, test_set, model))
    logger.log_metrics({'train/loss': 0.5, 'train/lr': 3e-4}, step=0)
    logger.log_image('sample/0', img_chw_tensor, step=0)
    logger.finish(final_ckpt_path='logs/test/latest.pt')

W&B is initialised only when ``config.wandb.enabled`` is ``true`` and the
effective mode is not ``'disabled'``.  The env-var ``WANDB_MODE`` takes
precedence over the config key.

Offline usage::

    WANDB_MODE=offline uv run python scripts/training/main.py configs/heat.yml

If W&B is disabled or the run name is not provided, a name is auto-generated
as ``<dataset>-<timestamp>`` (e.g. ``heat-2026-06-23-0814``).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def make_run_name(config_path: str) -> str:
    """Generate a run name like ``heat-2026-06-23-0814`` from a config path."""
    base = os.path.splitext(os.path.basename(config_path))[0]
    ts = datetime.now().strftime('%Y-%m-%d-%H%M')
    return f'{base}-{ts}'


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return 'unknown'


def _gpu_info() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def _dataset_stats(dataset, max_samples: int = 200) -> dict[str, Any]:
    """Compute min/max/mean/std on up to *max_samples* items (subset estimate).

    For large file-backed datasets (e.g. NS at 7.7 GB) this only reads
    *max_samples* individual items from the HDF5 file, never the full array.
    """
    n_total = len(dataset)
    n = min(max_samples, n_total)
    step = max(1, n_total // n)
    indices = list(range(0, n_total, step))[:n]

    vals: list[torch.Tensor] = []
    for idx in indices:
        item = dataset[idx]
        item = item[0] if isinstance(item, (list, tuple)) else item
        vals.append(item.float().cpu())

    arr = torch.stack(vals)
    return {
        'min':       float(arr.min()),
        'max':       float(arr.max()),
        'mean':      float(arr.mean()),
        'std':       float(arr.std()),
        'n_sampled': n,
        'n_total':   n_total,
    }


# ---------------------------------------------------------------------------
# Metadata builder (logged once at run start)
# ---------------------------------------------------------------------------

def build_run_metadata(
    config,
    train_set,
    test_set,
    model,
    stats_max_samples: int = 200,
) -> dict[str, Any]:
    """Collect all run-start metadata into a flat dict for wandb.config."""
    meta: dict[str, Any] = {}

    # --- system ---
    meta['sys/python_version'] = sys.version.split()[0]
    meta['sys/platform']       = platform.platform()
    meta['sys/torch_version']  = torch.__version__
    meta['sys/cuda_available'] = torch.cuda.is_available()
    meta['sys/gpus']           = ', '.join(_gpu_info()) or 'none'
    meta['sys/git_commit']     = _git_commit()

    # --- dataset ---
    ds_cfg = getattr(config, 'datasets', None)
    meta['dataset/name']       = getattr(ds_cfg, 'type', 'unknown') if ds_cfg else 'unknown'
    meta['dataset/train_size'] = len(train_set)
    meta['dataset/test_size']  = len(test_set)

    s0 = train_set[0]
    s0 = s0[0] if isinstance(s0, (list, tuple)) else s0
    meta['dataset/sample_shape'] = list(s0.shape)
    meta['dataset/dtype']        = str(s0.dtype)

    if ds_cfg:
        tr_cfg = getattr(ds_cfg, 'train', None)
        te_cfg = getattr(ds_cfg, 'test',  None)
        meta['dataset/train_file'] = getattr(tr_cfg, 'data_file', 'on-the-fly')
        meta['dataset/test_file']  = getattr(te_cfg, 'data_file', 'on-the-fly')

    # dataset stats on a cheap subset
    tr_stats = _dataset_stats(train_set, max_samples=stats_max_samples)
    for k, v in tr_stats.items():
        meta[f'dataset/train_{k}'] = v

    # --- model ---
    meta['model/class']            = type(model).__name__
    meta['model/encoder_class']    = type(model.model).__name__
    meta['model/total_params']     = sum(p.numel() for p in model.parameters())
    meta['model/trainable_params'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    enc_cfg = getattr(config, 'encoder', None)
    if enc_cfg:
        for k in ('n_modes', 'hidden_channels', 'proj_channels', 'n_layers', 'emb_channels'):
            if hasattr(enc_cfg, k):
                meta[f'encoder/{k}'] = getattr(enc_cfg, k)

    m_cfg = getattr(config, 'model', None)
    if m_cfg:
        for k in ('kernel', 'kernel_length', 'kernel_variance'):
            if hasattr(m_cfg, k):
                meta[f'prior/{k}'] = getattr(m_cfg, k)

    # --- training ---
    tr = getattr(config, 'train', None)
    if tr:
        for k in ('batch_size', 'max_iter', 'log_freq', 'val_freq',
                  'save_freq', 'max_grad_norm', 'seed'):
            if hasattr(tr, k):
                meta[f'train/{k}'] = getattr(tr, k)
        opt = getattr(tr, 'optimizer', None)
        if opt:
            for k in ('type', 'lr', 'weight_decay', 'beta1', 'beta2'):
                if hasattr(opt, k):
                    meta[f'optimizer/{k}'] = getattr(opt, k)
        sch = getattr(tr, 'scheduler', None)
        if sch:
            for k in ('type', 'factor', 'patience', 'min_lr'):
                if hasattr(sch, k):
                    meta[f'scheduler/{k}'] = getattr(sch, k)

    return meta


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """W&B-only experiment logger.

    Initialises a W&B run when ``config.wandb.enabled`` is ``true`` and the
    effective mode is not ``'disabled'``.  All methods are no-ops when W&B is
    disabled, so callers never need to guard against a None logger.

    The env-var ``WANDB_MODE`` overrides ``config.wandb.mode``.
    """

    def __init__(self, config, config_path: str = '') -> None:
        self._run       = None  # wandb.Run or None
        self._log_model = False

        wcfg    = getattr(config, 'wandb', None)
        enabled = bool(getattr(wcfg, 'enabled', False)) if wcfg is not None else False
        mode    = str(getattr(wcfg, 'mode',    'online')) if wcfg is not None else 'online'

        # WANDB_MODE env var takes precedence
        env_mode = os.environ.get('WANDB_MODE', '')
        if env_mode:
            mode = env_mode

        self._log_model = bool(getattr(wcfg, 'log_model', False)) if wcfg is not None else False

        if not enabled or mode == 'disabled':
            return

        import wandb

        run_name = (getattr(wcfg, 'name', None) or None) if wcfg else None
        if not run_name and config_path:
            run_name = make_run_name(config_path)

        try:
            self._run = wandb.init(
                project = getattr(wcfg, 'project', 'pretrain') if wcfg else 'pretrain',
                entity  = getattr(wcfg, 'entity',  None)       if wcfg else None,
                group   = getattr(wcfg, 'group',   None)       if wcfg else None,
                name    = run_name,
                mode    = mode,
                tags    = list(getattr(wcfg, 'tags', []) or []) if wcfg else [],
                reinit  = 'finish_previous',
            )
        except Exception as exc:
            print(f'[ExperimentLogger] W&B init failed: {exc}. Continuing without logging.')

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log_config(self, config_dict: dict[str, Any]) -> None:
        """Update W&B run config with the given metadata dict."""
        if self._run is not None:
            self._run.config.update(config_dict, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log scalar metrics to W&B."""
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_image(self, key: str, img_tensor: torch.Tensor, step: int) -> None:
        """Log a (C, H, W) float32 tensor as a W&B image."""
        if self._run is not None:
            import wandb
            arr = img_tensor.permute(1, 2, 0).cpu().numpy()
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            self._run.log({key: wandb.Image(arr)}, step=step)

    def log_summary(self, key: str, value: Any) -> None:
        """Set a W&B run summary value (e.g. final checkpoint path)."""
        if self._run is not None:
            self._run.summary[key] = value

    def finish(self, final_ckpt_path: str | None = None) -> None:
        """Finish the W&B run.

        If *final_ckpt_path* is provided and ``wandb.log_model: true``,
        uploads the checkpoint file as a W&B artifact before finishing.
        """
        if self._run is not None:
            if final_ckpt_path and os.path.exists(final_ckpt_path) and self._log_model:
                import wandb
                art = wandb.Artifact('checkpoint', type='model')
                art.add_file(final_ckpt_path)
                self._run.log_artifact(art)
            self._run.finish()
