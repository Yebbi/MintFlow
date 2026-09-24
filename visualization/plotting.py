# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
"""
visualization/plotting.py
=========================
Standalone figure-saving wrappers around the existing draw() utilities in
scripts/training/vis_utils.py.

All functions are headless (matplotlib 'agg' backend) and require no GPU,
W&B, or TensorBoard.

Coordinate convention reminder
-------------------------------
draw_2d expects a tensor of shape (nx, nt):
    axis-0 → spatial dimension x  (rows in imshow)
    axis-1 → temporal dimension t  (columns in imshow)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure agg backend is active before any other matplotlib import.
# vis_utils already calls plt.switch_backend('agg') at import time.
from scripts.training.vis_utils import draw


def tensor_to_uint8_hwc(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert the CHW float32 tensor returned by draw() to a HWC uint8 array.

    Parameters
    ----------
    img_tensor:
        Tensor of shape (C, H, W) with values in roughly [0, 1].
        May be on any device.

    Returns
    -------
    np.ndarray of shape (H, W, C), dtype uint8.
    """
    arr = img_tensor.detach().cpu().float()
    arr = arr.permute(1, 2, 0).numpy()          # CHW → HWC
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def save_draw(
    x: torch.Tensor | np.ndarray,
    path: str | Path,
    fmt: str = "png",
    **vis_kwargs: Any,
) -> Path:
    """Draw tensor *x* with the existing draw() function and save to *path*.

    Parameters
    ----------
    x:
        Sample tensor with shape ``(nx, nt)``. Numpy arrays are accepted and
        converted to float32 tensors.
    path:
        Destination file path.  The extension is appended / replaced with
        *fmt* if the path has no recognised image extension.
    fmt:
        ``'png'`` (default) or ``'pdf'``.
    **vis_kwargs:
        Passed directly to draw() — e.g. ``vmin``, ``vmax``, ``cmap``,
        ``downsample``, ``nrow``.

    Returns
    -------
    Path object pointing to the saved file.
    """
    from PIL import Image

    path = Path(path)
    # Normalise extension
    if path.suffix.lower() not in (".png", ".pdf", ".jpg", ".jpeg"):
        path = path.with_suffix(f".{fmt.lstrip('.')}")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Accept numpy input
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))

    x = x.detach().cpu().float()

    # draw() returns (C, H, W) float tensor in [0, 1]
    img_tensor = draw(x, **vis_kwargs)
    arr = tensor_to_uint8_hwc(img_tensor)

    if path.suffix.lower() == ".pdf":
        # For PDF we still go through PIL (raster PDF); for vector PDF
        # a future enhancement could use matplotlib directly.
        Image.fromarray(arr).save(str(path), format="PDF")
    else:
        Image.fromarray(arr).save(str(path))

    return path


def save_sample_image(
    sample: torch.Tensor | np.ndarray,
    path: str | Path,
    vis_kwargs: dict[str, Any] | None = None,
    fmt: str = "png",
) -> Path:
    """Convenience wrapper: save one dataset sample as an image.

    Parameters
    ----------
    sample:
        Single sample tensor.  No batch dimension.
    path:
        Output file path.
    vis_kwargs:
        Visualisation parameters dict (``vmin``, ``vmax``, ``cmap``, …).
        If *None*, uses draw() defaults.
    fmt:
        Output format, ``'png'`` or ``'pdf'``.

    Returns
    -------
    Path to the saved file.
    """
    vis_kwargs = vis_kwargs or {}
    return save_draw(sample, path, fmt=fmt, **vis_kwargs)
