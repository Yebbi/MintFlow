# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
"""
visualization/io.py
====================
Output directory creation and metadata helpers for reproducibility.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def get_git_commit() -> str | None:
    """Return the short git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

def save_metadata_json(path: str | Path, metadata: dict[str, Any]) -> Path:
    """Serialise *metadata* to a pretty-printed JSON file.

    Parameters
    ----------
    path:
        Destination file path (typically ``<run_dir>/metadata.json``).
    metadata:
        Arbitrary JSON-serialisable dict.

    Returns
    -------
    Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for non-standard types (e.g. numpy integers)."""
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):        # numpy array / torch tensor
        return obj.tolist()
    return str(obj)
