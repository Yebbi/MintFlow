"""Shared, parallel-filesystem-safe artifact utilities."""
from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the artifact in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write CSV rows with a stable union of their fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact into dictionaries without implicit type coercion."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))
