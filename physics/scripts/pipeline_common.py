"""Shared orchestration primitives for production benchmark entry points."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BenchmarkConfiguration:
    label: str
    method: str
    extra: tuple[str, ...] = ()


def benchmark_configurations(vanilla_n_eval: int) -> tuple[BenchmarkConfiguration, ...]:
    """Return the canonical comparison matrix shared by Heat and RD1D."""
    n_eval = str(vanilla_n_eval)
    return (
        BenchmarkConfiguration("vanilla_ffm", "vanilla", ("--n-eval", n_eval)),
        BenchmarkConfiguration("pcfm_standard", "pcfm"),
        BenchmarkConfiguration("pcfm_intermediate_only", "pcfm", ("--intermediate-only",)),
        BenchmarkConfiguration("eci_native_standard", "eci", ("--eci-mode", "native")),
        BenchmarkConfiguration(
            "eci_native_intermediate_only", "eci",
            ("--eci-mode", "native", "--intermediate-only"),
        ),
        BenchmarkConfiguration("eci_gn_standard", "eci", ("--eci-mode", "gn")),
        BenchmarkConfiguration(
            "eci_gn_intermediate_only", "eci",
            ("--eci-mode", "gn", "--intermediate-only"),
        ),
        BenchmarkConfiguration("diffusionpde", "diffusionpde"),
        BenchmarkConfiguration("dflow", "dflow"),
        BenchmarkConfiguration(
            "mintflow_pseudoinverse_no_final_projection", "mintflow",
            ("--mintflow-correction-mode", "pseudoinverse", "--no-final-projection"),
        ),
        BenchmarkConfiguration(
            "mintflow_pseudoinverse_with_final_projection", "mintflow",
            ("--mintflow-correction-mode", "pseudoinverse", "--final-projection"),
        ),
        BenchmarkConfiguration(
            "mintflow_damped_no_final_projection", "mintflow",
            ("--mintflow-correction-mode", "damped", "--no-final-projection"),
        ),
        BenchmarkConfiguration(
            "mintflow_damped_with_final_projection", "mintflow",
            ("--mintflow-correction-mode", "damped", "--final-projection"),
        ),
        BenchmarkConfiguration(
            "standalone_final_projection", "final_projection", ("--n-eval", n_eval)
        ),
    )


def find_completed_run(
    label_root: Path,
    seed: int,
    num_samples: int,
    required_metadata: dict[str, Any] | None = None,
) -> Path | None:
    """Find the newest complete run satisfying the requested protocol."""
    candidates: list[Path] = []
    metadata_paths = label_root.rglob("metadata.json") if label_root.exists() else ()
    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        run = metadata_path.parent
        if (
            int(metadata.get("seed", -1)) == seed
            and int(metadata.get("num_samples", -1)) == num_samples
            and all(
                metadata.get(key) == value
                for key, value in (required_metadata or {}).items()
            )
            and (run / "samples.npy").is_file()
        ):
            candidates.append(run)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_logged(
    command: list[str],
    log_path: Path,
    dry_run: bool = False,
    *,
    cwd: Path = REPO,
) -> None:
    """Print a command and, unless dry-running, execute it with a dedicated log."""
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
