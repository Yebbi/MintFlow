#!/usr/bin/env python3
"""Create standalone dual-axis figures for the Heat Type-2 ablation study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  Registers publication plotting styles.
from matplotlib.ticker import ScalarFormatter

# Easy-to-replace data dictionaries. These example values are populated from the
# completed Heat Type-2 ablation. ``constraint`` is the mean combined normalized
# conservation + IC + BC violation.
INTEGRATION_RESOLUTION_DATA = {
    "steps": [50, 100, 200, 400],
    "fid": [0.0158215723, 0.0029853264, 0.0012550333, 0.0007280924],
    "constraint": [5.2385063e-5, 1.0106740e-5, 3.5713047e-6, 3.8455732e-6],
}

OPTIMAL_TIME_REGULARIZATION_DATA = {
    "lambda": [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "fid": [
        0.0012544710,
        0.0012544710,
        0.0012544710,
        0.0012550333,
        0.0013068679,
        0.0015821099,
    ],
    "constraint": [
        3.5804876e-6,
        3.5804876e-6,
        3.5804876e-6,
        3.5713047e-6,
        3.4832202e-6,
        3.2239400e-6,
    ],
}

PRIMARY_COLOR = "#0072B2"
SECONDARY_COLOR = "#009E73"


def _validate_data(data: dict[str, list[float]], x_key: str) -> None:
    """Fail early when substituted data columns have inconsistent lengths."""
    lengths = {key: len(data[key]) for key in (x_key, "fid", "constraint")}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Data columns must have equal lengths; received {lengths}")
    if not lengths[x_key]:
        raise ValueError("At least one ablation point is required")


def _lambda_tick(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.round(np.log10(value)))
    if np.isclose(value, 10.0**exponent):
        return "1" if exponent == 0 else rf"$10^{{{exponent}}}$"
    return f"{value:g}"


def _style_dual_axes(
    ax: plt.Axes,
    constraint_ax: plt.Axes,
    title: str,
    xlabel: str,
    legend_location: str,
) -> None:
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("FID score", color=PRIMARY_COLOR, fontsize=12)
    constraint_ax.set_ylabel(
        "Mean constraint violation", color=SECONDARY_COLOR, fontsize=12
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.tick_params(axis="y", colors=PRIMARY_COLOR)
    constraint_ax.tick_params(axis="y", colors=SECONDARY_COLOR, labelsize=10)
    ax.spines["left"].set_color(PRIMARY_COLOR)
    constraint_ax.spines["right"].set_color(SECONDARY_COLOR)
    ax.grid(True, which="major", color="0.85", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    # Keep small metric values readable without embedding scale factors in labels.
    for axis in (ax.yaxis, constraint_ax.yaxis):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axis.set_major_formatter(formatter)
        axis.get_offset_text().set_fontsize(10)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = constraint_ax.get_legend_handles_labels()
    ax.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc=legend_location,
        frameon=False,
        fontsize=10,
    )


def plot_integration_resolution(
    data: dict[str, list[float]], output_path: Path
) -> None:
    """Plot FID and constraint violation against the Euler step count."""
    _validate_data(data, "steps")
    steps = np.asarray(data["steps"], dtype=int)

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        constraint_ax = ax.twinx()

        ax.plot(
            steps,
            data["fid"],
            color=PRIMARY_COLOR,
            linestyle="-",
            marker="o",
            linewidth=2.0,
            markersize=6,
            label="Distribution Fidelity: FID",
        )
        constraint_ax.plot(
            steps,
            data["constraint"],
            color=SECONDARY_COLOR,
            linestyle="--",
            marker="s",
            linewidth=2.0,
            markersize=6,
            label="Constraint: mean constraint violation",
        )
        ax.set_xticks(steps)
        _style_dual_axes(
            ax,
            constraint_ax,
            title="Integration Resolution Ablation",
            xlabel="Number of steps",
            legend_location="upper right",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_optimal_time_regularization(
    data: dict[str, list[float]], output_path: Path
) -> None:
    """Plot FID and constraint violation against optimal-time regularization."""
    _validate_data(data, "lambda")
    lambdas = np.asarray(data["lambda"], dtype=float)
    positions = np.arange(lambdas.size)
    tick_labels = [_lambda_tick(value) for value in lambdas]

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        constraint_ax = ax.twinx()

        ax.plot(
            positions,
            data["fid"],
            color=PRIMARY_COLOR,
            linestyle="-",
            marker="o",
            linewidth=2.0,
            markersize=6,
            label="Distribution Fidelity: FID",
        )
        constraint_ax.plot(
            positions,
            data["constraint"],
            color=SECONDARY_COLOR,
            linestyle="--",
            marker="s",
            linewidth=2.0,
            markersize=6,
            label="Constraint: mean constraint violation",
        )
        ax.set_xticks(positions, labels=tick_labels)
        _style_dual_axes(
            ax,
            constraint_ax,
            title=r"Optimal-Time Regularization ($\lambda$)",
            xlabel=r"Regularization parameter $\lambda$",
            legend_location="center left",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ablation/heat_type2/figures"),
        help="Directory for the two independent 300-DPI PNG figures.",
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    plot_integration_resolution(
        INTEGRATION_RESOLUTION_DATA,
        args.output_dir / "integration_resolution_ablation.png",
    )
    plot_optimal_time_regularization(
        OPTIMAL_TIME_REGULARIZATION_DATA,
        args.output_dir / "optimal_time_regularization_ablation.png",
    )


if __name__ == "__main__":
    main()
