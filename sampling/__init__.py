"""Sampling package with dependency-light lazy public exports.
Public names are preserved but resolved only when accessed, so importing a
constraint task does not require the optional ODE and FNO stacks.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Residuals": ("sampling.constraints", "Residuals"),
    "FFM_sampler": ("sampling.ffm_sampler", "FFM_sampler"),
    "run_vanilla_sampling": ("sampling.methods", "run_vanilla_sampling"),
    "run_pcfm_sampling": ("sampling.methods", "run_pcfm_sampling"),
    "run_eci_sampling": ("sampling.methods", "run_eci_sampling"),
    "run_mintflow_sampling": ("sampling.methods", "run_mintflow_sampling"),
    "compute_jacobian": ("sampling.pcfm_sampling", "compute_jacobian"),
    "make_grid": ("sampling.pcfm_sampling", "make_grid"),
    "pcfm_batched": ("sampling.pcfm_sampling", "pcfm_batched"),
    "forward_solve_with_trajectory": ("sampling.adjoint_correction", "forward_solve_with_trajectory"),
    "terminal_residual_and_jacobian": ("sampling.adjoint_correction", "terminal_residual_and_jacobian"),
    "select_candidate_indices": ("sampling.adjoint_correction", "select_candidate_indices"),
    "parse_candidate_time_range": ("sampling.adjoint_correction", "parse_candidate_time_range"),
    "discrete_euler_adjoint": ("sampling.adjoint_correction", "discrete_euler_adjoint"),
    "pseudoinverse_minimum_norm_correction": ("sampling.adjoint_correction", "pseudoinverse_minimum_norm_correction"),
    "damped_minimum_norm_correction": ("sampling.adjoint_correction", "damped_minimum_norm_correction"),
    "solve_linearized_correction": ("sampling.adjoint_correction", "solve_linearized_correction"),
    "select_best_candidate": ("sampling.adjoint_correction", "select_best_candidate"),
    "compute_candidate_scores": ("sampling.adjoint_correction", "compute_candidate_scores"),
    "boundary_penalty": ("sampling.adjoint_correction", "boundary_penalty"),
    "reintegrate_euler_suffix": ("sampling.adjoint_correction", "reintegrate_euler_suffix"),
    "make_sampling_run_dir": ("sampling.io", "make_sampling_run_dir"),
    "save_samples": ("sampling.io", "save_samples"),
    "save_metadata": ("sampling.io", "save_metadata"),
    "save_metrics_json": ("sampling.io", "save_metrics_json"),
    "save_metrics_csv": ("sampling.io", "save_metrics_csv"),
}

__all__ = list(_EXPORTS)
__version__ = "0.1.0"


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
