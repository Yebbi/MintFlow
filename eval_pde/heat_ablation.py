"""Evaluation primitives for the isolated Heat MintFlow ablations.

All distribution metrics use the immutable numerical-reference POD embedding.
Constraint and physical-consistency metrics are evaluated in raw field space.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import PDEConfig, load_grid
from .distances import (
    empirical_wasserstein2,
    frechet_feature_distance,
    median_pairwise_distance,
    mmd,
    polynomial_kid,
)
from .embeddings import FrozenPODEmbedding
from .type1_evaluation import constraint_values, scenario2_constraint_components


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_max": float(finite.max()),
    }


def heat_physical_consistency(
    samples: np.ndarray,
    diffusivities: np.ndarray,
    cfg: PDEConfig,
    *,
    chunk_size: int = 100,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Vectorized raw-space Heat residual and same-IC solver error.

    The per-sample diffusivity is frozen from the aligned Vanilla FFM pool,
    matching the production Type-1 evaluation protocol. The Fourier solution
    is exact up to time/grid discretization for each method's own initial
    column and the periodic boundary condition.
    """
    values = np.asarray(samples)
    alpha = np.asarray(diffusivities, dtype=np.float64)
    if values.shape != (len(alpha), cfg.nx, cfg.nt):
        raise ValueError(
            f"Expected samples shape ({len(alpha)}, {cfg.nx}, {cfg.nt}), got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(alpha)):
        raise ValueError("Heat physical-consistency inputs must be finite")
    x, t = load_grid(cfg)
    dx = float(x[1] - x[0])
    period = float(cfg.x_max - cfg.x_min)
    elapsed = np.asarray(t - t[0], dtype=np.float64)
    wave_number = 2.0 * np.pi * np.fft.rfftfreq(cfg.nx, d=period / cfg.nx)
    residual_rms = np.empty(len(values), dtype=np.float64)
    simulation_relative_l2 = np.empty(len(values), dtype=np.float64)

    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        field = np.asarray(values[start:stop], dtype=np.float64)
        temporal = np.gradient(field, t, axis=2, edge_order=2)
        laplacian = (
            np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)
        ) / dx**2
        residual = temporal - alpha[start:stop, None, None] * laplacian
        residual_rms[start:stop] = np.sqrt(np.mean(residual**2, axis=(1, 2)))

        coefficients = np.fft.rfft(field[:, :, 0], axis=1)
        decay = np.exp(
            -alpha[start:stop, None, None]
            * wave_number[None, :, None] ** 2
            * elapsed[None, None, :]
        )
        simulation = np.fft.irfft(
            coefficients[:, :, None] * decay, n=cfg.nx, axis=1
        )
        difference_norm = np.linalg.norm(
            (field - simulation).reshape(stop - start, -1), axis=1
        )
        simulation_norm = np.linalg.norm(
            simulation.reshape(stop - start, -1), axis=1
        )
        simulation_relative_l2[start:stop] = difference_norm / np.maximum(
            simulation_norm, np.finfo(np.float64).tiny
        )

    per_sample = {
        "pde_residual_rms": residual_rms,
        "simulation_relative_l2": simulation_relative_l2,
    }
    summary = {
        **_summary(residual_rms, "pde_residual_rms"),
        **_summary(simulation_relative_l2, "simulation_relative_l2"),
    }
    return summary, per_sample


def evaluate_heat_ablation_run(
    samples: np.ndarray,
    numerical_reference: np.ndarray,
    embedding: FrozenPODEmbedding,
    diffusivities: np.ndarray,
    cfg: PDEConfig,
    metadata: Mapping[str, Any],
    *,
    seed: int,
    scenario_target_bank=None,
    feasibility_tolerance: float = 1e-5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    generated = np.asarray(samples)
    reference = np.asarray(numerical_reference)
    if generated.shape != reference.shape or generated.shape != (
        len(reference), cfg.nx, cfg.nt
    ):
        raise ValueError(
            f"Generated/reference arrays must share shape (N,{cfg.nx},{cfg.nt}); "
            f"got {generated.shape} and {reference.shape}"
        )
    generated_features = embedding.transform(generated)
    reference_features = embedding.transform(reference)
    bandwidth = median_pairwise_distance(reference_features, seed=seed)
    mmd_result = mmd(
        generated_features,
        reference_features,
        seed=seed,
        median_distance=bandwidth,
    )
    if scenario_target_bank is None:
        components = {
            "conservation": constraint_values(generated, cfg),
        }
        combined = components["conservation"]
    else:
        components = scenario2_constraint_components(
            generated, cfg, scenario_target_bank
        )
        combined = components["combined_normalized"]
    conservation = components["conservation"]
    physical, per_sample = heat_physical_consistency(
        generated, diffusivities, cfg
    )
    diagnostics = list(metadata.get("mintflow_per_sample", []))
    if len(diagnostics) != len(generated):
        raise ValueError(
            "MintFlow diagnostics must align one-to-one with samples: "
            f"{len(diagnostics)} != {len(generated)}"
        )
    selected_times = np.asarray([row["s_star"] for row in diagnostics], dtype=np.float64)
    correction_norms = np.asarray(
        [row["correction_norm"] for row in diagnostics], dtype=np.float64
    )
    mintflow_residual_inf_before = np.asarray(
        [row["terminal_residual_inf_before"] for row in diagnostics],
        dtype=np.float64,
    )
    mintflow_residual_inf_after = np.asarray(
        [row["terminal_residual_inf_after"] for row in diagnostics],
        dtype=np.float64,
    )
    selected_ranks = np.empty(len(diagnostics), dtype=np.float64)
    candidate_counts = np.empty(len(diagnostics), dtype=np.float64)
    selected_correction_terms = np.empty(len(diagnostics), dtype=np.float64)
    selected_time_terms = np.empty(len(diagnostics), dtype=np.float64)
    selected_weighted_time_terms = np.empty(len(diagnostics), dtype=np.float64)
    score_margins = np.empty(len(diagnostics), dtype=np.float64)
    expected_candidate_count = min(
        int(metadata["mintflow_num_candidates"]),
        int(np.floor(float(metadata["mintflow_candidate_t_max"]) * int(
            metadata["mintflow_forward_steps"]
        )))
        - int(np.ceil(float(metadata["mintflow_candidate_t_min"]) * int(
            metadata["mintflow_forward_steps"]
        )))
        + 1,
    )
    for sample_index, row in enumerate(diagnostics):
        indices = sorted(int(key) for key in row["candidate_scores"])
        best_index = int(row["best_idx"])
        if len(indices) != expected_candidate_count or best_index not in indices:
            raise ValueError(
                f"Candidate diagnostics mismatch at sample {sample_index}: "
                f"got {len(indices)}, expected {expected_candidate_count}"
            )
        selected_ranks[sample_index] = indices.index(best_index) / max(
            len(indices) - 1, 1
        )
        candidate_counts[sample_index] = len(indices)
        scores = np.asarray(
            [float(row["candidate_scores"][str(index)]) for index in indices],
            dtype=np.float64,
        )
        breakdown = row["candidate_score_breakdown"][str(best_index)]
        correction_term = float(breakdown["correction_term"])
        time_term = float(breakdown["time_term"])
        recomputed_score = correction_term + float(
            metadata["mintflow_time_penalty"]
        ) * time_term
        if not np.isclose(
            recomputed_score, float(row["candidate_scores"][str(best_index)]),
            rtol=2e-6, atol=2e-7,
        ):
            raise ValueError(
                f"Candidate score is not connected to lambda at sample {sample_index}"
            )
        if int(indices[int(np.argmin(scores))]) != best_index:
            raise ValueError(f"Stored best candidate is not argmin at sample {sample_index}")
        selected_correction_terms[sample_index] = correction_term
        selected_time_terms[sample_index] = time_term
        selected_weighted_time_terms[sample_index] = (
            float(metadata["mintflow_time_penalty"]) * time_term
        )
        score_margins[sample_index] = (
            float(np.partition(scores, 1)[1] - scores.min())
            if len(scores) > 1 else float("nan")
        )

    projection_rows = list(metadata.get("final_projection_per_sample", []))
    if bool(metadata.get("post_sampling_final_projection_enabled", False)):
        if len(projection_rows) != len(generated):
            raise ValueError("Enabled final projection lacks per-sample diagnostics")
        projection_before = np.asarray(
            [row["initial_residual_norm"] for row in projection_rows],
            dtype=np.float64,
        )
        projection_after = np.asarray(
            [row["final_residual_norm"] for row in projection_rows],
            dtype=np.float64,
        )
        projection_l2 = np.asarray(
            [row.get("projection_correction_l2", np.nan) for row in projection_rows],
            dtype=np.float64,
        )
    else:
        projection_before = np.full(len(generated), np.nan)
        projection_after = np.full(len(generated), np.nan)
        projection_l2 = np.zeros(len(generated), dtype=np.float64)
    total_time = float(metadata["generation_time_s"])
    total_nfe = int(metadata["vector_field_forward_calls_total"])
    metrics = {
        "n": int(len(generated)),
        "FID": frechet_feature_distance(generated_features, reference_features),
        "KID": polynomial_kid(generated_features, reference_features),
        "MMD": float(np.sqrt(mmd_result["mmd2_biased_clipped"])),
        "W2": empirical_wasserstein2(generated_features, reference_features),
        "conservation_R_inf_mean": float(conservation.mean()),
        "conservation_R_inf_std": float(conservation.std(ddof=1)),
        "conservation_R_inf_median": float(np.median(conservation)),
        "conservation_R_inf_max": float(conservation.max()),
        "conservation_pass_rate_at_1e-5": float(
            np.mean(conservation <= feasibility_tolerance)
        ),
        "combined_normalized_R_inf_mean": float(combined.mean()),
        "combined_normalized_R_inf_std": float(combined.std(ddof=1)),
        "combined_normalized_R_inf_max": float(combined.max()),
        "combined_pass_rate_at_1e-5": float(
            np.mean(combined <= feasibility_tolerance)
        ),
        **physical,
        "generation_time_s": total_time,
        "time_per_sample_s": total_time / len(generated),
        "total_NFE": total_nfe,
        "NFE_per_sample": total_nfe / len(generated),
        "peak_gpu_memory_GiB": float(metadata["peak_gpu_memory_bytes"]) / 1024.0**3,
        **_summary(selected_times, "selected_time"),
        "selected_time_at_upper_fraction": float(
            np.mean(np.isclose(
                selected_times,
                float(metadata["mintflow_candidate_t_max"]),
                rtol=0.0,
                atol=0.5 / float(metadata["mintflow_forward_steps"]) + 1e-7,
            ))
        ),
        **_summary(correction_norms, "correction_norm"),
        **_summary(
            mintflow_residual_inf_before,
            "mintflow_pre_correction_normalized_R_inf",
        ),
        **_summary(
            mintflow_residual_inf_after,
            "mintflow_pre_projection_normalized_R_inf",
        ),
        "selected_candidate_rank_mean": float(selected_ranks.mean()),
        "selected_candidate_earliest_fraction": float(np.mean(selected_ranks == 0.0)),
        "selected_candidate_interior_fraction": float(np.mean(
            (selected_ranks > 0.0) & (selected_ranks < 1.0)
        )),
        "selected_candidate_latest_fraction": float(np.mean(selected_ranks == 1.0)),
        **_summary(selected_correction_terms, "selected_score_correction_term"),
        **_summary(selected_time_terms, "selected_score_time_term"),
        **_summary(
            selected_weighted_time_terms, "selected_score_weighted_time_term"
        ),
        **_summary(score_margins, "candidate_score_margin"),
        **_summary(projection_before, "final_projection_residual_before"),
        **_summary(projection_after, "final_projection_residual_after"),
        **_summary(projection_l2, "final_projection_correction_l2"),
        "diagnostic_candidate_count": int(candidate_counts[0]),
        "diagnostic_score_recomputation_passed": True,
        "diagnostic_config_connection_passed": True,
        "pod_feature_dim": embedding.feature_dim,
        "pod_reference_median_distance": bandwidth,
    }
    per_sample.update({
        "conservation_R_inf": np.asarray(conservation, dtype=np.float64),
        "selected_time": selected_times,
        "correction_norm": correction_norms,
        "mintflow_pre_correction_normalized_R_inf": mintflow_residual_inf_before,
        "mintflow_pre_projection_normalized_R_inf": mintflow_residual_inf_after,
        "combined_normalized_R_inf": np.asarray(combined, dtype=np.float64),
        "selected_candidate_rank": selected_ranks,
        "selected_score_correction_term": selected_correction_terms,
        "selected_score_time_term": selected_time_terms,
        "selected_score_weighted_time_term": selected_weighted_time_terms,
        "candidate_score_margin": score_margins,
        "final_projection_residual_before": projection_before,
        "final_projection_residual_after": projection_after,
        "final_projection_correction_l2": projection_l2,
    })
    if scenario_target_bank is not None:
        ic = components["initial_condition"]
        boundary = components["boundary_condition"]
        metrics.update({
            **_summary(ic, "IC_R_inf"),
            **_summary(boundary, "BC_R_inf"),
            "IC_pass_rate_at_1e-5": float(np.mean(ic <= feasibility_tolerance)),
            "BC_pass_rate_at_1e-5": float(
                np.mean(boundary <= feasibility_tolerance)
            ),
        })
        per_sample.update({
            "IC_R_inf": np.asarray(ic, dtype=np.float64),
            "BC_R_inf": np.asarray(boundary, dtype=np.float64),
        })
    return metrics, per_sample


def load_frozen_heat_diffusivities(path: Path, expected_n: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    values = np.asarray(payload["heat_diffusivities"], dtype=np.float64)
    if values.shape != (expected_n,):
        raise ValueError(f"Expected {expected_n} frozen diffusivities, got {values.shape}")
    return values
