"""The four frozen-POD distances retained by Scenario 1/2 evaluation."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

EPS = 1e-12

DEFAULT_MMD_BANDWIDTH_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def _as_matching_feature_matrices(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert two finite, two-dimensional feature matrices."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2-D feature matrices with matching feature dimensions")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("feature matrices must contain only finite values")
    return X, Y


def _psd_square_root(matrix: np.ndarray) -> np.ndarray:
    """Symmetric square root of a positive-semidefinite matrix."""
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def frechet_feature_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Gaussian Frechet distance (FID analogue) in a physical feature space.

    This is the squared 2-Wasserstein distance between Gaussian fits to the
    two feature distributions.  The caller, rather than this function,
    defines the embedding; the benchmark uses reference-fit PDE features.
    """
    X, Y = _as_matching_feature_matrices(X, Y)
    if min(len(X), len(Y)) < 2:
        return float("nan")
    mean_x, mean_y = X.mean(axis=0), Y.mean(axis=0)
    cov_x = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    cov_y = np.atleast_2d(np.cov(Y, rowvar=False, ddof=1))
    sqrt_x = _psd_square_root(cov_x)
    middle_sqrt = _psd_square_root(sqrt_x @ cov_y @ sqrt_x)
    value = float(
        np.dot(mean_x - mean_y, mean_x - mean_y)
        + np.trace(cov_x + cov_y - 2.0 * middle_sqrt)
    )
    # Round-off can make identical/near-identical inputs slightly negative.
    return max(value, 0.0)


def polynomial_kid(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased degree-three polynomial-kernel MMD^2 (KID analogue).

    The kernel is ``(x @ y / d + 1)^3``.  As an unbiased finite-sample
    estimator, KID may be slightly negative even though population MMD^2 is
    non-negative.
    """
    X, Y = _as_matching_feature_matrices(X, Y)
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return float("nan")
    dimension = X.shape[1]
    k_xx = (X @ X.T / dimension + 1.0) ** 3
    k_yy = (Y @ Y.T / dimension + 1.0) ** 3
    k_xy = (X @ Y.T / dimension + 1.0) ** 3
    within_x = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    within_y = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    return float(within_x + within_y - 2.0 * k_xy.mean())


def empirical_wasserstein2(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact equal-weight empirical Wasserstein-2 distance in feature space.

    For the benchmark's equal-size sample sets this solves the optimal
    one-to-one assignment using the Hungarian algorithm and returns the
    square root of its mean squared Euclidean transport cost.
    """
    X, Y = _as_matching_feature_matrices(X, Y)
    if len(X) != len(Y):
        raise ValueError("exact empirical W2 currently requires equal sample counts")
    if not len(X):
        return float("nan")
    squared_cost = cdist(X, Y, metric="sqeuclidean")
    rows, columns = linear_sum_assignment(squared_cost)
    return float(np.sqrt(np.mean(squared_cost[rows, columns])))


 # ---------------------------------------------------------------------------
# MMD (unbiased multiscale RBF)
# ---------------------------------------------------------------------------

def median_pairwise_distance(X: np.ndarray, max_samples: int = 1000, seed: int = 0) -> float:
    """Median pairwise Euclidean distance of *X* (subsampled to
    ``max_samples`` rows for tractability on large reference sets)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_samples, replace=False)
        X = X[idx]
    if X.shape[0] < 2:
        return 1.0
    d = pdist(X, metric="euclidean")
    med = float(np.median(d))
    return med if med > 0 else 1.0


def _rbf_kernel_mean(sqdist: np.ndarray, bandwidths: list[float]) -> np.ndarray:
    """Average RBF kernels so the multiscale kernel remains unit-bounded."""
    if not bandwidths:
        raise ValueError("at least one MMD bandwidth is required")
    k = np.zeros_like(sqdist)
    for bw in bandwidths:
        bw = max(bw, EPS)
        k += np.exp(-sqdist / (2.0 * bw ** 2))
    return k / len(bandwidths)


def mmd(
    gen_features: np.ndarray,
    ref_features: np.ndarray,
    bandwidth_scales: tuple[float, ...] = DEFAULT_MMD_BANDWIDTH_SCALES,
    seed: int = 0,
    median_distance: float | None = None,
) -> dict[str, float]:
    """Unbiased multiscale-RBF MMD^2 between *gen_features* and
    *ref_features*.

    The multiscale kernel is the **mean** (not sum) of the component RBF
    kernels, keeping ``k(x, x) == 1`` regardless of how many bandwidths are
    requested.  By default the median heuristic is fit on ``ref_features``.
    Benchmarks comparing several reference distributions should pass one
    externally calibrated ``median_distance`` so the metric uses a fixed,
    symmetric kernel in every table.
    """
    X, Y = _as_matching_feature_matrices(gen_features, ref_features)
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        return {
            "mmd2_unbiased": float("nan"), "mmd2_clipped": float("nan"),
            "mmd2_biased": float("nan"), "mmd2_biased_clipped": float("nan"),
            "median_distance": float("nan"),
        }

    med = (
        median_pairwise_distance(Y, seed=seed)
        if median_distance is None
        else float(median_distance)
    )
    if not np.isfinite(med) or med <= 0:
        raise ValueError("median_distance must be finite and positive")
    bandwidths = [scale * med for scale in bandwidth_scales]

    XX = cdist(X, X, metric="sqeuclidean")
    YY = cdist(Y, Y, metric="sqeuclidean")
    XY = cdist(X, Y, metric="sqeuclidean")

    Kxx = _rbf_kernel_mean(XX, bandwidths)
    Kyy = _rbf_kernel_mean(YY, bandwidths)
    Kxy = _rbf_kernel_mean(XY, bandwidths)

    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = Kxy.sum() / (m * n)

    mmd2_unbiased = float(sum_xx + sum_yy - 2.0 * sum_xy)
    # The V-statistic includes self-similarities and is the squared RKHS
    # distance between the two empirical kernel mean embeddings.  Unlike the
    # unbiased U-statistic, it is non-negative in exact arithmetic and can be
    # square-rooted to report an actual finite-sample MMD distance.
    mmd2_biased = float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return {
        "mmd2_unbiased": mmd2_unbiased,
        "mmd2_clipped": max(mmd2_unbiased, 0.0),
        "mmd2_biased": mmd2_biased,
        "mmd2_biased_clipped": max(mmd2_biased, 0.0),
        "median_distance": med,
    }
