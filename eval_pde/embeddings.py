"""Frozen linear embeddings for distributional evaluation of PDE fields."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import PDEConfig, load_grid

DEFAULT_POD_MAX_COMPONENTS = 20
DEFAULT_POD_VARIANCE_THRESHOLD = 0.999
POD_CACHE_VERSION = 1


def _array_fingerprint(values: np.ndarray) -> str:
    """Content hash used to bind a frozen embedding to its calibration pool."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("Quadrature grids must be strictly increasing one-dimensional arrays")
    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def field_quadrature_weights(cfg: PDEConfig) -> np.ndarray:
    """Tensor-product integration weights for a field stored as ``(x, t)``."""
    x_grid, t_grid = load_grid(cfg)
    if cfg.periodic_x:
        # The periodic Heat grid omits the duplicated right endpoint.
        x_weights = np.full(len(x_grid), (cfg.x_max - cfg.x_min) / len(x_grid))
    else:
        x_weights = _trapezoid_weights(x_grid)
    t_weights = _trapezoid_weights(t_grid)
    return np.multiply.outer(x_weights, t_weights)


class FrozenPODEmbedding:
    """Reference-independent evaluation coordinates fitted on one calibration pool.

    The transform is the orthogonal projection of the mean-centered field onto
    a POD basis under the grid's discrete physical ``L2(x,t)`` inner product.
    No coordinate-wise whitening, nonlinear features, clipping, or refitting is
    performed when a sample set is transformed.
    """

    def __init__(
        self,
        max_components: int = DEFAULT_POD_MAX_COMPONENTS,
        variance_threshold: float = DEFAULT_POD_VARIANCE_THRESHOLD,
    ) -> None:
        if max_components < 1:
            raise ValueError("max_components must be positive")
        if not 0.0 < variance_threshold <= 1.0:
            raise ValueError("variance_threshold must lie in (0, 1]")
        self.max_components = int(max_components)
        self.variance_threshold = float(variance_threshold)
        self._fitted = False

    def fit(self, calibration_samples: np.ndarray, cfg: PDEConfig) -> "FrozenPODEmbedding":
        samples = np.asarray(calibration_samples)
        expected = (cfg.nx, cfg.nt)
        if samples.ndim != 3 or samples.shape[1:] != expected or len(samples) < 2:
            raise ValueError(
                f"Calibration samples must have shape (N, {cfg.nx}, {cfg.nt}) with N >= 2; "
                f"got {samples.shape}"
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("POD calibration samples must contain only finite raw fields")

        samples64 = np.asarray(samples, dtype=np.float64)
        self.sample_shape_ = expected
        self.equation_ = cfg.name
        self.calibration_n_ = len(samples64)
        self.calibration_fingerprint_ = _array_fingerprint(samples)
        self.sqrt_quadrature_weights_ = np.sqrt(field_quadrature_weights(cfg)).reshape(-1)
        weighted = samples64.reshape(len(samples64), -1) * self.sqrt_quadrature_weights_
        self.mean_weighted_ = weighted.mean(axis=0)
        centered = weighted - self.mean_weighted_
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)

        rank_tolerance = (
            singular_values[0] * max(centered.shape) * np.finfo(np.float64).eps
            if singular_values.size
            else 0.0
        )
        numerical_rank = int(np.sum(singular_values > rank_tolerance))
        if numerical_rank == 0:
            raise ValueError("POD calibration pool has zero centered numerical rank")
        variances = singular_values[:numerical_rank] ** 2
        cumulative = np.cumsum(variances) / variances.sum()
        threshold_components = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
        retained = min(self.max_components, numerical_rank, threshold_components)

        self.components_ = np.asarray(right_vectors[:retained], dtype=np.float64)
        self.explained_variance_ratio_ = variances[:retained] / variances.sum()
        self.retained_variance_ratio_ = float(self.explained_variance_ratio_.sum())
        self.numerical_rank_ = numerical_rank
        self._fitted = True
        return self

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Project raw fields without modifying or refitting the frozen basis."""
        if not self._fitted:
            raise RuntimeError("FrozenPODEmbedding must be fit before transform")
        values = np.asarray(samples)
        if values.ndim != 3 or values.shape[1:] != self.sample_shape_:
            raise ValueError(
                f"Samples must have shape (N, {self.sample_shape_[0]}, {self.sample_shape_[1]}); "
                f"got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("POD transform does not impute, clip, or discard non-finite raw fields")
        weighted = (
            np.asarray(values, dtype=np.float64).reshape(len(values), -1)
            * self.sqrt_quadrature_weights_
        )
        return (weighted - self.mean_weighted_) @ self.components_.T

    @property
    def feature_dim(self) -> int:
        if not self._fitted:
            raise RuntimeError("FrozenPODEmbedding has not been fit")
        return int(len(self.components_))

    def metadata(self) -> dict:
        if not self._fitted:
            raise RuntimeError("FrozenPODEmbedding has not been fit")
        return {
            "cache_version": POD_CACHE_VERSION,
            "kind": "quadrature_weighted_centered_pod",
            "equation": self.equation_,
            "sample_shape": list(self.sample_shape_),
            "calibration_n": self.calibration_n_,
            "calibration_fingerprint_sha256": self.calibration_fingerprint_,
            "max_components": self.max_components,
            "variance_threshold": self.variance_threshold,
            "numerical_rank": self.numerical_rank_,
            "retained_components": self.feature_dim,
            "retained_variance_ratio": self.retained_variance_ratio_,
            "coordinate_scaling": "sqrt tensor-product physical quadrature weights",
            "centering": "weighted calibration-pool mean",
            "whitening": False,
            "nonlinear_features": False,
            "score_clipping": False,
        }

    def save(self, path: Path) -> None:
        """Persist the complete immutable transform and its provenance."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted POD embedding")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            components=self.components_,
            mean_weighted=self.mean_weighted_,
            sqrt_quadrature_weights=self.sqrt_quadrature_weights_,
            explained_variance_ratio=self.explained_variance_ratio_,
            metadata=np.asarray(json.dumps(self.metadata())),
        )

    @classmethod
    def load(cls, path: Path) -> "FrozenPODEmbedding":
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            if metadata.get("cache_version") != POD_CACHE_VERSION:
                raise ValueError("Unsupported frozen POD cache version")
            result = cls(metadata["max_components"], metadata["variance_threshold"])
            result.components_ = payload["components"].copy()
            result.mean_weighted_ = payload["mean_weighted"].copy()
            result.sqrt_quadrature_weights_ = payload["sqrt_quadrature_weights"].copy()
            result.explained_variance_ratio_ = payload["explained_variance_ratio"].copy()
        result.sample_shape_ = tuple(metadata["sample_shape"])
        result.equation_ = metadata["equation"]
        result.calibration_n_ = int(metadata["calibration_n"])
        result.calibration_fingerprint_ = metadata["calibration_fingerprint_sha256"]
        result.retained_variance_ratio_ = float(metadata["retained_variance_ratio"])
        result.numerical_rank_ = int(metadata["numerical_rank"])
        result._fitted = True
        return result


def fit_or_load_frozen_pod(
    path: Path,
    calibration_samples: np.ndarray,
    cfg: PDEConfig,
    *,
    max_components: int = DEFAULT_POD_MAX_COMPONENTS,
    variance_threshold: float = DEFAULT_POD_VARIANCE_THRESHOLD,
) -> FrozenPODEmbedding:
    """Reuse a cache only when its full calibration content and setup match."""
    fingerprint = _array_fingerprint(np.asarray(calibration_samples))
    path = Path(path)
    if path.exists():
        try:
            cached = FrozenPODEmbedding.load(path)
            metadata = cached.metadata()
            if (
                metadata["calibration_fingerprint_sha256"] == fingerprint
                and metadata["equation"] == cfg.name
                and tuple(metadata["sample_shape"]) == (cfg.nx, cfg.nt)
                and metadata["max_components"] == max_components
                and metadata["variance_threshold"] == variance_threshold
            ):
                return cached
        except (KeyError, OSError, ValueError):
            pass
    embedding = FrozenPODEmbedding(max_components, variance_threshold).fit(
        calibration_samples, cfg
    )
    embedding.save(path)
    return embedding
