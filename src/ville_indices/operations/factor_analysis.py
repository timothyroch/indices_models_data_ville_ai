"""PCA/factor-analysis helpers for SoVI-like methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    eigenvectors: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any]


def run_pca(standardized: pd.DataFrame) -> PCAResult:
    """Run PCA on a complete standardized matrix using SVD."""

    if standardized.shape[0] < 2:
        raise ValueError("PCA requires at least 2 observations.")
    if standardized.shape[1] < 1:
        raise ValueError("PCA requires at least 1 usable variable.")
    matrix = standardized.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("PCA input contains NaN or infinite values.")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values**2) / (centered.shape[0] - 1)
    total_variance = float(np.sum(eigenvalues))
    if total_variance <= 0:
        raise ValueError("PCA input has zero total variance.")
    explained = eigenvalues / total_variance
    cumulative = np.cumsum(explained)
    eigenvectors = vt.T
    loadings = eigenvectors * np.sqrt(eigenvalues)
    scores = centered @ eigenvectors
    metadata = {
        "method": "pca_svd",
        "n_observations": int(centered.shape[0]),
        "n_variables": int(centered.shape[1]),
        "total_variance": total_variance,
    }
    return PCAResult(
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        eigenvectors=eigenvectors,
        loadings=loadings,
        scores=scores,
        metadata=metadata,
    )


def retain_factors(
    eigenvalues: np.ndarray,
    *,
    method: str,
    threshold: float = 1.0,
    n_factors: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Determine the number of retained factors from a retention rule."""

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if method == "eigenvalue_gt":
        retained = int(np.sum(eigenvalues > threshold))
    elif method == "fixed_n":
        if n_factors is None:
            raise ValueError("fixed_n factor retention requires n_factors.")
        retained = int(n_factors)
    else:
        raise NotImplementedError(f"Factor retention method '{method}' is not implemented.")

    if retained <= 0:
        raise ValueError(
            f"Factor retention method '{method}' retained zero factors."
        )
    if retained > len(eigenvalues):
        raise ValueError(
            f"Requested {retained} factors, but only {len(eigenvalues)} components are available."
        )
    metadata = {
        "method": method,
        "threshold": threshold if method == "eigenvalue_gt" else None,
        "n_factors_requested": n_factors if method == "fixed_n" else None,
        "n_factors_retained": retained,
    }
    return retained, metadata


def rotated_factor_scores(
    standardized: pd.DataFrame,
    rotated_loadings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute approximate projection-based factor scores after rotation."""

    matrix = standardized.to_numpy(dtype=float)
    loadings = np.asarray(rotated_loadings, dtype=float)
    gram = loadings.T @ loadings
    coefficients = loadings @ np.linalg.pinv(gram)
    scores = matrix @ coefficients
    metadata = {
        "method": "rotated_projection",
        "approximation": (
            "Scores are computed by projecting standardized variables onto "
            "rotated loading-derived coefficients."
        ),
    }
    return scores, coefficients, metadata
