"""Rotation utilities for factor-analysis methods."""

from __future__ import annotations

from typing import Any

import numpy as np


def varimax(
    loadings: np.ndarray,
    *,
    gamma: float = 1.0,
    normalize: bool = True,
    max_iter: int = 500,
    tol: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Perform orthogonal varimax rotation.

    This follows the common SVD-based implementation of Kaiser-normalized
    varimax. It returns rotated loadings, the rotation matrix, and metadata.
    """

    phi = np.asarray(loadings, dtype=float)
    if phi.ndim != 2:
        raise ValueError("loadings must be a 2D array.")
    n_variables, n_factors = phi.shape
    if n_factors == 0:
        raise ValueError("Cannot rotate zero factors.")
    if n_factors == 1:
        return phi.copy(), np.eye(1), {
            "method": "varimax",
            "normalize": normalize,
            "gamma": gamma,
            "max_iter": max_iter,
            "tol": tol,
            "iterations": 0,
            "converged": True,
            "single_factor_behavior": "rotation_not_needed",
        }

    working = phi.copy()
    communalities = None
    if normalize:
        communalities = np.sqrt(np.sum(working**2, axis=1))
        communalities[communalities == 0] = 1.0
        working = working / communalities[:, np.newaxis]

    rotation = np.eye(n_factors)
    previous = 0.0
    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        projected = working @ rotation
        u, singular_values, vh = np.linalg.svd(
            working.T
            @ (
                projected**3
                - (gamma / n_variables)
                * projected
                @ np.diag(np.diag(projected.T @ projected))
            )
        )
        rotation = u @ vh
        current = float(np.sum(singular_values))
        iterations = iteration
        if previous and current / previous < 1.0 + tol:
            converged = True
            break
        previous = current

    rotated = working @ rotation
    if normalize and communalities is not None:
        rotated = rotated * communalities[:, np.newaxis]

    metadata = {
        "method": "varimax",
        "normalize": normalize,
        "gamma": gamma,
        "max_iter": max_iter,
        "tol": tol,
        "iterations": iterations,
        "converged": converged,
    }
    return rotated, rotation, metadata
