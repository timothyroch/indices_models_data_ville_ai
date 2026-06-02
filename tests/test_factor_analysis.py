import numpy as np
import pandas as pd
import pytest

from ville_indices.operations.factor_analysis import (
    retain_factors,
    rotated_factor_scores,
    run_pca,
)


def test_pca_outputs_shapes_and_variance() -> None:
    frame = pd.DataFrame(
        {
            "a": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "b": [-0.8, -0.4, 0.0, 0.4, 0.8],
            "c": [1.0, 0.5, 0.0, -0.5, -1.0],
        }
    )

    result = run_pca(frame)

    assert result.eigenvalues.shape == (3,)
    assert result.loadings.shape == (3, 3)
    assert result.scores.shape == (5, 3)
    assert np.isclose(result.explained_variance_ratio.sum(), 1.0)


def test_factor_retention_fixed_and_eigenvalue_rules() -> None:
    eigenvalues = np.array([2.5, 1.2, 0.4])

    fixed, fixed_meta = retain_factors(eigenvalues, method="fixed_n", n_factors=2)
    gt, gt_meta = retain_factors(eigenvalues, method="eigenvalue_gt", threshold=1.0)

    assert fixed == 2
    assert fixed_meta["method"] == "fixed_n"
    assert gt == 2
    assert gt_meta["threshold"] == 1.0


def test_factor_retention_zero_factors_fails() -> None:
    with pytest.raises(ValueError, match="retained zero factors"):
        retain_factors(np.array([0.4, 0.2]), method="eigenvalue_gt", threshold=1.0)


def test_rotated_factor_scores_shape() -> None:
    standardized = pd.DataFrame(
        {"a": [-1.0, 0.0, 1.0], "b": [1.0, 0.0, -1.0], "c": [0.5, 0.0, -0.5]}
    )
    loadings = np.array([[0.8, 0.1], [0.7, 0.2], [0.1, 0.9]])

    scores, coefficients, metadata = rotated_factor_scores(standardized, loadings)

    assert scores.shape == (3, 2)
    assert coefficients.shape == (3, 2)
    assert metadata["method"] == "rotated_projection"
