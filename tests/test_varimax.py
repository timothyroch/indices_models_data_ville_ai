import numpy as np

from ville_indices.operations.rotation import varimax


def test_varimax_preserves_shape_and_is_finite() -> None:
    loadings = np.array(
        [
            [0.8, 0.1],
            [0.7, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )

    rotated, rotation, metadata = varimax(loadings)

    assert rotated.shape == loadings.shape
    assert rotation.shape == (2, 2)
    assert np.isfinite(rotated).all()
    assert metadata["method"] == "varimax"
    assert metadata["iterations"] > 0
