"""
Contract tests for the reusable fusion component projection primitive.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_component_projection.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            fusion/
                component_projection.py

This suite freezes the projection-layer contract independently from
concat-projection fusion and orchestration.

The tested contract includes:

- controlled activation and normalization vocabularies;
- exact baseline operation order;
- strict dimension, dtype, device, and finiteness validation;
- empty-batch support;
- deterministic evaluation behavior;
- dtype conversion through module casting;
- finite forward and backward passes;
- architecture and parameter fingerprint separation;
- state-dict round trips;
- finite-parameter corruption detection.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.component_projection import (
    COMPONENT_PROJECTION_SCHEMA_VERSION,
    ComponentProjection,
    ComponentProjectionActivation,
    ComponentProjectionNormalization,
)


INPUT_DIM = 5
OUTPUT_DIM = 7
COMPONENT_NAME = "static_state"


# =============================================================================
# Helpers
# =============================================================================


def _projection(
    *,
    input_dim: int = INPUT_DIM,
    output_dim: int = OUTPUT_DIM,
    component_name: str = COMPONENT_NAME,
    activation: (
        ComponentProjectionActivation | str
    ) = ComponentProjectionActivation.GELU,
    normalization: (
        ComponentProjectionNormalization | str
    ) = ComponentProjectionNormalization.LAYER_NORM,
    dropout: float = 0.0,
) -> ComponentProjection:
    return ComponentProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        component_name=component_name,
        activation=activation,
        normalization=normalization,
        dropout=dropout,
    )


def _values(
    rows: int = 4,
    *,
    width: int = INPUT_DIM,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> torch.Tensor:
    values = (
        torch.arange(
            rows * width,
            dtype=dtype,
        )
        .reshape(rows, width)
        / 10.0
    )
    values.requires_grad_(requires_grad)
    return values


# =============================================================================
# Published schema and controlled vocabularies
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        COMPONENT_PROJECTION_SCHEMA_VERSION,
        str,
    )
    assert COMPONENT_PROJECTION_SCHEMA_VERSION.strip()


def test_activation_vocabulary_is_exact() -> None:
    assert tuple(
        member.value
        for member in ComponentProjectionActivation
    ) == ("gelu",)


def test_normalization_vocabulary_is_exact() -> None:
    assert tuple(
        member.value
        for member in ComponentProjectionNormalization
    ) == (
        "none",
        "layer_norm",
    )


# =============================================================================
# Constructor and baseline factory
# =============================================================================


def test_constructor_preserves_contract() -> None:
    projection = _projection(
        dropout=0.25,
    )

    assert projection.input_dim == INPUT_DIM
    assert projection.output_dim == OUTPUT_DIM
    assert projection.component_name == COMPONENT_NAME
    assert projection.activation is (
        ComponentProjectionActivation.GELU
    )
    assert projection.normalization is (
        ComponentProjectionNormalization.LAYER_NORM
    )
    assert projection.dropout == 0.25


def test_constructor_normalizes_string_enums() -> None:
    projection = _projection(
        activation="gelu",
        normalization="none",
    )

    assert projection.activation is (
        ComponentProjectionActivation.GELU
    )
    assert projection.normalization is (
        ComponentProjectionNormalization.NONE
    )


def test_layer_norm_structure_is_explicit() -> None:
    projection = _projection(
        normalization=(
            ComponentProjectionNormalization.LAYER_NORM
        ),
        dropout=0.2,
    )

    assert isinstance(projection.linear, nn.Linear)
    assert isinstance(
        projection.activation_layer,
        nn.GELU,
    )
    assert isinstance(
        projection.normalization_layer,
        nn.LayerNorm,
    )
    assert isinstance(
        projection.dropout_layer,
        nn.Dropout,
    )
    assert projection.dropout_layer.p == 0.2


def test_no_normalization_uses_identity() -> None:
    projection = _projection(
        normalization=(
            ComponentProjectionNormalization.NONE
        ),
    )

    assert isinstance(
        projection.normalization_layer,
        nn.Identity,
    )


def test_baseline_factory_with_layer_norm() -> None:
    projection = ComponentProjection.baseline(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        component_name=COMPONENT_NAME,
        dropout=0.1,
        layer_norm=True,
    )

    assert projection.activation is (
        ComponentProjectionActivation.GELU
    )
    assert projection.normalization is (
        ComponentProjectionNormalization.LAYER_NORM
    )
    assert projection.dropout == 0.1


def test_baseline_factory_without_layer_norm() -> None:
    projection = ComponentProjection.baseline(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        component_name=COMPONENT_NAME,
        layer_norm=False,
    )

    assert projection.normalization is (
        ComponentProjectionNormalization.NONE
    )
    assert isinstance(
        projection.normalization_layer,
        nn.Identity,
    )


def test_baseline_factory_matches_direct_contract() -> None:
    baseline = ComponentProjection.baseline(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        component_name=COMPONENT_NAME,
        dropout=0.2,
        layer_norm=True,
    )
    direct = ComponentProjection(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        component_name=COMPONENT_NAME,
        activation="gelu",
        normalization="layer_norm",
        dropout=0.2,
    )

    assert baseline.architecture_dict() == (
        direct.architecture_dict()
    )
    assert baseline.architecture_fingerprint() == (
        direct.architecture_fingerprint()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("input_dim", 0),
        ("input_dim", -1),
        ("input_dim", True),
        ("input_dim", 1.5),
        ("output_dim", 0),
        ("output_dim", -1),
        ("output_dim", True),
        ("output_dim", 1.5),
    ),
)
def test_constructor_rejects_invalid_dimensions(
    field: str,
    value: Any,
) -> None:
    kwargs: dict[str, Any] = {
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "component_name": COMPONENT_NAME,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match="positive integer"):
        ComponentProjection(**kwargs)


@pytest.mark.parametrize(
    "component_name",
    (
        "",
        " ",
        123,
    ),
)
def test_constructor_rejects_invalid_component_name(
    component_name: Any,
) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        _projection(
            component_name=component_name,
        )


@pytest.mark.parametrize(
    "dropout",
    (
        -0.1,
        1.0,
        1.1,
        float("inf"),
        float("nan"),
    ),
)
def test_constructor_rejects_invalid_dropout_values(
    dropout: float,
) -> None:
    with pytest.raises(ValueError):
        _projection(dropout=dropout)


def test_constructor_rejects_boolean_dropout() -> None:
    with pytest.raises(TypeError, match="numeric"):
        _projection(dropout=True)  # type: ignore[arg-type]


def test_constructor_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError):
        _projection(
            activation="relu",
        )


def test_constructor_rejects_unknown_normalization() -> None:
    with pytest.raises(ValueError):
        _projection(
            normalization="batch_norm",
        )


def test_baseline_rejects_non_boolean_layer_norm() -> None:
    with pytest.raises(TypeError, match="Boolean"):
        ComponentProjection.baseline(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            component_name=COMPONENT_NAME,
            layer_norm=1,  # type: ignore[arg-type]
        )


# =============================================================================
# Forward contract
# =============================================================================


def test_forward_shape_and_finiteness() -> None:
    projection = _projection()
    projection.eval()

    output = projection(_values(rows=4))

    assert output.shape == (
        4,
        OUTPUT_DIM,
    )
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert bool(
        torch.isfinite(output).all().item()
    )


def test_forward_preserves_empty_item_axis() -> None:
    projection = _projection()
    projection.eval()

    output = projection(
        torch.empty(
            0,
            INPUT_DIM,
            dtype=torch.float32,
        )
    )

    assert output.shape == (
        0,
        OUTPUT_DIM,
    )


def test_forward_casts_input_to_module_dtype() -> None:
    projection = _projection()
    projection.eval()

    values = _values(
        rows=3,
        dtype=torch.float64,
    )
    output = projection(values)

    assert output.dtype == projection.dtype
    assert output.dtype == torch.float32


def test_module_cast_to_float64_changes_output_dtype() -> None:
    projection = _projection().double()
    projection.eval()

    values = _values(
        rows=3,
        dtype=torch.float64,
    )
    output = projection(values)

    assert projection.dtype == torch.float64
    assert output.dtype == torch.float64


def test_forward_is_deterministic_in_eval_mode() -> None:
    projection = _projection(
        dropout=0.75,
    )
    projection.eval()
    values = _values(rows=5)

    first = projection(values)
    second = projection(values)

    assert torch.equal(first, second)


def test_layer_norm_outputs_are_centered_per_item() -> None:
    projection = _projection(
        normalization="layer_norm",
        dropout=0.0,
    )
    projection.eval()

    output = projection(_values(rows=4))

    assert torch.allclose(
        output.mean(dim=-1),
        torch.zeros(4),
        atol=1e-5,
        rtol=1e-5,
    )


def test_no_normalization_does_not_force_zero_mean() -> None:
    projection = _projection(
        normalization="none",
        dropout=0.0,
    )
    projection.eval()

    with torch.no_grad():
        projection.linear.weight.fill_(0.25)
        projection.linear.bias.fill_(0.5)

    output = projection(
        torch.ones(
            3,
            INPUT_DIM,
        )
    )

    assert not torch.allclose(
        output.mean(dim=-1),
        torch.zeros(3),
        atol=1e-6,
        rtol=1e-6,
    )


def test_forward_rejects_non_tensor() -> None:
    projection = _projection()

    with pytest.raises(TypeError, match="tensor"):
        projection(  # type: ignore[arg-type]
            [[0.0] * INPUT_DIM]
        )


@pytest.mark.parametrize(
    "values",
    (
        torch.zeros(INPUT_DIM),
        torch.zeros(2, INPUT_DIM, 1),
    ),
)
def test_forward_rejects_invalid_rank(
    values: torch.Tensor,
) -> None:
    projection = _projection()

    with pytest.raises(ValueError, match="shape"):
        projection(values)


def test_forward_rejects_width_mismatch() -> None:
    projection = _projection()

    with pytest.raises(ValueError, match="feature width"):
        projection(
            torch.zeros(
                2,
                INPUT_DIM + 1,
            )
        )


def test_forward_rejects_nonfloating_values() -> None:
    projection = _projection()

    with pytest.raises(ValueError, match="floating-point"):
        projection(
            torch.zeros(
                2,
                INPUT_DIM,
                dtype=torch.long,
            )
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_forward_rejects_nonfinite_values(
    bad_value: float,
) -> None:
    projection = _projection()
    values = _values(rows=2)
    values[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        projection(values)


def test_forward_rejects_nonfinite_projected_output() -> None:
    projection = _projection(
        normalization="none",
    )
    values = _values(rows=2)

    with torch.no_grad():
        projection.linear.weight[
            0,
            0,
        ] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        projection(values)


# =============================================================================
# Backward contract
# =============================================================================


def test_backward_pass_produces_finite_input_gradients() -> None:
    projection = _projection()
    values = _values(
        rows=4,
        requires_grad=True,
    )

    output = projection(values)
    loss = output.square().mean()
    loss.backward()

    assert values.grad is not None
    assert values.grad.shape == values.shape
    assert bool(
        torch.isfinite(
            values.grad
        ).all().item()
    )


def test_backward_pass_produces_finite_parameter_gradients() -> None:
    projection = _projection()
    values = _values(
        rows=4,
        requires_grad=True,
    )

    projection(values).sum().backward()

    for parameter in projection.parameters():
        assert parameter.grad is not None
        assert bool(
            torch.isfinite(
                parameter.grad
            ).all().item()
        )


def test_eval_forward_retains_gradient_flow() -> None:
    projection = _projection(
        dropout=0.5,
    )
    projection.eval()
    values = _values(
        rows=3,
        requires_grad=True,
    )

    projection(values).sum().backward()

    assert values.grad is not None
    assert bool(
        torch.isfinite(
            values.grad
        ).all().item()
    )


# =============================================================================
# Architecture and parameter identity
# =============================================================================


def test_architecture_dict_is_complete() -> None:
    projection = _projection(
        dropout=0.2,
    )

    assert projection.architecture_dict() == {
        "schema_version": (
            COMPONENT_PROJECTION_SCHEMA_VERSION
        ),
        "component_name": COMPONENT_NAME,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "activation": "gelu",
        "normalization": "layer_norm",
        "dropout": 0.2,
        "operation_order": [
            "linear",
            "activation",
            "normalization",
            "dropout",
        ],
    }


def test_architecture_fingerprint_is_stable() -> None:
    first = _projection()
    second = _projection()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


@pytest.mark.parametrize(
    "projection",
    (
        lambda: _projection(
            input_dim=INPUT_DIM + 1,
        ),
        lambda: _projection(
            output_dim=OUTPUT_DIM + 1,
        ),
        lambda: _projection(
            component_name="memory_state",
        ),
        lambda: _projection(
            normalization="none",
        ),
        lambda: _projection(
            dropout=0.25,
        ),
    ),
)
def test_architecture_fingerprint_changes_with_contract(
    projection: Any,
) -> None:
    baseline = _projection()
    changed = projection()

    assert baseline.architecture_fingerprint() != (
        changed.architecture_fingerprint()
    )


def test_parameter_fingerprint_is_reproducible_under_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = _projection()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        second = _projection()

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_parameter_fingerprint_changes_after_mutation() -> None:
    projection = _projection()
    before = projection.parameter_fingerprint()
    architecture = (
        projection.architecture_fingerprint()
    )

    with torch.no_grad():
        projection.linear.weight[
            0,
            0,
        ] += 1.0

    assert projection.parameter_fingerprint() != before
    assert projection.architecture_fingerprint() == (
        architecture
    )


def test_component_name_changes_architecture_not_state_shape() -> None:
    first = _projection(
        component_name="static_state",
    )
    second = _projection(
        component_name="memory_state",
    )

    assert tuple(first.state_dict()) == (
        tuple(second.state_dict())
    )
    assert first.architecture_fingerprint() != (
        second.architecture_fingerprint()
    )


def test_state_dict_keys_are_stable_with_layer_norm() -> None:
    projection = _projection(
        normalization="layer_norm",
    )

    assert tuple(
        projection.state_dict()
    ) == (
        "linear.weight",
        "linear.bias",
        "normalization_layer.weight",
        "normalization_layer.bias",
    )


def test_state_dict_keys_exclude_identity_parameters() -> None:
    projection = _projection(
        normalization="none",
    )

    assert tuple(
        projection.state_dict()
    ) == (
        "linear.weight",
        "linear.bias",
    )


def test_state_dict_round_trip_preserves_outputs() -> None:
    source = _projection(
        dropout=0.0,
    )
    target = _projection(
        dropout=0.0,
    )
    target.load_state_dict(
        source.state_dict(),
        strict=True,
    )

    source.eval()
    target.eval()
    values = _values(rows=4)

    assert torch.equal(
        source(values),
        target(values),
    )
    assert source.parameter_fingerprint() == (
        target.parameter_fingerprint()
    )


def test_finite_parameter_check_passes_for_valid_module() -> None:
    projection = _projection()
    projection.assert_finite_parameters()


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_finite_parameter_check_detects_corruption(
    bad_value: float,
) -> None:
    projection = _projection()

    with torch.no_grad():
        projection.linear.weight[
            0,
            0,
        ] = bad_value

    with pytest.raises(
        ValueError,
        match="NaN|infinity",
    ):
        projection.assert_finite_parameters()


def test_extra_repr_contains_contract_identity() -> None:
    projection = _projection(
        dropout=0.2,
    )
    representation = projection.extra_repr()

    assert "static_state" in representation
    assert f"input_dim={INPUT_DIM}" in representation
    assert f"output_dim={OUTPUT_DIM}" in representation
    assert "gelu" in representation
    assert "layer_norm" in representation
    assert "dropout=0.2" in representation


# =============================================================================
# Optional device contract
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_forward_rejects_input_module_device_mismatch() -> None:
    projection = _projection().cuda()
    values = _values(rows=2)

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        projection(values)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_forward_and_backward_are_finite() -> None:
    projection = _projection().cuda()
    values = _values(
        rows=3,
        requires_grad=True,
    ).cuda()
    values.retain_grad()

    output = projection(values)
    output.square().mean().backward()

    assert output.device.type == "cuda"
    assert bool(
        torch.isfinite(output).all().item()
    )
    assert values.grad is not None
    assert bool(
        torch.isfinite(
            values.grad
        ).all().item()
    )
