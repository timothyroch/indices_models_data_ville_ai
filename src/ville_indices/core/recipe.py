"""YAML recipe loading and structured recipe models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}")
    return dict(value)


@dataclass
class NormalizationConfig:
    method: str = "none"
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NormalizationConfig":
        data = _as_mapping(data)
        known = {"method", "parameters", "status", "reason"}
        parameters = dict(data.get("parameters", {}))
        extras = {k: v for k, v in data.items() if k not in known}
        parameters.update(extras)
        return cls(
            method=data.get("method", "none"),
            parameters=parameters,
            status=data.get("status"),
            reason=data.get("reason"),
        )


@dataclass
class WeightConfig:
    value: float | None = None
    source: str | None = None
    status: str | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WeightConfig":
        data = _as_mapping(data)
        known = {"value", "source", "status", "reason"}
        value = data.get("value")
        return cls(
            value=float(value) if value is not None else None,
            source=data.get("source"),
            status=data.get("status"),
            reason=data.get("reason"),
            extra={k: v for k, v in data.items() if k not in known},
        )


@dataclass
class VariableConfig:
    key: str
    canonical_name: str
    required: bool = True
    unit: str | None = None
    scale: str | None = None
    direction: str = "none"
    numeric: bool = True
    nonnegative: bool | None = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    weight: WeightConfig = field(default_factory=WeightConfig)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> "VariableConfig":
        data = _as_mapping(data)
        known = {
            "canonical_name",
            "required",
            "unit",
            "scale",
            "direction",
            "numeric",
            "nonnegative",
            "normalization",
            "weight",
            "decisions",
        }
        canonical_name = data.get("canonical_name")
        if not canonical_name:
            raise ValueError(f"Variable '{key}' is missing canonical_name")
        return cls(
            key=key,
            canonical_name=str(canonical_name),
            required=bool(data.get("required", True)),
            unit=data.get("unit"),
            scale=data.get("scale"),
            direction=data.get("direction", "none"),
            numeric=bool(data.get("numeric", True)),
            nonnegative=data.get("nonnegative"),
            normalization=NormalizationConfig.from_dict(data.get("normalization")),
            weight=WeightConfig.from_dict(data.get("weight")),
            decisions=list(data.get("decisions", [])),
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recipe:
    name: str
    version: str
    construct_measured: str
    score_direction: str
    reproduction_level: str
    spatial_id_column: str
    variables: dict[str, VariableConfig]
    source: str | None = None
    method_reference: str | dict[str, Any] | None = None
    missing_data: dict[str, Any] = field(default_factory=lambda: {"strategy": "error"})
    aggregation: dict[str, Any] = field(default_factory=lambda: {"method": "custom"})
    classification: dict[str, Any] = field(default_factory=lambda: {"method": "none"})
    outputs: dict[str, Any] = field(default_factory=dict)
    assumptions: list[dict[str, Any] | str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        data = _as_mapping(data)
        required_top_level = [
            "name",
            "version",
            "construct_measured",
            "score_direction",
            "reproduction_level",
            "spatial_id_column",
            "variables",
        ]
        missing = [key for key in required_top_level if key not in data]
        if missing:
            raise ValueError(f"Recipe is missing required fields: {', '.join(missing)}")

        variables = {
            key: VariableConfig.from_dict(key, value)
            for key, value in _as_mapping(data["variables"]).items()
        }
        known = set(required_top_level) | {
            "source",
            "method_reference",
            "missing_data",
            "aggregation",
            "classification",
            "outputs",
            "assumptions",
            "decisions",
        }
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            construct_measured=str(data["construct_measured"]),
            score_direction=str(data["score_direction"]),
            reproduction_level=str(data["reproduction_level"]),
            spatial_id_column=str(data["spatial_id_column"]),
            variables=variables,
            source=data.get("source"),
            method_reference=data.get("method_reference"),
            missing_data=dict(data.get("missing_data", {"strategy": "error"})),
            aggregation=dict(data.get("aggregation", {"method": "custom"})),
            classification=dict(data.get("classification", {"method": "none"})),
            outputs=dict(data.get("outputs", {})),
            assumptions=list(data.get("assumptions", [])),
            decisions=list(data.get("decisions", [])),
            extra={k: v for k, v in data.items() if k not in known},
        )

    @property
    def required_variables(self) -> list[str]:
        return [
            config.canonical_name
            for config in self.variables.values()
            if config.required
        ]

    @property
    def optional_variables(self) -> list[str]:
        return [
            config.canonical_name
            for config in self.variables.values()
            if not config.required
        ]

    @property
    def variable_columns(self) -> list[str]:
        return [config.canonical_name for config in self.variables.values()]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variables"] = {
            key: variable.to_dict() for key, variable in self.variables.items()
        }
        return data


def load_recipe(path: str | Path) -> Recipe:
    """Load a YAML recipe from disk."""

    recipe_path = Path(path)
    with recipe_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Recipe.from_dict(data)
