"""
General I/O utilities for the urban graph benchmark package.

This module is intentionally lightweight and benchmark-agnostic. It handles:

- YAML / JSON config loading
- JSON / YAML / text artifact writing
- safe atomic writes
- optional file hashing
- simple serialization helpers for common Python objects

Benchmark-specific inventory and dataset-building logic belongs in
``ville_hgnn.data.inventory`` and downstream modules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping as ABCMapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


JSON_INDENT = 2
DEFAULT_ENCODING = "utf-8"
DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


class ConfigLoadError(RuntimeError):
    """Raised when a YAML/JSON config cannot be loaded."""


class ArtifactWriteError(RuntimeError):
    """Raised when an artifact cannot be written safely."""


def as_path(path: str | Path) -> Path:
    """Convert a string/path-like object to ``Path``."""

    return path if isinstance(path, Path) else Path(str(path))


def ensure_parent_dir(path: str | Path) -> Path:
    """Create and return the parent directory for a file path."""

    file_path = as_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path.parent


def _require_yaml() -> None:
    """Raise a useful error when PyYAML is unavailable."""

    if yaml is None:
        raise ConfigLoadError(
            "PyYAML is required for YAML read/write operations. "
            "Install it with `pip install pyyaml` or use JSON configs."
        ) from _YAML_IMPORT_ERROR


def read_text(path: str | Path, encoding: str = DEFAULT_ENCODING) -> str:
    """Read a text file."""

    return as_path(path).read_text(encoding=encoding)


def write_text(
    path: str | Path,
    text: str,
    encoding: str = DEFAULT_ENCODING,
    atomic: bool = True,
) -> Path:
    """Write a text artifact, creating parent directories if needed."""

    return safe_write_text(path=path, text=text, encoding=encoding, atomic=atomic)


def write_markdown(
    path: str | Path,
    text: str,
    encoding: str = DEFAULT_ENCODING,
    atomic: bool = True,
) -> Path:
    """Write a Markdown artifact, creating parent directories if needed."""

    return write_text(path=path, text=text, encoding=encoding, atomic=atomic)


def load_json(path: str | Path, encoding: str = DEFAULT_ENCODING) -> Any:
    """Load a JSON file."""

    with as_path(path).open("r", encoding=encoding) as handle:
        return json.load(handle)


def dump_json_string(
    obj: Any,
    indent: int | None = JSON_INDENT,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialize an object to a JSON string using the project serializer."""

    return json.dumps(
        to_jsonable(obj),
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
    )


def write_json(
    path: str | Path,
    obj: Any,
    encoding: str = DEFAULT_ENCODING,
    indent: int | None = JSON_INDENT,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    atomic: bool = True,
) -> Path:
    """Write an object as JSON."""

    text = dump_json_string(
        obj,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
    )
    if indent is not None:
        text += "\n"

    return safe_write_text(path=path, text=text, encoding=encoding, atomic=atomic)


def load_yaml(path: str | Path, encoding: str = DEFAULT_ENCODING) -> Any:
    """Load a YAML file."""

    _require_yaml()

    with as_path(path).open("r", encoding=encoding) as handle:
        return yaml.safe_load(handle)  # type: ignore[union-attr]


def dump_yaml_string(
    obj: Any,
    sort_keys: bool = False,
    allow_unicode: bool = True,
) -> str:
    """Serialize an object to a YAML string using the project serializer."""

    _require_yaml()

    return yaml.safe_dump(  # type: ignore[union-attr]
        to_jsonable(obj),
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )


def write_yaml(
    path: str | Path,
    obj: Any,
    encoding: str = DEFAULT_ENCODING,
    sort_keys: bool = False,
    allow_unicode: bool = True,
    atomic: bool = True,
) -> Path:
    """Write an object as YAML."""

    text = dump_yaml_string(
        obj,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )

    return safe_write_text(path=path, text=text, encoding=encoding, atomic=atomic)


def load_config(path: str | Path, encoding: str = DEFAULT_ENCODING) -> dict[str, Any]:
    """
    Load a YAML or JSON configuration file.

    Supported extensions:

    - ``.yaml``
    - ``.yml``
    - ``.json``

    The returned object must be a mapping.
    """

    config_path = as_path(path)
    suffix = config_path.suffix.lower()

    try:
        if suffix in {".yaml", ".yml"}:
            config = load_yaml(config_path, encoding=encoding)
        elif suffix == ".json":
            config = load_json(config_path, encoding=encoding)
        else:
            raise ConfigLoadError(
                f"Unsupported config extension for {config_path}. "
                "Expected .yaml, .yml, or .json."
            )
    except ConfigLoadError:
        raise
    except Exception as exc:
        raise ConfigLoadError(f"Failed to load config {config_path}: {exc}") from exc

    if config is None:
        return {}

    if not isinstance(config, ABCMapping):
        raise ConfigLoadError(
            f"Config {config_path} must contain a mapping at the top level; "
            f"got {type(config).__name__}."
        )

    return dict(config)


def write_config(
    path: str | Path,
    config: Mapping[str, Any],
    encoding: str = DEFAULT_ENCODING,
    atomic: bool = True,
) -> Path:
    """Write a config as YAML or JSON based on file extension."""

    config_path = as_path(path)
    suffix = config_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        return write_yaml(config_path, config, encoding=encoding, atomic=atomic)

    if suffix == ".json":
        return write_json(config_path, config, encoding=encoding, atomic=atomic)

    raise ArtifactWriteError(
        f"Unsupported config extension for {config_path}. "
        "Expected .yaml, .yml, or .json."
    )


def safe_write_text(
    path: str | Path,
    text: str,
    encoding: str = DEFAULT_ENCODING,
    atomic: bool = True,
) -> Path:
    """
    Safely write text to a file.

    When ``atomic=True``, content is written to a temporary file in the same
    directory and then moved into place with ``os.replace``. This avoids partial
    artifacts if a process fails mid-write.
    """

    output_path = as_path(path)
    ensure_parent_dir(output_path)

    if not atomic:
        output_path.write_text(text, encoding=encoding)
        return output_path

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)

        os.replace(temp_path, output_path)
        return output_path

    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise ArtifactWriteError(f"Failed to write text artifact {output_path}: {exc}") from exc


def safe_write_bytes(
    path: str | Path,
    data: bytes,
    atomic: bool = True,
) -> Path:
    """Safely write bytes to a file."""

    output_path = as_path(path)
    ensure_parent_dir(output_path)

    if not atomic:
        output_path.write_bytes(data)
        return output_path

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)

        os.replace(temp_path, output_path)
        return output_path

    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise ArtifactWriteError(f"Failed to write bytes artifact {output_path}: {exc}") from exc


def write_jsonl(
    path: str | Path,
    records: Iterable[Any],
    encoding: str = DEFAULT_ENCODING,
    atomic: bool = True,
) -> Path:
    """
    Write an iterable of records as JSON Lines.

    Records are streamed one by one instead of materializing the whole JSONL
    file in memory.
    """

    output_path = as_path(path)
    ensure_parent_dir(output_path)

    def _write_records(file_handle: Any) -> None:
        for record in records:
            file_handle.write(json.dumps(to_jsonable(record), ensure_ascii=False))
            file_handle.write("\n")

    if not atomic:
        with output_path.open("w", encoding=encoding) as handle:
            _write_records(handle)
        return output_path

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            _write_records(handle)

        os.replace(temp_path, output_path)
        return output_path

    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise ArtifactWriteError(f"Failed to write JSONL artifact {output_path}: {exc}") from exc


def read_jsonl(path: str | Path, encoding: str = DEFAULT_ENCODING) -> list[Any]:
    """Read a JSON Lines file."""

    records: list[Any] = []

    with as_path(path).open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number} in {path}: {exc}"
                ) from exc

    return records


def to_jsonable(obj: Any) -> Any:
    """
    Convert common Python/scientific objects into JSON/YAML-safe values.

    This helper avoids importing heavy optional dependencies. It recognizes
    pandas/numpy-like objects by behavior where possible.
    """

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, Decimal):
        if obj.is_nan() or obj.is_infinite():
            return None
        return float(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))

    if isinstance(obj, ABCMapping):
        return {
            str(to_jsonable(key)): to_jsonable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]

    if isinstance(obj, set):
        converted = [to_jsonable(value) for value in obj]
        return sorted(converted, key=lambda value: repr(value))

    # numpy scalar-like values
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:
            pass

    # pandas Timestamp-like values
    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    # pandas Series / Index / numpy arrays
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except Exception:
            pass

    # Last resort: stable string representation.
    return str(obj)


def file_hash(
    path: str | Path,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Compute a file hash using a streaming reader."""

    file_path = as_path(path)

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def maybe_file_hash(
    path: str | Path | None,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str | None:
    """Compute a file hash when the file exists; otherwise return ``None``."""

    if path is None:
        return None

    file_path = as_path(path)

    if not file_path.exists() or not file_path.is_file():
        return None

    return file_hash(file_path, algorithm=algorithm, chunk_size=chunk_size)


def hash_files(
    paths: Iterable[str | Path],
    algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> dict[str, str | None]:
    """Compute hashes for multiple files, returning ``None`` for missing paths."""

    return {
        str(as_path(path)): maybe_file_hash(
            path,
            algorithm=algorithm,
            chunk_size=chunk_size,
        )
        for path in paths
    }


def config_hash(
    config: Mapping[str, Any],
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> str:
    """
    Compute a stable hash of a loaded config mapping.

    This is useful for provenance when the config object has already been
    modified in memory.
    """

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    payload = dump_json_string(
        config,
        indent=None,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    hasher.update(payload.encode(DEFAULT_ENCODING))
    return hasher.hexdigest()


def artifact_record(
    path: str | Path,
    kind: str,
    description: str | None = None,
    include_hash: bool = False,
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> dict[str, Any]:
    """Create a small serializable metadata record for an artifact."""

    artifact_path = as_path(path)
    exists = artifact_path.exists()

    return {
        "path": str(artifact_path),
        "kind": kind,
        "description": description,
        "exists": exists,
        "is_file": artifact_path.is_file() if exists else False,
        "is_dir": artifact_path.is_dir() if exists else False,
        "size_bytes": artifact_path.stat().st_size if exists and artifact_path.is_file() else None,
        "hash_algorithm": hash_algorithm if include_hash else None,
        "hash": maybe_file_hash(artifact_path, algorithm=hash_algorithm) if include_hash else None,
    }


def write_artifact_manifest(
    path: str | Path,
    artifacts: Sequence[Mapping[str, Any]],
    include_manifest_hash: bool = False,
) -> Path:
    """Write a JSON manifest describing generated artifacts."""

    manifest: dict[str, Any] = {
        "artifacts": list(artifacts),
    }

    if include_manifest_hash:
        manifest["manifest_hash"] = config_hash(manifest)

    return write_json(path, manifest, sort_keys=False)


def read_optional_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON if the file exists, otherwise return ``default``."""

    json_path = as_path(path)
    if not json_path.exists():
        return default
    return load_json(json_path)


def read_optional_yaml(path: str | Path, default: Any = None) -> Any:
    """Read YAML if the file exists, otherwise return ``default``."""

    yaml_path = as_path(path)
    if not yaml_path.exists():
        return default
    return load_yaml(yaml_path)


__all__ = [
    "ArtifactWriteError",
    "ConfigLoadError",
    "DEFAULT_ENCODING",
    "DEFAULT_HASH_ALGORITHM",
    "DEFAULT_HASH_CHUNK_SIZE",
    "JSON_INDENT",
    "artifact_record",
    "as_path",
    "config_hash",
    "dump_json_string",
    "dump_yaml_string",
    "ensure_parent_dir",
    "file_hash",
    "hash_files",
    "load_config",
    "load_json",
    "load_yaml",
    "maybe_file_hash",
    "read_jsonl",
    "read_optional_json",
    "read_optional_yaml",
    "read_text",
    "safe_write_bytes",
    "safe_write_text",
    "to_jsonable",
    "write_artifact_manifest",
    "write_config",
    "write_json",
    "write_jsonl",
    "write_markdown",
    "write_text",
    "write_yaml",
]