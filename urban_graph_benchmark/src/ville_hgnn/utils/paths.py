"""
Path utilities for the urban graph benchmark package.

This module is intentionally lightweight and benchmark-agnostic. It handles:

- repository-root detection
- path resolution relative to the repository root
- output-directory creation
- existence checks
- recursive candidate search from config-style dictionaries

Benchmark-specific logic belongs in ``ville_hgnn.data.inventory`` and later
dataset-building modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


UNRESOLVED_SENTINELS = {
    "",
    "DECISION_NEEDED",
    "TODO",
    "TBD",
    "UNKNOWN",
    "NA",
    "N/A",
    "NULL",
    "NONE",
}

DEFAULT_EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}

DEFAULT_REPO_MARKERS = (
    "pyproject.toml",
    ".git",
    "data",
    "recipes",
    "urban_graph_benchmark",
)


@dataclass(frozen=True)
class PathStatus:
    """Small serializable summary of a path candidate."""

    path: str
    resolved_path: str
    exists: bool
    is_file: bool
    is_dir: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "resolved_path": self.resolved_path,
            "exists": self.exists,
            "is_file": self.is_file,
            "is_dir": self.is_dir,
        }


def is_unresolved_value(value: Any) -> bool:
    """Return True when a config value is intentionally unresolved."""

    if value is None:
        return True

    if isinstance(value, str):
        normalized = value.strip()
        return normalized.upper() in UNRESOLVED_SENTINELS

    return False


def as_path(value: str | Path) -> Path:
    """Convert a string/path-like value to ``Path`` without resolving it."""

    return value if isinstance(value, Path) else Path(str(value))


def find_repo_root(
    start: str | Path | None = None,
    markers: Sequence[str] = DEFAULT_REPO_MARKERS,
) -> Path:
    """
    Find the repository root by walking upward from ``start``.

    A directory is considered a candidate root if it contains either:

    - ``pyproject.toml`` or ``.git``; or
    - at least two of the supplied directory/file markers.

    This makes the function usable from scripts, package modules, notebooks, and
    tests without hardcoding a fixed number of parent directories.
    """

    current = Path(start).resolve() if start is not None else Path.cwd().resolve()

    if current.is_file():
        current = current.parent

    candidates = [current, *current.parents]

    for directory in candidates:
        has_pyproject = (directory / "pyproject.toml").exists()
        has_git = (directory / ".git").exists()

        if has_pyproject or has_git:
            return directory

        marker_hits = sum((directory / marker).exists() for marker in markers)
        if marker_hits >= 2:
            return directory

    raise FileNotFoundError(
        "Could not locate repository root. "
        f"Searched upward from {current} using markers: {list(markers)}"
    )


def resolve_path(
    path: str | Path | None,
    repo_root: str | Path | None = None,
    base: str | Path | None = None,
    allow_unresolved: bool = True,
) -> Path | None:
    """
    Resolve a path-like config value.

    Relative paths are interpreted relative to ``base`` if provided, otherwise
    relative to ``repo_root``. If neither is provided, ``find_repo_root()`` is
    used.

    ``None`` and sentinel values such as ``DECISION_NEEDED`` return ``None``
    when ``allow_unresolved=True``.
    """

    if is_unresolved_value(path):
        if allow_unresolved:
            return None
        raise ValueError(f"Unresolved path value is not allowed: {path!r}")

    raw_path = as_path(path)  # type: ignore[arg-type]

    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)

    if base is not None:
        anchor = Path(base).resolve(strict=False)
    elif repo_root is not None:
        anchor = Path(repo_root).resolve(strict=False)
    else:
        anchor = find_repo_root()

    return (anchor / raw_path).resolve(strict=False)


def resolve_many(
    paths: Iterable[str | Path | None],
    repo_root: str | Path | None = None,
    base: str | Path | None = None,
    allow_unresolved: bool = True,
) -> list[Path]:
    """Resolve many config path values, dropping unresolved values by default."""

    resolved: list[Path] = []

    for path in paths:
        resolved_path = resolve_path(
            path,
            repo_root=repo_root,
            base=base,
            allow_unresolved=allow_unresolved,
        )
        if resolved_path is not None:
            resolved.append(resolved_path)

    return resolved


def ensure_dir(path: str | Path | None, repo_root: str | Path | None = None) -> Path | None:
    """Create a directory if the path is resolved."""

    resolved = resolve_path(path, repo_root=repo_root, allow_unresolved=True)

    if resolved is None:
        return None

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent_dir(path: str | Path | None, repo_root: str | Path | None = None) -> Path | None:
    """Create the parent directory for a file path if the path is resolved."""

    resolved = resolve_path(path, repo_root=repo_root, allow_unresolved=True)

    if resolved is None:
        return None

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.parent


def ensure_output_directories(
    config: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> list[Path]:
    """
    Create output directories declared in a config dictionary.

    The current benchmark config declares directories under:

    ``paths.output_directories_to_create``

    This helper only reads that generic location and does not assume anything
    about the benchmark itself.
    """

    repo = Path(repo_root).resolve(strict=False) if repo_root is not None else find_repo_root()
    path_config = config.get("paths", {}) if isinstance(config, Mapping) else {}
    directories = path_config.get("output_directories_to_create", [])

    created: list[Path] = []
    for directory in directories or []:
        resolved = ensure_dir(directory, repo_root=repo)
        if resolved is not None:
            created.append(resolved)

    return created


def path_status(
    path: str | Path | None,
    repo_root: str | Path | None = None,
    label_path: str | None = None,
) -> PathStatus:
    """Return a serializable status object for a path-like config value."""

    display_path = label_path if label_path is not None else str(path)

    resolved = resolve_path(path, repo_root=repo_root, allow_unresolved=True)

    if resolved is None:
        return PathStatus(
            path=display_path,
            resolved_path="",
            exists=False,
            is_file=False,
            is_dir=False,
        )

    return PathStatus(
        path=display_path,
        resolved_path=str(resolved),
        exists=resolved.exists(),
        is_file=resolved.is_file(),
        is_dir=resolved.is_dir(),
    )


def first_existing_path(
    candidates: Iterable[str | Path | None],
    repo_root: str | Path | None = None,
    require_file: bool = False,
    require_dir: bool = False,
) -> Path | None:
    """
    Return the first existing candidate path, or ``None`` if none exists.

    Set ``require_file`` or ``require_dir`` to restrict acceptable matches.
    """

    for candidate in candidates:
        resolved = resolve_path(candidate, repo_root=repo_root, allow_unresolved=True)

        if resolved is None or not resolved.exists():
            continue

        if require_file and not resolved.is_file():
            continue

        if require_dir and not resolved.is_dir():
            continue

        return resolved

    return None


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Deduplicate paths while preserving order."""

    seen: set[str] = set()
    unique: list[Path] = []

    for path in paths:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    return unique


def recursive_candidate_search(
    search_roots: Iterable[str | Path],
    filename_patterns: Iterable[str],
    repo_root: str | Path | None = None,
    include_dirs: bool = False,
    max_results: int | None = None,
    exclude_dir_names: Iterable[str] = DEFAULT_EXCLUDED_DIR_NAMES,
) -> list[Path]:
    """
    Search recursively under one or more roots for files matching patterns.

    Patterns use shell-style wildcards, for example:

    ``*tract*spatial*frame*.geojson``

    Matching is applied to the file name, not the full path, keeping behavior
    predictable across machines.

    Common generated folders such as ``.git``, ``.venv``, and ``__pycache__``
    are skipped by default.
    """

    repo = Path(repo_root).resolve(strict=False) if repo_root is not None else find_repo_root()
    patterns = [pattern for pattern in filename_patterns if not is_unresolved_value(pattern)]
    excluded = set(exclude_dir_names)

    if not patterns:
        return []

    matches: list[Path] = []

    for root_value in search_roots:
        root = resolve_path(root_value, repo_root=repo, allow_unresolved=True)

        if root is None or not root.exists() or not root.is_dir():
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in excluded]

            current_dir = Path(dirpath)

            if include_dirs:
                for dirname in dirnames:
                    candidate_dir = current_dir / dirname
                    if any(fnmatchcase(candidate_dir.name, pattern) for pattern in patterns):
                        matches.append(candidate_dir.resolve(strict=False))

                        if max_results is not None and len(matches) >= max_results:
                            return unique_paths(matches)

            for filename in filenames:
                candidate = current_dir / filename
                if any(fnmatchcase(candidate.name, pattern) for pattern in patterns):
                    matches.append(candidate.resolve(strict=False))

                    if max_results is not None and len(matches) >= max_results:
                        return unique_paths(matches)

    return unique_paths(matches)


def recursive_search_from_config(
    recursive_search_config: Mapping[str, Any] | None,
    repo_root: str | Path | None = None,
    max_results: int | None = None,
) -> list[Path]:
    """
    Run recursive search from a config block.

    Expected shape:

    .. code-block:: yaml

        recursive_search:
          enabled: true
          search_roots:
            - data
            - outputs
          filename_patterns:
            - "*svi*.csv"
    """

    if not recursive_search_config:
        return []

    if not bool(recursive_search_config.get("enabled", False)):
        return []

    search_roots = recursive_search_config.get("search_roots", [])
    filename_patterns = recursive_search_config.get("filename_patterns", [])
    exclude_dir_names = recursive_search_config.get(
        "exclude_dir_names",
        DEFAULT_EXCLUDED_DIR_NAMES,
    )

    return recursive_candidate_search(
        search_roots=search_roots,
        filename_patterns=filename_patterns,
        repo_root=repo_root,
        max_results=max_results,
        exclude_dir_names=exclude_dir_names,
    )


def collect_candidate_paths(
    explicit_path: str | Path | None = None,
    path_candidates: Iterable[str | Path | None] | None = None,
    recursive_search: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    existing_only: bool = False,
    require_file: bool = False,
    require_dir: bool = False,
    max_recursive_results: int | None = None,
) -> list[Path]:
    """
    Collect candidate paths from explicit path, path candidates, and recursive search.

    This function is deliberately generic. It does not decide which candidate is
    semantically correct; inventory code should inspect and score candidates.
    """

    repo = Path(repo_root).resolve(strict=False) if repo_root is not None else find_repo_root()

    collected: list[Path] = []

    explicit = resolve_path(explicit_path, repo_root=repo, allow_unresolved=True)
    if explicit is not None:
        collected.append(explicit)

    for candidate in path_candidates or []:
        resolved = resolve_path(candidate, repo_root=repo, allow_unresolved=True)
        if resolved is not None:
            collected.append(resolved)

    collected.extend(
        recursive_search_from_config(
            recursive_search,
            repo_root=repo,
            max_results=max_recursive_results,
        )
    )

    collected = unique_paths(collected)

    if existing_only:
        collected = [path for path in collected if path.exists()]

    if require_file:
        collected = [path for path in collected if path.is_file()]

    if require_dir:
        collected = [path for path in collected if path.is_dir()]

    return collected


def candidate_status_table(
    candidates: Iterable[str | Path | None],
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return a list of dictionaries describing candidate path existence."""

    return [
        path_status(candidate, repo_root=repo_root).to_dict()
        for candidate in candidates
    ]


def get_nested(config: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Safely access a nested dictionary value."""

    value: Any = config

    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]

    return value


def collect_candidates_from_config_section(
    config: Mapping[str, Any],
    section_keys: Sequence[str],
    repo_root: str | Path | None = None,
    existing_only: bool = False,
    require_file: bool = False,
    require_dir: bool = False,
    max_recursive_results: int | None = None,
) -> list[Path]:
    """
    Collect candidates from a config section with common path fields.

    Expected section fields:

    - ``path`` or ``raw_path``
    - ``path_candidates`` or ``raw_path_candidates``
    - ``recursive_search``
    """

    section = get_nested(config, section_keys, default={})

    if not isinstance(section, Mapping):
        return []

    explicit_path = section.get("path", section.get("raw_path"))
    path_candidates = section.get("path_candidates", section.get("raw_path_candidates", []))
    recursive_search = section.get("recursive_search", None)

    return collect_candidate_paths(
        explicit_path=explicit_path,
        path_candidates=path_candidates,
        recursive_search=recursive_search,
        repo_root=repo_root,
        existing_only=existing_only,
        require_file=require_file,
        require_dir=require_dir,
        max_recursive_results=max_recursive_results,
    )


__all__ = [
    "DEFAULT_EXCLUDED_DIR_NAMES",
    "DEFAULT_REPO_MARKERS",
    "UNRESOLVED_SENTINELS",
    "PathStatus",
    "as_path",
    "candidate_status_table",
    "collect_candidate_paths",
    "collect_candidates_from_config_section",
    "ensure_dir",
    "ensure_output_directories",
    "ensure_parent_dir",
    "find_repo_root",
    "first_existing_path",
    "get_nested",
    "is_unresolved_value",
    "path_status",
    "recursive_candidate_search",
    "recursive_search_from_config",
    "resolve_many",
    "resolve_path",
    "unique_paths",
]