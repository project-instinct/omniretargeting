"""Registry for source motion data adapters."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Mapping

from omniretargeting.data_sources.base import DataSource

DataSourceFactory = Callable[[Path, Mapping[str, Any], Mapping[str, Any] | None], DataSource]

_DATA_SOURCE_FACTORIES: dict[str, DataSourceFactory] = {}


def register_data_source(source_type: str, factory: DataSourceFactory) -> None:
    normalized_type = _normalize_source_type(source_type)
    if not normalized_type:
        raise ValueError("source_type must be a non-empty string.")
    _DATA_SOURCE_FACTORIES[normalized_type] = factory


def get_data_source_factory(source_type: str) -> DataSourceFactory:
    normalized_type = _normalize_source_type(source_type)
    if normalized_type not in _DATA_SOURCE_FACTORIES:
        _import_adapter_module(normalized_type)
    try:
        return _DATA_SOURCE_FACTORIES[normalized_type]
    except KeyError as exc:
        available = ", ".join(sorted(_DATA_SOURCE_FACTORIES)) or "none"
        raise ValueError(f"Unsupported source type {source_type!r}. Registered sources: {available}.") from exc


def create_data_source(
    source_type: str,
    motion_file: str | Path,
    source_config: Mapping[str, Any] | None = None,
    runtime_options: Mapping[str, Any] | None = None,
) -> DataSource:
    factory = get_data_source_factory(source_type)
    return factory(Path(motion_file), dict(source_config or {}), dict(runtime_options or {}))


def registered_source_types() -> list[str]:
    return sorted(_DATA_SOURCE_FACTORIES)


def _normalize_source_type(source_type: str) -> str:
    return str(source_type).strip().lower().replace("-", "_")


def _import_adapter_module(source_type: str) -> None:
    module_name = f"omniretargeting.data_sources.{source_type}"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return
        raise
