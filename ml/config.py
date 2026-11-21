"""Configuration loading.

The YAML file in ``configs/default.yaml`` is the single source of truth for pipeline parameters.
Paths are resolved relative to the repository root so that no absolute local paths are hardcoded.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("COGNITRACK_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """Dictionary with attribute access and path helpers.

    Nested dictionaries are wrapped recursively, so ``cfg.audio.target_sample_rate`` works.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(f"Config has no key '{item}'") from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def path(self, key: str) -> Path:
        """Resolve ``paths.<key>`` against the project root and create the directory."""
        resolved = resolve_path(self["paths"][key])
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load the default config, optionally merged with a user config file and in-memory overrides."""
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        with open(resolve_path(path), encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide default configuration (honours $COGNITRACK_CONFIG)."""
    return load_config(os.environ.get("COGNITRACK_CONFIG"))
