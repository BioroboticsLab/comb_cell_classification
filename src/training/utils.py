from __future__ import annotations
from typing import Union

from pathlib import Path

from src.core.logs.paths import project_root as _project_root  # noqa: E402 - shared root finder (handles site-packages installs)


def get_suffixes_as_tuple(string: str, split_string: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in string.split(split_string) if s.strip())


def check_extension(string: str, extension: tuple[str, ...]) -> bool:
    return string.lower().endswith(extension)


def resolve_to_root_path(path: Union[Path, str]) -> Path:
    """Resolves a path relative to the project root."""
    path = Path(path)
    return _project_root() / path
