"""Load and validate a training sweep config.yaml.

Free of TensorFlow imports so `ccc training --config` can validate file in foreground before a gpu is claimed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.core.config.config import BeeCombConfig
from src.core.config.config_utils import AUGMENTATION_CODE_SLOTS, _set_nested_attr
from src.core.utils import PROJECT_ROOT

CONFIGS_DIR = PROJECT_ROOT / "configs"

_AUGMENTATION_NAMES = {name for name, _ in AUGMENTATION_CODE_SLOTS}

def summarize_config(path: Path) -> tuple[str, bool]:
    """One-line summary of a config YAML, or an INVALID marker."""
    try:
        base, sweep = load_training_yaml(path)
    except SystemExit as e:
        return f"INVALID: {e}", False
    runs = 1
    for values in sweep.values():
        runs *= len(values)
    return f"{runs} sweep run(s), {len(base)} base override(s)", True

def discover_configs() -> list[tuple[Path, str, bool]]:
    """configs/*.yaml|yml newest first, each with (summary, valid)."""
    if not CONFIGS_DIR.is_dir():
        return []
    files = sorted((p for p in CONFIGS_DIR.iterdir()
                    if p.is_file() and p.suffix in (".yaml", ".yml")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [(p, *summarize_config(p)) for p in files]

def load_model_names() -> list[str]:
    """Model names from models.yaml (the valid values for training.model), registry order."""
    with open(PROJECT_ROOT / "models.yaml") as f:
        return list(yaml.safe_load(f).keys())

def _fail(path: Path, message: str) -> None:
    raise SystemExit(f"Config {path}: {message}")

def _change_to_type(current: Any, value: Any) -> Any:
    # YAML has no tuple/set literals; match the dataclass field's type so e.g. dataset.roi_size_tuple stays a tuple after an override.
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current, set) and isinstance(value, list):
        return set(value)
    return value

def _get_nested_part(obj: Any, dotted_path: str) -> Any:
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj

def load_training_yaml(path: Path) -> tuple[dict[str, Any], dict[str, list]]:
    """Parse + validate a training config.yaml, returning ``(base, sweep)``.

    ``base`` maps dotted BeeCombConfig paths to scalar overrides; ``sweep`` maps
    dotted paths to value lists (cartesian product = one run per combination).
    Exits with a clear message on any problem.
    """
    if not path.is_file():
        raise SystemExit(f"Config file not found: {path}")
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        _fail(path, f"invalid YAML: {e}")

    if data is None:
        data = {}
    if not isinstance(data, dict):
        _fail(path, f"top level must be a mapping with 'base'/'sweep' keys, got {type(data).__name__}")
    unknown = set(data) - {"base", "sweep"}
    if unknown:
        _fail(path, f"unknown top-level key(s) {sorted(unknown)}; allowed: base, sweep")

    base = data.get("base") or {}
    sweep = data.get("sweep") or {}
    for section_name, section in (("base", base), ("sweep", sweep)):
        if not isinstance(section, dict):
            _fail(path, f"'{section_name}' must be a mapping of dotted config paths, got {type(section).__name__}")

    for key, values in sweep.items():
        if not isinstance(values, list) or not values:
            _fail(path, f"sweep.{key} must be a non-empty list (each entry is one sweep arm)")

    probe = BeeCombConfig()
    try:
        for key, value in base.items():
            base[key] = _change_to_type(_get_nested_part(probe, key), value)
            _set_nested_attr(probe, key, base[key])
        for key, values in sweep.items():
            current = _get_nested_part(probe, key)
            sweep[key] = [_change_to_type(current, v) for v in values]
            _set_nested_attr(probe, key, sweep[key][0])
    except AttributeError as e:
        _fail(path, f"unknown config path: {e}")

    if "training.model" in sweep:
        known = set(load_model_names())
        bad = [m for m in sweep["training.model"] if m not in known]
        if bad:
            _fail(path, f"unknown model(s) {bad}; valid names are the models.yaml keys")

    if "training.augmentation" in sweep:
        for combo in sweep["training.augmentation"]:
            if not isinstance(combo, list) or not set(combo) <= _AUGMENTATION_NAMES:
                _fail(path, f"training.augmentation entries must be lists drawn from {sorted(_AUGMENTATION_NAMES)}, got {combo!r}")

    return base, sweep
