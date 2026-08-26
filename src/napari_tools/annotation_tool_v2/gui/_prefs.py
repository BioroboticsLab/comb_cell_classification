"""Settings remembered across runs on this machine (one JSON, same place as the TUI's state files)."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.utils import PROJECT_ROOT

DEFAULT_PREFS_FILE = Path(__file__).resolve().parents[1] / "default_prefs.json"  # committed defaults
PREFS_FILE = PROJECT_ROOT / ".annotation_tool_v2_prefs.json"  # local overrides, gitignored


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_prefs() -> dict:
    return _read_json(DEFAULT_PREFS_FILE) | _read_json(PREFS_FILE)


def save_prefs(**values) -> None:
    prefs = _read_json(PREFS_FILE)  # only local overrides, defaults stay in git
    prefs.update(values)
    PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREFS_FILE.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
