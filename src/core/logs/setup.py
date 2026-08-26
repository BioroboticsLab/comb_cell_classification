"""Root-logger setup for foreground commands: console + per-run log file."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.config.config import RUN_FILES
from src.core.logs.paths import LOG_DIRNAME, new_run_dir

def setup_run_logging(kind: str, level: int = logging.INFO) -> Path:
    """Install root handlers logging this run to ``output/<kind>/<ts>/log/output.log`` and the console, returning the created run directory."""
    run_dir = new_run_dir(kind)
    log_dir = run_dir / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_dir / RUN_FILES.log_filename)
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return run_dir
