"""Shared per-run output layout: one timestamped run dir per invocation under ``<project>/output/<kind>/``, holding the run's ``log/`` subdir plus its top-level run files (``gpu.claim``, ``args.json``, ...)."""

from __future__ import annotations

from datetime import datetime
from functools import cache
from pathlib import Path

from src.core.config.config import PathsConfig, RUN_FILES

LOG_DIRNAME = RUN_FILES.log_dirname


@cache
def project_root() -> Path:
    """The repo root: first ancestor of this file — or of the cwd — containing pyproject.toml.

    The cwd walk covers installs into site-packages (uv tool / pip), where no
    pyproject.toml exists above __file__. Falls back to the cwd itself so an
    installed ccc run from anywhere still works, with output/ under the cwd.
    """
    for start in (Path(__file__).resolve(), Path.cwd()):
        for parent in (start, *start.parents):
            if (parent / "pyproject.toml").is_file():
                return parent
    return Path.cwd()

def output_root() -> Path:
    p = PathsConfig().output_path
    return p if p.is_absolute() else project_root() / p

def command_log_dir(kind: str) -> Path:
    """Base directory holding all runs of command ``kind`` (e.g. ``"training"``)."""
    return output_root() / kind

def new_run_dir(kind: str, format: str = "%Y-%m-%d_%H-%M-%S") -> Path:
    """A fresh, timestamped run directory under ``command_log_dir(kind)`` (not created here; the caller decides when to ``mkdir``)."""
    timestamp = datetime.now().strftime(format)
    return command_log_dir(kind) / timestamp

def log_dir_of(run_dir: Path) -> Path:
    """The directory holding ``run_dir``'s ``output.log``: its ``log/`` subdir when present, else ``run_dir`` itself (already a log dir — idempotent)."""
    sub = run_dir / LOG_DIRNAME
    return sub if sub.is_dir() else run_dir

def run_log_file(run_dir: Path) -> Path:
    """Path of a run's current ``output.log``."""
    return log_dir_of(run_dir) / RUN_FILES.log_filename

def run_dirs_newest_first(base_dir: Path) -> list[Path]:
    """All run dirs under ``base_dir`` (a ``command_log_dir``), newest first."""
    if not base_dir.is_dir():
        return []
    # Dir names are zero-padded timestamps, so lexical sort == chronological.
    return sorted((p for p in base_dir.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)

def command_run_dirs(kind: str) -> list[Path]:
    """All run dirs of command ``kind`` across the output tree, newest first."""
    return run_dirs_newest_first(command_log_dir(kind))

def all_run_dirs() -> list[Path]:
    """Every run dir across every command kind (for cross-command bookkeeping such as GPU-claim accounting), unsorted."""
    roots = [output_root()]
    return [r for root in roots if root.is_dir()
              for k in root.iterdir() if k.is_dir()
              for r in k.iterdir() if r.is_dir()]

def du_mb(path: Path) -> int:
    """Total size of all files under ``path``, in whole megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total // (1024 * 1024)
