"""Collapse Keras progress-bar spam in written logs by keeping only the last ``\\r``-overwritten segment of each line — what a terminal would show."""

from __future__ import annotations

from pathlib import Path

from src.core.config.config import RUN_FILES
from src.core.logs.paths import log_dir_of


def despam_text(text: str) -> str:
    return "".join(line.split("\r")[-1] + "\n" for line in text.split("\n"))


def clean_progress_bars(file: Path) -> None:
    if not file.is_file():
        return
    # newline="" keeps the \r bytes; the default universal-newline mode turns them
    # into \n on read, which would hide the very spam this strips.
    with file.open(errors="replace", newline="") as fh:
        data = fh.read()
    if "\r" not in data:
        return
    tmp = file.with_suffix(file.suffix + ".cleaning")
    tmp.write_text(despam_text(data))
    tmp.replace(file)


def clean_log_dir(log_dir: Path) -> None:
    """De-spam every log part of a run. Accepts a run dir (resolves to its ``log/`` subdir) or a log dir directly."""
    log_dir = log_dir_of(log_dir)
    if not log_dir.is_dir():
        return
    for f in sorted(log_dir.glob(f"{RUN_FILES.log_filename}*")):
        clean_progress_bars(f)
