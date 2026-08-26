"""Read a runs ``output.log`` (iterate, print, tail)."""
from __future__ import annotations

import select
import shutil
import subprocess
import sys
import termios
import tty
from pathlib import Path

from src.core.config.config import RUN_FILES
from src.core.logs.paths import log_dir_of

TAIL_LINES = 100

def log_file_of(log_dir: Path) -> Path:
    """A run's ``output.log``. Accepts a run dir (resolves to its ``log/`` subdir) or a log dir directly."""
    return log_dir_of(log_dir) / RUN_FILES.log_filename

def iter_log_lines(log_dir: Path):
    """Yields de-spammed lines of runs log."""
    log_file = log_file_of(log_dir)
    if not log_file.is_file():
        return
    with open(log_file, errors="replace", newline="\n") as fh:
        for line in fh:
            yield line.rstrip("\n").split("\r")[-1]

def print_log(log_dir: Path, out=None) -> None:
    """Stream runs full log to ``out`` (default stdout)."""
    out = out or sys.stdout
    log_file = log_file_of(log_dir)
    if not log_file.is_file():
        sys.exit(f"No log file found in {log_dir}")
    with open(log_file, errors="replace", newline="\n") as fh:
        shutil.copyfileobj(fh, out)  # stream, don't load the whole (possibly huge) file

def tail_follow(log_file: Path, lines: int = TAIL_LINES) -> None:
    """``tail -n <lines> -f``, stoppable with typing ``q`` or Ctrl+C."""
    proc = subprocess.Popen(["tail", "-n", str(lines), "-f", str(log_file)])
    try:
        _wait_for_q(proc)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait()

def _wait_for_q(proc: subprocess.Popen) -> None:
    """Block until ``proc`` exits or ``q`` is typed (single keypress, no Enter)."""
    if not sys.stdin.isatty():
        proc.wait()
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # char-at-a-time reads; keeps ISIG, so Ctrl+C still raises KeyboardInterrupt
    try:
        while proc.poll() is None:
            if select.select([sys.stdin], [], [], 0.2)[0] and sys.stdin.read(1) in "qQ":
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
