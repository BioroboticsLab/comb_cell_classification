"""Run lifecycle for long-running ccc commands — every background run is one tmux session (``ccc-<kind>-<timestamp>``) whose pane tees the run's ``output.log``, plus the cross-process GPU claim that keeps concurrent runs off the same card."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from filelock import FileLock

from src.core.config.config import RUN_FILES
from src.core.logs.paths import (
    LOG_DIRNAME,
    all_run_dirs,
    command_log_dir,
    output_root,
    run_log_file,
)
from src.core.utils import PROJECT_ROOT


def _tmux(*args: str) -> subprocess.CompletedProcess:
    """runs tmux process with provided arguments"""
    try:
        return subprocess.run(["tmux", *args], capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("tmux not found: ccc runs every background run in a tmux session (install it with `apt install tmux`).")


def session_name(run_dir: Path) -> str:
    """The run's tmux session name, derived from its output dir: ``ccc-<kind>-<timestamp>``."""
    return f"ccc-{run_dir.parent.name}-{run_dir.name}"


def live_sessions() -> set[str]:
    """Names of every running tmux session (empty when no server is running)."""
    return set(_tmux("list-sessions", "-F", "#{session_name}").stdout.split())


def session_pid(run_dir: Path) -> Optional[int]:
    """PID of the run's tmux pane — the shell running the command, root of the run's process tree — or ``None`` if the session is gone."""
    result = _tmux("list-panes", "-s", "-t", f"={session_name(run_dir)}", "-F", "#{pane_pid}")
    # First pane = the one the session was created with (a user who splits the
    # window while attached adds later ones).
    for line in result.stdout.split():
        return int(line)
    return None

def run_alive(run_dir: Path, sessions: Optional[set[str]] = None) -> bool:
    """Whether the run is still going — its tmux session exists.

    Pass ``sessions`` from :func:`live_sessions` to test many run dirs without
    re-querying tmux for each one.
    """
    return session_name(run_dir) in (live_sessions() if sessions is None else sessions)

class AllGpusBusy(RuntimeError):
    """No free GPU: every one is busy or claimed by a live run. Callers decide how to present ``claims``."""

    def __init__(self, claims: dict[int, Path]) -> None:
        super().__init__("All GPUs are busy or already claimed by a live run; not starting a new one.")
        self.claims = claims


def run_gpu_file(run_dir: Path) -> Path:
    return run_dir / RUN_FILES.gpu_filename


def write_run_gpu(run_dir: Path, gpu_id: int) -> None:
    run_gpu_file(run_dir).write_text(f"{gpu_id}\n")


def read_run_gpu(run_dir: Path) -> Optional[int]:
    try:
        return int(run_gpu_file(run_dir).read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _query_gpu_usage() -> Optional[list[tuple[int, int]]]:
    """(index, memory.used MiB) per GPU via ``nvidia-smi``, or ``None`` if there are no GPUs / ``nvidia-smi`` is missing — deliberately duplicated from training's ``find_free_gpu`` so the light CLI parent never imports TensorFlow."""
    try:
        out = subprocess.run( ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5, ).stdout
    except FileNotFoundError:
        return None
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Failed to query GPU memory usage with nvidia-smi: {e}") from e
    usage = [(int(idx), int(mem)) for idx, mem in (line.strip().split(", ") for line in out.splitlines())]
    return usage or None

def active_gpu_claims(exclude: Optional[Path] = None) -> dict[int, Path]:
    """GPU index -> the run dir currently claiming it, scanned across every command kind, counting only runs that are still alive.

    A finished run's claim file is left behind (nothing cleans up after a run
    that simply exits), so liveness — not the file — is what makes a claim hold.
    """
    claims: dict[int, Path] = {}
    sessions = live_sessions()
    for d in all_run_dirs():
        if d == exclude or not run_alive(d, sessions):
            continue
        gpu = read_run_gpu(d)
        if gpu is not None:
            claims[gpu] = d
    return claims

def gpu_claim_lock(timeout: float = 60) -> FileLock:
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(root / ".gpu.lock", timeout=timeout)

def claim_free_gpu(run_dir: Path, max_used_mib: int = 1000) -> Optional[int]:
    """
    Reserve least-used unclaimed GPU for ``run_dir`` and return its index — ``None`` when the machine has no GPU at all (callers fall back to CPU), :class:`AllGpusBusy` when every card is taken.
    The caller must hold :func:`gpu_claim_lock`: reading the claims, writing this one and launching the run have to be a single critical section."""
    usage = _query_gpu_usage()
    if usage is None:
        return None

    claimed = active_gpu_claims(exclude=run_dir)
    free = [(idx, mem) for idx, mem in usage if idx not in claimed and mem <= max_used_mib]
    if not free:
        raise AllGpusBusy(claimed)

    # Of the free GPUs, take one with the least memory in use.
    idx, _ = min(free, key=lambda t: t[1])
    write_run_gpu(run_dir, idx)
    return idx

def launch_in_tmux(cmd: list[str], run_dir: Path, cwd: Path) -> str:
    """Start ``cmd`` in a detached tmux session named after the run, returning the session name.

    The pane tees its combined output into the run's ``output.log``: attaching shows the run live, while the log file stays the record ``<cmd> log`` reads.
    """
    name = session_name(run_dir)
    # builds command as string
    shell_cmd = f"{shlex.join(cmd)} 2>&1 | tee -a {shlex.quote(str(run_log_file(run_dir)))}"
    tmux_result = _tmux("new-session", "-d", "-s", name, "-c", str(cwd), shell_cmd)
    if tmux_result.returncode != 0:
        sys.exit(f"tmux could not start the run: {tmux_result.stderr.strip()}")
    # A user's ~/.tmux.conf may set remain-on-exit globally, which would keep the session (and with it this run's "still alive" verdict) around forever
    _tmux("set-option", "-t", f"={name}", "remain-on-exit", "off")
    return name

def stop_run(run_dir: Path, job_label: str = "run", timeout: float = 30) -> None:
    """Kill the run's tmux session and every process it spawned — SIGTERM the pane's process tree, SIGKILL whatever survives ``timeout`` — then drop the run's GPU claim.

    Blocks until every process is gone, so never call it from a UI thread.
    """
    pid = session_pid(run_dir)
    if pid is None:
        print(f"No live {job_label} session found; nothing to stop.")
        return
    print(f"Stopping {job_label} (tmux session {session_name(run_dir)})...")
    # Record that this run ended by user action so status reporting shows it as "stopped" rather than inferring from the log.
    try:
        (run_dir / "stopped").touch()
    except OSError:
        pass

    # Snapshot the tree *before* signalling: once the pane's shell dies its children are reparented to init and can no longer be found from this PID.
    try:
        leader = psutil.Process(pid)
        procs = [leader, *leader.children(recursive=True)]
    except psutil.Error:
        procs = []

    _tmux("kill-session", "-t", f"={session_name(run_dir)}")
    # Closing the pane only signals its foreground process group; terminate the snapshot too, which catches MirroredStrategy/'spawn' children that left it.
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass

    # Escalate rather than wait: TensorFlow inside a CUDA call can be slow to service SIGTERM, and a run that never handles it at all used to hang this call (and its caller's thread) indefinitely.
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        print(f"PID {p.pid} still alive after {timeout:g}s; sending SIGKILL.")
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)

    run_gpu_file(run_dir).unlink(missing_ok=True)
    print(f"{job_label.capitalize()} stopped.")


@dataclass
class LaunchedRun:
    """Handle to a newly launched run."""

    run_dir: Path
    session: str    # its tmux session name
    gpu: str        # "cpu" or a GPU index like "0"

    @property
    def gpu_label(self) -> str:
        return "CPU" if self.gpu == "cpu" else f"GPU {self.gpu}"

    @property
    def log_file(self) -> Path:
        return self.run_dir / LOG_DIRNAME / RUN_FILES.log_filename


def start_run(kind: str, module: str, entry: str, child_args: list[str], extra_files: Optional[dict[str, str]] = None) -> LaunchedRun:
    """Start ``python -m <module> <entry> <child_args> --gpu-id .. --run-dir ..`` in its own tmux session on a free GPU.

    Creates ``output/<kind>/<timestamp>/``, claims a GPU (raising
    :class:`AllGpusBusy` and removing the dir if none is free), writes
    ``extra_files`` ({filename: content}) into the run dir, launches the session
    and returns a :class:`LaunchedRun`. Shared by ``src.cli.train`` and
    ``src.cli.classify``, whose start blocks are identical.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = command_log_dir(kind) / timestamp
    
    # create runfolder
    (run_dir / LOG_DIRNAME).mkdir(parents=True, exist_ok=True)

    # claiming a lock using a filelock (.gpu.lock)
    with gpu_claim_lock():
        try:
            gpu_id = claim_free_gpu(run_dir)
        except AllGpusBusy:
            shutil.rmtree(run_dir, ignore_errors=True)  # nothing ran, leave no empty run dir behind
            raise
        gpu = "cpu" if gpu_id is None else str(gpu_id)

        for name, content in (extra_files or {}).items():
            (run_dir / name).write_text(content)

        session = launch_in_tmux(
            # launches tmux session for run
            [sys.executable, "-u", "-m", module, entry, *child_args, "--gpu-id", gpu, "--run-dir", str(run_dir)],
            run_dir=run_dir,
            cwd=PROJECT_ROOT,
        )
    return LaunchedRun(run_dir=run_dir, session=session, gpu=gpu)
