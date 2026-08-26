"""Shared CLI layer for managing background runs (``ccc training``, ``ccc classify``): kill/restart target resolution, the interactive run picker, and the ``log`` subcommand (tail/list/show/delete/cleanup)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, NoReturn, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rich.text import Text

from src.core.logs.despam import clean_log_dir
from src.core.logs.paths import (
    command_log_dir,
    command_run_dirs,
    du_mb,
    run_dirs_newest_first,
    run_log_file,
)
from src.core.logs.reading import iter_log_lines, print_log, tail_follow
from src.core.runs import (
    AllGpusBusy,
    live_sessions,
    read_run_gpu,
    run_alive,
    session_name,
    stop_run,
)

UNSET = object()  # needed for argparse to distinguish between set and unset flags 

DELETE_STATUSES = ["success", "partial", "crashed", "stopped", "unknown", "all"]

STATUS_STYLE = {
    "success": "green",
    "partial": "dark_orange",
    "crashed": "red",
    "stopped": "yellow",
    "running": "cyan",
    "unknown": "dim",
}

_TRACEBACK = "Traceback (most recent call last)"
@dataclass
class RunKind:
    """Everything command-specific the shared run-management CLI needs."""

    kind: str                                       # run dirs live under output/<kind>/
    job_label: str                                  # stop_run label + prompts/table titles
    detail_header: str                              # per-run detail column, e.g. "Variants" / "Images"
    summarize: Callable[[Path], "tuple[str, Text]"] # run dir -> (status, detail)


def exit_all_gpus_busy(e: AllGpusBusy) -> NoReturn:
    """Report who holds which GPU and exit — how every start path reacts to :class:`AllGpusBusy`."""
    if e.claims:
        print("Currently claimed GPUs:")
        for idx, d in sorted(e.claims.items()):
            print(f"  GPU {idx}: {d}")
    sys.exit(str(e))


def active_runs(rk: RunKind) -> list[Path]:
    """Currently-alive run dirs of this kind, newest first."""
    sessions = live_sessions()  # one tmux query for the whole scan
    return [d for d in command_run_dirs(rk.kind) if run_alive(d, sessions)]


def fmt_started(log_dir: Path) -> str:
    try:
        return datetime.strptime(log_dir.name, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return log_dir.name


def basic_status(log_dir: Path) -> str:
    """Generic status for runs without command-specific log parsing: stopped marker, else crashed on a traceback, else success once the run logged anything."""
    if (log_dir / "stopped").exists():
        return "stopped"
    any_lines = False
    for line in iter_log_lines(log_dir):
        any_lines = True
        if _TRACEBACK in line:
            return "crashed"
    return "success" if any_lines else "unknown"


def resolve_log_id(rk: RunKind, idval: str) -> Path:
    # Same merged ordering as --list-all, so index IDs stay consistent.
    dirs = command_run_dirs(rk.kind)
    for d in dirs:  # exact timestamp dir-name match
        if d.name == idval:
            return d
    if idval.isdigit() and 1 <= int(idval) <= len(dirs):  # 1-based index from --list-all
        return dirs[int(idval) - 1]
    sys.exit(f"No log with ID {idval!r}. Run `{rk.job_label} log --list-all` to see valid IDs.")


def resolve_active_target(rk: RunKind, target: Optional[str], active: list[Path], flag: str) -> list[Path]:
    """Resolve which active run(s) ``--kill``/``--restart`` should act on, showing the interactive picker table when several runs are active and no ID was given."""
    # target: None = flag given no value, "all", or an ID (index/timestamp).
    if not active:
        print(f"No {rk.job_label} running.")
        return []
    if target is None:
        if len(active) > 1:
            return [select_active_run(rk, active, action=flag)]
        return [active[0]]
    if target == "all":
        return list(active)
    run = resolve_log_id(rk, target)
    if not run_alive(run):
        sys.exit(f"Run {target!r} is not currently active.")
    return [run]


def select_active_run(rk: RunKind, active: list[Path], action: str = "tail") -> Path:
    """Show a rich table of the active runs and return the one the user picks (*action* names the follow-up in the prompt)."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Active {rk.job_label} runs", header_style="bold", title_justify="left")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Started")
    table.add_column("tmux session")
    table.add_column("GPU")
    table.add_column(rk.detail_header)
    for i, d in enumerate(active, 1):
        gpu = read_run_gpu(d)
        gpu_label = f"GPU {gpu}" if gpu is not None else "CPU"
        table.add_row(str(i), d.name, session_name(d), gpu_label, rk.summarize(d)[1])
    console.print(table)

    while True:
        try:
            raw = input(f"Select a run to {action} (1..{len(active)}, q to abort): ").strip()
        except EOFError:
            sys.exit("\nNo selection (no input available).")
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit("Aborted.")
        if raw.isdigit() and 1 <= int(raw) <= len(active):
            return active[int(raw) - 1]
        console.print(f"[red]Invalid ID {raw!r}[/red]; enter a number 1..{len(active)} or q.")

def do_tail(rk: RunKind, idval: Optional[str]) -> None:
    active = active_runs(rk)
    if idval is None:
        if not active:
            print(f"No {rk.job_label} running.")
            return
        run = select_active_run(rk, active) if len(active) > 1 else active[0]
    else:
        run = resolve_log_id(rk, idval)
        if not run_alive(run):
            sys.exit(f"Run {idval!r} is not currently active (use `{rk.job_label} log --show {idval}` to view its saved log).")
    log_file = run_log_file(run)
    if not log_file.is_file():
        sys.exit(f"Log file not found: {log_file}")
    print(f"Tailing {log_file} (Ctrl+C to stop, `tmux attach -t {session_name(run)}` for the live session)")
    tail_follow(log_file)

def do_complete_log(rk: RunKind) -> None:
    dirs = command_run_dirs(rk.kind)
    if not dirs:
        sys.exit(f"No {rk.job_label} runs found.")
    print_log(dirs[0])  # newest run: the most recently *started* one, active or not

def do_list_logs(rk: RunKind) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    base_dir = command_log_dir(rk.kind)
    dirs = command_run_dirs(rk.kind)
    if not dirs:
        print(f"No runs found under {base_dir}")
        return
    table = Table(title=f"{rk.job_label.capitalize()} runs — {base_dir}", header_style="bold", title_justify="left")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Started")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column(rk.detail_header)
    table.add_column("Size", justify="right")
    for i, d in enumerate(dirs, 1):
        status, detail = rk.summarize(d)
        gpu = read_run_gpu(d)
        started = Text(fmt_started(d))
        table.add_row(
            str(i), started,
            Text(status, style=STATUS_STYLE.get(status, "dim")),
            str(gpu) if gpu is not None else "—",
            detail, f"{du_mb(d)}M",
        )
    Console().print(table)

def do_show_log(rk: RunKind, idval: str) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    log_dir = resolve_log_id(rk, idval)
    status, detail = rk.summarize(log_dir)
    header = Text()
    header.append("Started: ", style="bold"); header.append(fmt_started(log_dir) + "\n")
    header.append("Status:  ", style="bold")
    header.append(status + "\n", style=STATUS_STYLE.get(status, "dim"))
    header.append(f"{rk.detail_header + ':':<9}", style="bold")
    header.append_text(detail)

    # Page the (potentially huge) log through `less` so it's scrollable/searchable (press `/` to search).
    # Fall back to stdout when no tty or less available
    pager = None
    if sys.stdout.isatty():
        try:
            pager = subprocess.Popen(["less", "-R"], stdin=subprocess.PIPE, text=True)
        except OSError:
            pager = None
    out = pager.stdin if pager else sys.stdout
    try:
        Console(file=out, force_terminal=bool(pager)).print(Panel(header, title=log_dir.name, expand=False))
        print_log(log_dir, out=out)
    except BrokenPipeError:
        pass  # user quit before all output was written
    finally:
        if pager:
            try:
                pager.stdin.close()
            except BrokenPipeError:
                pass
            pager.wait()

def do_delete_logs(rk: RunKind, statuses: list[str], assume_yes: bool = False) -> None:
    """Delete every run dir whose status is in ``statuses`` (or all), never touching a currently-active run. A run dir holds the run's logs plus everything it produced (wandb files, model checkpoints, ...) — deleting removes them all."""
    base_dir = command_log_dir(rk.kind)
    wanted = set(statuses)
    targets: list[tuple[Path, str]] = []
    for d in run_dirs_newest_first(base_dir):
        status, _ = rk.summarize(d)
        if status == "running":
            continue
        if "all" in wanted or status in wanted:
            targets.append((d, status))
    if not targets:
        print(f"No runs to delete (matching: {', '.join(sorted(wanted))}).")
        return
    print(f"About to delete {len(targets)} run(s), INCLUDING everything they produced (wandb data, model checkpoints, ...):")
    for d, status in targets:
        print(f"  {fmt_started(d):<21} {status:<8} {du_mb(d):>7}M  {d}")
    if not assume_yes and input("Delete these permanently? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted.")
        return
    deleted = 0
    for d, _ in targets:
        try:
            shutil.rmtree(d)
            deleted += 1
        except OSError as e:
            print(f"  failed to delete {d}: {e}")
    print(f"Deleted {deleted} run(s).")


def do_cleanup_logs(rk: RunKind) -> None:
    base_dir = command_log_dir(rk.kind)
    if not base_dir.is_dir():
        print(f"No runs directory at {base_dir}")
        return
    print(f"Cleaning Keras progress-bar updates under {base_dir}...")
    total_before = du_mb(base_dir)
    for d in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        before = du_mb(d)
        clean_log_dir(d)
        print(f"  {str(d):<60} {before}M -> {du_mb(d)}M")
    print(f"Total: {total_before}M -> {du_mb(base_dir)}M")


def do_kill(rk: RunKind, target: Optional[str]) -> None:
    for run in resolve_active_target(rk, target, active_runs(rk), "kill"):
        stop_run(run, job_label=rk.job_label)

def add_log_subparser(parser: argparse.ArgumentParser, rk: RunKind, statuses: list[str] = DELETE_STATUSES) -> None:
    """Attach the shared ``log`` subcommand (tail/list/show/delete/cleanup) to *parser*."""
    sub = parser.add_subparsers(dest="command")
    p_log = sub.add_parser(
        "log",
        help=f"View, list, or delete {rk.job_label} run logs.",
        description=f"View, list, or delete {rk.job_label} run logs. With no option, tails the active run's log.",
    )
    g = p_log.add_mutually_exclusive_group()
    g.add_argument( "--currently-running", nargs="?", const=None, default=UNSET, metavar="ID", help="Tail a currently-running run's log (this is the default); if several are active a table is shown to pick one; ID tails that specific active run. ", )
    g.add_argument("--list-all", action="store_true", help=f"List all {rk.job_label} runs (start time, status, GPU) in a table.")
    g.add_argument("--show", metavar="ID", default=None, help="Print a run's full log by its --list-all ID (index or timestamp).")
    g.add_argument("--complete-log", action="store_true", help="Print the current run's full log.")
    g.add_argument("--cleanup", action="store_true", help="Collapse Keras progress-bar updates in every log dir.")
    g.add_argument( "--delete", nargs="+", metavar="STATUS", choices=statuses, help=f"Delete runs with the given status(es): {' '.join(s for s in statuses if s != 'all')} | all. A currently-active run is never deleted.", )
    p_log.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt (used with --delete).")

def dispatch_log(rk: RunKind, args: argparse.Namespace) -> None:
    """Run the ``log`` subcommand parsed by :func:`add_log_subparser`."""
    if args.list_all:
        do_list_logs(rk)
    elif args.show is not None:
        do_show_log(rk, args.show)
    elif args.delete:
        do_delete_logs(rk, args.delete, assume_yes=args.yes)
    elif args.complete_log:
        do_complete_log(rk)
    elif args.cleanup:
        do_cleanup_logs(rk)
    else:
        do_tail(rk, None if args.currently_running is UNSET else args.currently_running)
