"""``ccc classify`` command — run a trained model in its own background tmux session to write predicted labels into annotation JSONs in place (or ``--reset`` them, or ``--evaluate`` against existing ground-truth labels), with the same kill/restart/log management as ``ccc training``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rich.text import Text

from src.core.runs import (
    run_alive,
    stop_run,
)
from src.cli.run_cli import (
    UNSET,
    RunKind,
    active_runs,
    add_log_subparser,
    basic_status,
    dispatch_log,
    do_kill,
    exit_all_gpus_busy,
    resolve_active_target,
)
from src.core.config.config import RUN_FILES

JOB_LABEL = "classify"
ARGS_FILENAME = RUN_FILES.args_filename
DEFAULT_BATCH_SIZE = 64
DEFAULT_BACKUP_DIR = "unlabeled_jsons"


def _saved_args(run_dir: Path) -> Optional[list[str]]:
    f = run_dir / ARGS_FILENAME
    if f.is_file():
        try:
            return json.loads(f.read_text())
        except (ValueError, OSError):
            return None
    return None


def _arg_value(saved: list[str], *flags: str) -> Optional[str]:
    for flag in flags:
        if flag in saved:
            i = saved.index(flag)
            if i + 1 < len(saved):
                return saved[i + 1]
    return None


def _summarize(run_dir: Path) -> "tuple[str, Text]":
    from rich.text import Text

    status = "running" if run_alive(run_dir) else basic_status(run_dir)
    saved = _saved_args(run_dir)
    if not saved:
        return status, Text("—", style="dim")
    detail = Text(_arg_value(saved, "-i", "--images-dir") or "—")
    model = _arg_value(saved, "-m", "--model-dir")
    if model:
        detail.append(f"  ({Path(model).name})", style="dim")
    if "--evaluate" in saved:
        detail.append("  [evaluate]", style="cyan")
    return status, detail



CLASSIFY = RunKind(kind="classify", job_label=JOB_LABEL, detail_header="Images", summarize=_summarize)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccc classify",
        description=(
            "Load a trained .keras model, predict a class for every annotation in "
            "each image's JSON, and write it into the 'label' field in place. "
            "Runs in the background in its own tmux session (watch it with `ccc classify log`). "
            "Pass --reset to blank all labels back to 'unlabeled' instead."
        ),
    )
    parser.add_argument("-i", "--images-dir", default=None, help="Directory of images whose sibling <stem>.json annotations get labelled. " "Accepts a glob (e.g. '.../cam-*') to process several directories at once. " "Required when starting a run.")
    parser.add_argument("--reset", action="store_true", help="Reset every annotation's label back to 'unlabeled' (no model is run) " "and exit. Skips the backup subfolder so pristine copies are untouched. " "Runs in the foreground.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluation mode: treat the labels already in the JSONs as ground " "truth, score the model against them (F1 macro/micro/weighted + " "confusion matrix), and write a report under the configured " "evaluation_results folder. The annotation files are NOT modified.")
    parser.add_argument("--offline", action="store_true", help="In --evaluate mode, log the report to W&B locally " "(WANDB_MODE=offline, no W&B API key, no server contact).")
    parser.add_argument("--eval-out", default=None, help="In --evaluate mode, write the report to this exact path instead of " "the auto-named file under the evaluation_results folder.")
    parser.add_argument("-m", "--model-dir", default=None, help="Use this exact model folder (holding the .keras checkpoint and its " "model_info.json / label_map.txt). If omitted, pick one interactively " "from a table of trained models under --models-root.")
    parser.add_argument("--model-id", type=int, default=None, help="Pick a model non-interactively by its ID in the --models-root table " "(as shown by the interactive picker).")
    parser.add_argument("--models-root", default=None, help="Folder scanned for trained models when selecting by table/ID " "(default: the output/ run tree plus models/).")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Crops per prediction batch (default: 64).", )
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True, help="Copy pristine annotation JSONs into the backup subfolder before " "labelling (default: on). Use --no-backup to skip.")
    parser.add_argument("--restore", action="store_true", help="Reset each annotation from its backup before labelling, so " "inference runs on the pristine JSON (e.g. to re-label with a new model).")
    parser.add_argument("--skip-labeled", action="store_true", help="Skip annotation files whose cells all already carry a label, so an " "interrupted run can be resumed. Partially labelled files are still " "processed.")
    parser.add_argument("--backup-dir", default=DEFAULT_BACKUP_DIR, help="Name of the backup subfolder under the images dir (default: unlabeled_jsons).")
    parser.add_argument("-k", "--kill", nargs="?", const=None, default=UNSET, metavar="ID", help="Kill classify run(s): bare kills the sole active run, or shows a picker table " "if several are active; ID kills that one; 'all' kills every active run.")
    parser.add_argument("-r", "--restart", nargs="?", const=None, default=UNSET, metavar="ID", help="Restart classify run(s): same ID/'all'/picker selection as --kill; each " "restarted run relaunches with the arguments it was originally started with.")
    # Internal flags passed by _start_run to the detached child.
    parser.add_argument("--gpu-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    add_log_subparser(parser, CLASSIFY, statuses=["success", "crashed", "stopped", "unknown", "all"])
    return parser


def _child_argv(args: argparse.Namespace, model_dir: str) -> list[str]:
    """The fully explicit argv for the detached child: every option this parser accepts, resolved (the child has no TTY and no defaults to fall back on)."""
    argv = [
        "-i", args.images_dir,
        "--model-dir", model_dir,
        "-b", str(args.batch_size),
        "--backup-dir", args.backup_dir,
    ]
    if args.offline:
        argv.append("--offline")
    if not args.backup:
        argv.append("--no-backup")
    if args.restore:
        argv.append("--restore")
    if args.skip_labeled:
        argv.append("--skip-labeled")
    return argv


def _start_run(child_args: list[str]) -> None:
    """Start one more background classify run on a free GPU (additive; fails fast if none is free)."""
    from src.core.runs import AllGpusBusy, start_run

    try:
        # child_args is either a fresh _child_argv() or a saved args.json replayed by --restart.
        run = start_run("classify", "src.cli.classify",entry="_classify",child_args=child_args,
                        extra_files={ARGS_FILENAME: json.dumps(child_args)})
    except AllGpusBusy as e:
        exit_all_gpus_busy(e)
    print(f"{JOB_LABEL} started in tmux session {run.session} on {run.gpu_label}")
    print(f"Run dir: {run.run_dir}")
    print(f"Logging to {run.log_file}")

    print()
    print(f"  tmux attach -t {run.session}   # live session, Ctrl+B D to detach")
    print(f"  tail -f {run.log_file}")
    print(f"  tmux kill-session -t {run.session}")


def _do_restart(target: Optional[str]) -> None:
    """Stop the selected run(s) and relaunch each with the args it was started with (saved in its args.json)."""
    for run in resolve_active_target(CLASSIFY, target, active_runs(CLASSIFY), "restart"):
        saved = _saved_args(run)
        if saved is None:
            sys.exit(f"Run {run.name} has no {ARGS_FILENAME}; cannot restart it. Start a new run with `ccc classify -i ...`.")
        stop_run(run, job_label=JOB_LABEL)
        _start_run(saved)


def _classify_driver(argv: list[str]) -> None:
    """Run inference/evaluation, running as the detached child process (``python -m src.cli.classify _classify``)."""
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(run_dir)

    from src.training.training.utils import GpuError, pin_process_to_gpu

    pin_process_to_gpu(GpuError.NO_GPU_FOUND if args.gpu_id == "cpu" else int(args.gpu_id))

    from src.inference.inference import run_inference

    run_inference(
        images_dir=args.images_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        backup_jsons=args.backup,
        restore=args.restore,
        backup_dirname=args.backup_dir,
        skip_labeled=args.skip_labeled,
        # Keep a per-run copy of every JSON this run labelled.
        labeled_copy_dir=run_dir / "jsons",
    )


def main(argv: Optional[list[str]] = None) -> None:
    # Internal mode, run inside the tmux session started by _start_run:
    #   python -m src.cli.classify _classify [driver args]   the labelling process
    raw = sys.argv[1:] if argv is None else argv
    if raw and raw[0] == "_classify":
        _classify_driver(raw[1:])
        return

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "log":
        dispatch_log(CLASSIFY, args)
        return
    if args.kill is not UNSET:
        do_kill(CLASSIFY, args.kill)
        return
    if args.restart is not UNSET:
        _do_restart(args.restart)
        return

    if args.images_dir is None:
        parser.error("-i/--images-dir is required to start a run (or use log/--kill/--restart).")

    if args.reset:
        from src.core.logs.paths import run_log_file
        from src.core.logs.setup import setup_run_logging
        from src.inference.reset import reset_labels

        run_dir = setup_run_logging("classify")
        print(f"Logging this run to {run_log_file(run_dir)}")
        reset_labels(images_dir=args.images_dir, backup_dirname=args.backup_dir)
        return

    model_dir = args.model_dir
    if model_dir is None:
        from src.core.logs.paths import output_root, project_root
        from src.inference.model_select import select_model

        roots = [Path(args.models_root)] if args.models_root else [output_root(), project_root() / "models"]
        model_dir = select_model(roots, model_id=args.model_id)

    _start_run(_child_argv(args, str(model_dir)))

if __name__ == "__main__":
    main()
