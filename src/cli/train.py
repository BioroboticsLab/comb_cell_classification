"""``ccc training`` command — launch and manage background training runs, one tmux session and one free GPU each."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
    dispatch_log,
    do_kill,
    exit_all_gpus_busy,
    resolve_active_target,
)
from src.core.config.config import RUN_FILES
from src.core.logs.reading import iter_log_lines

HOME_CCC = Path.home() / "ccc"
ENV_FILE = HOME_CCC / ".last_training_env"
JOB_LABEL = "training"


def _pop_flag_value(argv: list[str], flag: str) -> tuple[list[str], Optional[str]]:
    """Remove ``flag VALUE`` from ``argv`` if present and return the remaining argv plus the value."""
    # Must be stripped before setup_training_project: its strict argparse would reject the unknown --gpu-id flag.
    argv = list(argv)
    if flag in argv:
        i = argv.index(flag)
        value = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
        return argv, value
    return argv, None

############################
#
# This is where the sweep is configured. Base config overrides + the variation spec below. Edit here to change what a ccc training run trains or create a config.yaml to point to
#
############################

def _train_driver(argv: list[str]) -> None:
    """Build the sweep configs and batch-train, running as the detached child process (``python -m src.cli.train _train``)."""
    import src.core.project_settings as settings
    import src.training.training.training as training
    from src.core.config.config import BeeCombConfig
    from src.core.config.config_utils import build_cfg_variations, _set_nested_attr
    from src.training.setup_training import setup_training_project
    from src.training.training.utils import GpuError

    # The parent CLI already claimed a GPU (or decided on CPU) before launching
    argv, gpu_id_raw = _pop_flag_value(argv, "--gpu-id")
    if gpu_id_raw is None:
        sys.exit("--gpu-id is required: start training via `ccc training`, which claims a GPU first.")
    gpu_id = GpuError.NO_GPU_FOUND if gpu_id_raw == "cpu" else int(gpu_id_raw)

    # Sweep YAML, already validated by the parent CLI.
    argv, config_path = _pop_flag_value(argv, "--config")

    # The parent created output/training/<ts>/ and hands it down so everything this run produces (wandb files, model checkpoints) lands under it.
    argv, run_dir_raw = _pop_flag_value(argv, "--run-dir")
    run_dir = Path(run_dir_raw) if run_dir_raw else None
    if run_dir is not None:
        # sets wandb dir under run_dir
        os.environ["WANDB_DIR"] = str(run_dir)

    # --offline moed for wandb
    if "--offline" in argv:
        argv = [a for a in argv if a != "--offline"]
        os.environ["WANDB_MODE"] = "offline"

    cfg = BeeCombConfig()
    setup_training_project(cfg, argv=argv) #important!

    if run_dir is not None:
        # Set on the base cfg BEFORE build_cfg_variations below: it deep-copies the cfg, so every sweep variation checkpoints under this run's model/.
        model_root = run_dir / "model"
        model_root.mkdir(parents=True, exist_ok=True)
        cfg.paths.model_save_path = model_root

    # YAML config. Base values override the BeeCombConfig defaults. Sweep lists below replace the spec defaults (explicit CLI sweep flags still win).
    base_overrides, sweep_overrides = {}, {}
    if config_path:
        from src.core.config.sweep import load_training_yaml
        base_overrides, sweep_overrides = load_training_yaml(Path(config_path))
        for path, value in base_overrides.items():
            _set_nested_attr(cfg, path, value)

    cfgs = list[BeeCombConfig]()

    if settings.project_settings.debug:
        ################
        #
        # This part is for debugging a training run
        #
        ################
        if sweep_overrides:
            logging.info("--debug: config sweep ignored (single MobileNetV2 run); base overrides applied.")
        cfg.training.model = "MobileNetV2"
        cfgs.append(cfg)

    else:
        # Sweep defaults come from BeeCombConfig (already including any YAML base overrides)
        spec = {
            "training.model":                       ("models",              [cfg.training.model]),
            "training.use_adaptive_lr":             ("use_adaptive_lr",     [cfg.training.use_adaptive_lr]),
            "training.learning_rate":               ("learning_rates",      [cfg.training.learning_rate]),
            "training.unfreeze_after_epochs":       ("unfreeze_after",      [cfg.training.unfreeze_after_epochs]),
            "training.finetune_learning_rate":      ("finetune_lrs",        [cfg.training.finetune_learning_rate]),
            "training.augmentation":                ("augmentation_combos", [cfg.training.augmentation]),
            "dataset.cell_outer_layer":             ("cell_outer_layers",   [cfg.dataset.cell_outer_layer]),
            "training.oversample_minority_classes": ("oversamples",         [cfg.training.oversample_minority_classes]),
            "training.use_class_weight":            ("use_class_weights",   [cfg.training.use_class_weight]),
        }
        # YAML sweep lists replace the defaults above; dims not in the spec (any valid dotted path) sweep too, with no CLI counterpart (attr=None).
        for path, values in sweep_overrides.items():
            spec[path] = (spec[path][0] if path in spec else None, values)
        cfgs = build_cfg_variations(cfg, spec)

    training.batch_train(dataset_root=cfg.paths.raw_data_path, cfgs=cfgs, gpu_id=gpu_id)


def _start_run(debug: bool, offline: bool = False, config: Optional[str] = None) -> None:
    """Start one more background training run on a free GPU (additive; fails fast if none is free), validation of config beforehand"""
    from src.core.runs import AllGpusBusy, start_run

    config_abs: Optional[Path] = None
    if config:
        from src.core.config.sweep import load_training_yaml
        config_abs = Path(config).expanduser().resolve()
        base, sweep = load_training_yaml(config_abs)
        n_runs = 1
        for values in sweep.values():
            n_runs *= len(values)
        print(f"Config {config_abs}: {len(base)} base override(s), {n_runs} sweep run(s).")

    if debug:
        print("Debug mode: --debug (wandb=ccc_debug, dataset=data/annotated_DEBUG).")
    if offline:
        print("Offline mode: --offline (WANDB_MODE=offline, no W&B API key, logs locally).")

    child_args = ["--non-interactive"]
    extra_files = {RUN_FILES.flags_filename: json.dumps({"debug": debug, "offline": offline})}
    if config_abs is not None:
        # Record which sweep YAML this run used so later tooling (e.g. an OOM retrain) can read what it was configured with.
        extra_files[RUN_FILES.config_ref_filename] = f"{config_abs}\n"
        child_args += ["--config", str(config_abs)]
    if debug:
        child_args.append("--debug")
    if offline:
        child_args.append("--offline")

    try:
        # for tmux session
        run = start_run(kind="training", module="src.cli.train", entry="_train", child_args=child_args, extra_files=extra_files)
    except AllGpusBusy as e:
        exit_all_gpus_busy(e)

    log_file = run.log_file
    # Persist run/log/session for the printed source helper .last_training_env
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(f"RUN_DIR={run.run_dir}\nLOG_FILE={log_file}\nLOG_DIR={log_file.parent}\nSESSION={run.session}\n")

    print(f"{JOB_LABEL} started in tmux session {run.session} on {run.gpu_label}")
    print(f"Run dir: {run.run_dir}")
    print(f"Logging to {log_file}")

    print()
    print(f"  tmux attach -t {run.session}   # live session, Ctrl+B D to detach")
    print(f"  tail -f {log_file}")
    print(f"  tmux kill-session -t {run.session}")
    print(f"  source {ENV_FILE}   # restore vars in a new shell")


_RE_START = re.compile(r"Starting config (\d+) \((.+?)\)\.")
_RE_OK = re.compile(r"Config (\d+) \((.+?)\) finished successfully\.")
_RE_FAIL = re.compile(r"Config (\d+) \((.+?)\) failed:")
_RE_SWEEP_FAIL = re.compile(r"RuntimeError: \d+ of \d+ training run\(s\) failed")
_TRACEBACK = "Traceback (most recent call last)"

def _parse_run(log_dir: Path) -> dict:
    """Extract model variants + failure markers from a run's log."""
    started: dict[int, str] = {}
    finished: set[int] = set()
    failed: set[int] = set()
    sweep_failed = has_traceback = False
    for line in iter_log_lines(log_dir):
        if (m := _RE_START.search(line)):
            started[int(m.group(1))] = m.group(2)
        elif (m := _RE_OK.search(line)):
            finished.add(int(m.group(1)))
        elif (m := _RE_FAIL.search(line)):
            failed.add(int(m.group(1)))
        elif _RE_SWEEP_FAIL.search(line):
            sweep_failed = True
        elif _TRACEBACK in line:
            has_traceback = True

    variants = [
        (started[i], "failed" if i in failed else "ok" if i in finished else "unknown")
        for i in sorted(started)
    ]
    return {
        "variants": variants,
        "sweep_failed": sweep_failed,
        "has_traceback": has_traceback,
        "any_started": bool(started),
        "all_finished": bool(started) and all(i in finished for i in started),
        # Configs run sequentially, so if the highest-numbered started config neither finished nor failed, the run was interrupted mid-config.
        "last_unfinished": bool(started) and max(started) not in finished and max(started) not in failed,
    }

def _status_from_parsed(log_dir: Path, p: dict) -> str:
    if (log_dir / "stopped").exists():
        return "stopped"
    if p["sweep_failed"]:
        return "partial"   # sweep completed but >=1 variant failed
    if p["has_traceback"]:
        return "crashed"   # died outright (uncaught error)
    if p["all_finished"]:
        return "success"
    if p["any_started"] and p["last_unfinished"]:
        return "stopped"   # killed mid-run, no marker (legacy / external kill)
    return "unknown"


def _summarize(log_dir: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return (status, [(model, status), ...]) for a run, parsed from its log."""
    p = _parse_run(log_dir)
    return ("running" if run_alive(log_dir) else _status_from_parsed(log_dir, p)), p["variants"]


def _variants_text(variants: list[tuple[str, str]]) -> Text:
    from rich.text import Text

    if not variants:
        return Text("—", style="dim")
    t = Text()
    for j, (model, st) in enumerate(variants):
        if j:
            t.append(", ")
        mark, style = {"ok": ("✓ ", "green"), "failed": ("✗ ", "red")}.get(st, ("? ", "dim"))
        t.append(mark, style=style)
        t.append(model)
    return t

def _summarize_rich(log_dir: Path) -> "tuple[str, Text]":
    status, variants = _summarize(log_dir)
    return status, _variants_text(variants)


TRAINING = RunKind(
    kind="training",
    job_label=JOB_LABEL,
    detail_header="Variants",
    summarize=_summarize_rich,
)


def _do_restart(target: Optional[str], debug: bool, offline: bool = False, config: Optional[str] = None) -> None:
    active = active_runs(TRAINING)
    for run in resolve_active_target(TRAINING, target, active, "restart"):
        stop_run(run, job_label=JOB_LABEL)
    _start_run(debug=debug, offline=offline, config=config)


def start_default_run() -> None:
    """Start one run with all defaults (used by the ccc main menu when the wizard picks all defaults)."""
    _start_run(debug=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccc training",
        description="Launch and manage training runs. With no subcommand, launches one more run (additive).",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Pass --debug to the training driver (ccc_debug wandb project, data/annotated_DEBUG dataset).")
    parser.add_argument("--offline", action="store_true", help="Run fully offline: sets WANDB_MODE=offline in the driver so no W&B API key is needed and results are logged locally.")
    parser.add_argument( "--config", metavar="YAML", default=None, help="Training sweep YAML: base overrides + sweep value " "lists (cartesian product). Explicit CLI sweep flags still win over the YAML.", )
    parser.add_argument( "-k", "--kill", nargs="?", const=None, default=UNSET, metavar="ID", help="Kill training run(s): bare kills the sole active run, or shows a picker table " "if several are active; ID kills that one; 'all' kills every active run.", )
    parser.add_argument( "-r", "--restart", nargs="?", const=None, default=UNSET, metavar="ID", help="Restart training run(s): same ID/'all'/picker selection as --kill, then starts a fresh run.", )
    
    add_log_subparser(parser, TRAINING)
    return parser

def main(argv: Optional[list[str]] = None) -> None:
    #################
    #
    # Runs inside of tmux as detached process
    #
    # Internal mode, run inside the tmux session started by _start_run:
    #   python -m src.cli.train _train [driver args]   the training process
    raw = sys.argv[1:] if argv is None else argv
    if raw and raw[0] == "_train":
        _train_driver(raw[1:])
        return
    #
    ################

    args = _build_parser().parse_args(argv)

    if args.command == "log":
        dispatch_log(TRAINING, args)
        return

    if args.kill is not UNSET:
        do_kill(TRAINING, args.kill)
        return
    if args.restart is not UNSET:
        _do_restart(args.restart, debug=args.debug, offline=args.offline, config=args.config)
        return

    _start_run(debug=args.debug, offline=args.offline, config=args.config)


if __name__ == "__main__":
    main()
