"""Read/write the per-model ``model_info.json`` next to each checkpoint — index-ordered class names, hyperparameters, timing, machine info, and F1 scores — written once before ``fit`` (crash-safe) and once after evaluation."""

from __future__ import annotations

import getpass
import json
import logging
import math
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from src.core.config.config import BeeCombConfig
    from src.core.evaluation import EvaluationResult

logger = logging.getLogger(__name__)

INFO_FILENAME = "model_info.json"

def f1_section(f1_scores: dict[str, float]) -> dict[str, Any]:
    """Split a ``compute_f1_score`` result (``f1_all_*`` aggregates, ``f1_<class>`` per class) into aggregate + per-class F1."""
    per_class = {
        key[len("f1_"):]: value
        for key, value in f1_scores.items()
        if key.startswith("f1_") and not key.startswith("f1_all_")
    }
    return {
        "macro": f1_scores.get("f1_all_macro"),
        "micro": f1_scores.get("f1_all_micro"),
        "weighted": f1_scores.get("f1_all_weighted"),
        "per_class": per_class,
    }

def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None

def _total_ram_gb() -> Optional[int]:
    """Installed RAM in whole GiB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return math.ceil(int(line.split()[1]) / 1024 / 1024)
    except OSError:
        pass
    return None

def _gpu_name() -> Optional[str]:
    """Name of the machine's GPU(s) via nvidia-smi; identical GPUs are collapsed."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
    if not names:
        return None
    uniq = sorted(set(names))
    return names[0] if len(uniq) == 1 else ", ".join(uniq)

def collect_machine_info() -> dict[str, Any]:
    """User + hardware (CPU / GPU / total RAM) of the machine running training."""
    return {
        "user": getpass.getuser(),
        "cpu": _cpu_model(),
        "gpu": _gpu_name(),
        "ram_gb": _total_ram_gb(),
    }

def build_timing(start: Optional[Union[datetime, float]], end: Optional[Union[datetime, float]] = None) -> dict[str, Any]:
    """Training timing block from start/end (``datetime`` or epoch-seconds); duration fields are filled only when both ends are known."""
    def to_dt(x: Optional[Union[datetime, float]]) -> Optional[datetime]:
        if x is None:
            return None
        return datetime.fromtimestamp(x) if isinstance(x, (int, float)) else x

    s, e = to_dt(start), to_dt(end)
    timing: dict[str, Any] = {
        "start": s.strftime("%Y-%m-%d %H:%M:%S") if s else None,
        "end": e.strftime("%Y-%m-%d %H:%M:%S") if e else None,
        "duration": None,
        "duration_seconds": None,
    }
    if s and e:
        secs = int((e - s).total_seconds())
        timing["duration_seconds"] = secs
        timing["duration"] = str(timedelta(seconds=secs))
    return timing

def build_model_info(
    cfg: BeeCombConfig,
    class_names: list[str],
    eval_result: Optional[EvaluationResult] = None,
    trained: Optional[dict[str, Any]] = None,
    machine: Optional[dict[str, Any]] = None,
    folder_name: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the ``model_info.json`` payload; ``eval_result=None`` gives the pre-fit write with ``f1`` still null."""
    hp = cfg.training_cfg_as_dict()
    hp.update({
        "adaptive_lr": cfg.training.use_adaptive_lr,
        "class_weight": cfg.training.use_class_weight,
        "oversample": cfg.training.oversample_minority_classes,
    })
    return {
        "model": cfg.training.model,
        "folder_name": folder_name,
        "trained": trained,
        "class_names": list(class_names),
        "hyperparameters": hp,
        "f1": f1_section(eval_result.f1_scores) if eval_result is not None else None,
        "machine": machine,
    }

def dump_model_info(model_dir: Path, info: dict[str, Any]) -> None:
    """Write a pre-built ``model_info`` dict to *model_dir*, best-effort (never raises)."""
    try:
        (model_dir / INFO_FILENAME).write_text(json.dumps(info, indent=2))
        n = len(info.get("class_names") or [])
        logger.info("Wrote %s (%d classes) to %s", INFO_FILENAME, n, model_dir)
    except OSError as e:
        logger.warning("Could not write %s to %s: %s", INFO_FILENAME, model_dir, e)

def write_model_info(
    model_dir: Path,
    cfg: BeeCombConfig,
    class_names: list[str],
    eval_result: Optional[EvaluationResult] = None,
    trained: Optional[dict[str, Any]] = None,
    machine: Optional[dict[str, Any]] = None,
    folder_name: Optional[str] = None,
) -> None:
    """Build the payload from a config + evaluation result and write it, best-effort."""
    dump_model_info(model_dir, build_model_info(cfg, class_names, eval_result=eval_result, trained=trained, machine=machine, folder_name=folder_name))

def read_model_info(model_dir: Path) -> Optional[dict[str, Any]]:
    """Return the parsed ``model_info.json`` for *model_dir*, or None if absent/invalid."""
    f = model_dir / INFO_FILENAME
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except (ValueError, OSError):
        return None

def class_names_from_info(model_dir: Path) -> Optional[list[str]]:
    """Index-ordered class names from *model_dir*'s ``model_info.json``, or None."""
    info = read_model_info(model_dir)
    names = info.get("class_names") if info else None
    return list(names) if names else None
