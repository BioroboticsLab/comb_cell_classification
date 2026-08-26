"""Interactive model picker for ``ccc classify``: lists trained models in a table and returns the chosen directory."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from src.core.model_info import read_model_info

if TYPE_CHECKING:
    from rich.table import Table

# <ts>_<model>_col<N>_lr<LR>[_adap-lr]_te<E>_<flag>
_NAME_RE = re.compile(
    r"^(?:(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_)?"
    r"(?P<model>.+?)_col(?P<col>\d+)_lr(?P<lr>[0-9.eE+-]+)"
    r"(?P<adap>_adap-lr)?_te(?P<te>\d+)_(?P<flag>cw|os|none|both)$"
)
@dataclass
class ModelInfo:
    path: Path
    name: str
    started: str = ""            # parsed timestamp, "YYYY-MM-DD HH:MM:SS"
    col: str = "—"
    lr: str = "—"
    adaptive: bool = False
    te: str = "—"
    cw: bool = False
    os: bool = False
    parsed: bool = False
    f1_macro: Optional[float] = None
    f1_micro: Optional[float] = None
    f1_weighted: Optional[float] = None

def _parse_run_name(name: str, path: Path) -> ModelInfo:
    """Best-effort parse of a run name into its hyperparameters (fallback for folders without a ``model_info.json``)."""
    m = _NAME_RE.match(name)
    if not m:
        return ModelInfo(path=path, name=name)
    started = ""
    if m["ts"]:
        try:
            started = datetime.strptime(m["ts"], "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            started = m["ts"]
    flag = m["flag"]
    return ModelInfo(
        path=path,
        name=m["model"],
        started=started,
        col=m["col"],
        lr=m["lr"],
        adaptive=bool(m["adap"]),
        te=m["te"],
        cw=flag in ("cw", "both"),
        os=flag in ("os", "both"),
        parsed=True,
    )

def _from_model_info(path: Path, info: dict) -> ModelInfo:
    """Build a :class:`ModelInfo` straight from a parsed ``model_info.json`` (the primary metadata source, since folders may be opaque hashes)."""
    hp = info.get("hyperparameters") or {}
    f1 = info.get("f1") or {}
    trained = info.get("trained") or {}
    lr, col, te = hp.get("learning_rate"), hp.get("cell_outer_layer"), hp.get("epochs")
    return ModelInfo(
        path=path,
        name=info.get("model") or path.name,
        started=trained.get("start") or "",
        col=str(col) if col is not None else "—",
        lr=str(lr) if lr is not None else "—",
        adaptive=bool(hp.get("adaptive_lr")),
        te=str(te) if te is not None else "—",
        cw=bool(hp.get("class_weight")),
        os=bool(hp.get("oversample")),
        parsed=lr is not None,  # gates the "(adaptive/fixed)" suffix in the table
        f1_macro=f1.get("macro"),
        f1_micro=f1.get("micro"),
        f1_weighted=f1.get("weighted"),
    )


def _model_at(path: Path, fallback_name: Optional[str] = None) -> ModelInfo:
    """A model at *path*: from its ``model_info.json`` if present, else name-parsed."""
    info = read_model_info(path)
    if info:
        return _from_model_info(path, info)
    return _parse_run_name(fallback_name or path.name, path)


def discover_models(models_root: Path) -> list[ModelInfo]:
    """Return all model dirs under *models_root* that hold a ``.keras`` checkpoint, newest first, with metadata filled in."""
    if not models_root.is_dir():
        return []
    patterns = ("*", "model/*", "*/model/*", "*/*/model/*")
    dirs = {d for pat in patterns for d in models_root.glob(pat)
            if d.is_dir() and any(d.glob("*.keras"))}
    models = [_model_at(d) for d in sorted(dirs)]
    models.sort(key=lambda m: (m.started, m.name), reverse=True) # Newest first by training start
    return models

def _f1_string(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "—"

def build_table(models: list[ModelInfo]) -> Table:
    from rich.table import Table

    table = Table(title="Trained models", header_style="bold", title_justify="left")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Model")
    table.add_column("Trained")
    table.add_column("col", justify="right")
    table.add_column("LR")
    table.add_column("te", justify="right")
    table.add_column("CW", justify="center")
    table.add_column("OS", justify="center")
    
    table.add_column("F1 macro", justify="right")
    table.add_column("F1 micro", justify="right")
    table.add_column("F1 wgt.", justify="right")

    yes, no = "[green]✓[/green]", "[dim]–[/dim]"
    for i, mi in enumerate(models, 1):
        lr = f"{mi.lr} ({'adaptive' if mi.adaptive else 'fixed'})" if mi.parsed else "—"
        table.add_row(
            str(i), mi.name, mi.started or "—", mi.col, lr, mi.te,
            yes if mi.cw else no, yes if mi.os else no,
            _f1_string(mi.f1_macro), _f1_string(mi.f1_micro), _f1_string(mi.f1_weighted),
        )
    return table


def select_model(models_roots: Union[Path, Sequence[Path]], model_id: Optional[int] = None) -> Path:
    """Return the chosen model directory from one or several scan roots, non-interactively via *model_id* or by prompting with the table."""
    from rich.console import Console

    roots = [models_roots] if isinstance(models_roots, Path) else list(models_roots)
    seen: set[Path] = set()
    models: list[ModelInfo] = []
    for root in roots:
        for m in discover_models(root):
            key = m.path.resolve()
            if key not in seen:
                seen.add(key)
                models.append(m)
    models.sort(key=lambda m: (m.started, m.name), reverse=True)
    if not models:
        sys.exit(f"No trained models (with a .keras checkpoint) found under {', '.join(str(r) for r in roots)}.")

    console = Console()
    if model_id is not None:
        if not (1 <= model_id <= len(models)):
            console.print(build_table(models))
            sys.exit(f"--model-id {model_id} out of range (1..{len(models)}).")
        return models[model_id - 1].path

    console.print(build_table(models))
    while True:
        try:
            raw = input(f"Select a model by ID (1..{len(models)}, q to abort): ").strip()
        except EOFError:
            sys.exit("\nNo selection (no input available).")
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit("Aborted.")
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1].path
        console.print(f"[red]Invalid ID {raw!r}[/red]; enter a number 1..{len(models)} or q.")
