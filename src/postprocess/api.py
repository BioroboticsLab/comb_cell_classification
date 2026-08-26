"""Frontend-facing postprocess orchestration (backup, spatial/temporal correction over files, revert) shared by the CLI, TUI and napari — moved out of ``src.cli.postprocess`` so non-CLI frontends don't re-implement it."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from src.postprocess.rules import (
    DEFAULT_MAX_ISLAND_SIZE,
    DEFAULT_MIN_BORDER_FRAC,
    CorrectionRule,
    build_rule,
)
from src.postprocess.spatial_correction import correct_file
from src.postprocess.temporal_correction import SequenceCorrector, correct_dir

logger = logging.getLogger(__name__)

BKP_DIR = "bkp"


def backup_file(pred_path: Path) -> None:
    """Copy ``pred_path`` into a sibling ``bkp/`` folder, never overwriting an existing backup."""
    # An existing backup is kept so repeated correction runs can't replace the original with an already-corrected file
    bkp_path = pred_path.parent / BKP_DIR / pred_path.name
    if bkp_path.exists():
        logger.info("Backup already exists, keeping it: %s", bkp_path)
        return
    bkp_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pred_path, bkp_path)
    logger.info("Backed up %s -> %s", pred_path.name, bkp_path)


def default_rules(max_island_size: int = DEFAULT_MAX_ISLAND_SIZE, min_border_frac: float = DEFAULT_MIN_BORDER_FRAC) -> list[CorrectionRule]:
    """Build the default spatial-correction pipeline."""
    return [build_rule("cluster_majority", max_island_size=max_island_size, min_border_frac=min_border_frac)]


def _input_files(input_path: Path) -> list[Path]:
    """JSON files to process: the file itself, or ``*.json`` directly in the dir."""
    return sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]


def correct_spatial(input_path: Path, rules: list[CorrectionRule], max_passes: int = 1, dry_run: bool = False, backup: bool = False) -> None:
    """Spatially correct a prediction JSON file or directory in place, optionally backing up originals to ``bkp/`` first."""
    input_path = Path(input_path)
    files = _input_files(input_path)
    if not files:
        logger.warning("No prediction JSON files in %s", input_path)
    total = 0
    for pred_path in files:
        if backup and not dry_run:
            backup_file(pred_path)
        log = correct_file(pred_path, pred_path, rules, max_passes=max_passes, dry_run=dry_run)
        total += len(log)
        logger.info("%s: %d corrections", pred_path.name, len(log))
    if dry_run:
        logger.info("[dry-run] %d files, %d corrections total, nothing written", len(files), total)


def correct_temporal( input_path: Path, corrector: SequenceCorrector, cam: Optional[str] = None, dry_run: bool = False, backup: bool = False, ) -> None:
    """Temporally correct a directory of prediction JSONs in place; run after spatial correction and after breakpoints are marked."""
    # Frames are grouped per camera and split into static blocks at the
    # sequence_breakpoint markers (camera moves invalidate cell matching);
    # within a block each cell's label is majority-voted across frames.
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError("correct-temporal needs a directory of frame JSONs, not a single file.")
    if backup and not dry_run:
        for pred_path in _input_files(input_path):
            backup_file(pred_path)
    total = correct_dir(input_path, corrector, cam=cam, dry_run=dry_run)
    logger.info("Temporal correction: %d correction(s) total%s", total, " (dry-run, nothing written)" if dry_run else "")


def revert(path: Path) -> None:
    """Restore a file or directory from its ``bkp/`` folder (backups are kept, so reverting is idempotent)."""
    path = Path(path)
    if path.is_dir():
        bkp_dir = path / BKP_DIR
        backups = sorted(bkp_dir.glob("*.json")) if bkp_dir.is_dir() else []
        if not backups:
            logger.warning("No backups found in %s", bkp_dir)
            return
        for bkp_path in backups:
            shutil.copy2(bkp_path, path / bkp_path.name)
            logger.info("Reverted %s", path / bkp_path.name)
        logger.info("Reverted %d files from %s", len(backups), bkp_dir)
    else:
        bkp_path = path.parent / BKP_DIR / path.name
        if not bkp_path.exists():
            raise ValueError(f"No backup found for {path} (expected {bkp_path})")
        shutil.copy2(bkp_path, path)
        logger.info("Reverted %s from %s", path, bkp_path)
