"""Builds a :class:`~src.postprocess.grid.CombGrid` from one frame's predictions and applies an ordered pipeline of correction rules to it."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

from src.core.annotations import Annotation, AnnotationDoc
from src.postprocess.grid import CombGrid
from src.postprocess.rules import Correction, CorrectionRule

logger = logging.getLogger(__name__)


class SpatialCorrector:
    """Runs an ordered pipeline of :class:`CorrectionRule` over a frame's annotations, repeating up to ``max_passes`` times until a pass changes nothing."""

    def __init__(
        self,
        rules: list[CorrectionRule],
        max_passes: int = 1,
        neighbor_dist: Optional[float] = None,
    ) -> None:
        self.rules = rules
        self.max_passes = max_passes
        self.neighbor_dist = neighbor_dist

    def correct(self, annotations: list[Annotation]) -> tuple[list[Annotation], list[Correction]]:
        """Correct ``annotations`` (input untouched) and return ``(corrected_copies, audit_log)`` in application order."""
        grid = CombGrid(annotations, neighbor_dist=self.neighbor_dist)
        applied: list[Correction] = []

        for pass_idx in range(self.max_passes):
            pass_changes = 0
            for rule in self.rules:
                proposed = rule.apply(grid)
                for corr in proposed:
                    # Re-check: an earlier rule this pass may have already changed it.
                    if grid.labels[corr.cell_index] == corr.new_label:
                        continue
                    grid.set_label(corr.cell_index, corr.new_label)
                    applied.append(corr)
                    pass_changes += 1
            logger.debug("Spatial correction pass %d applied %d changes", pass_idx, pass_changes)
            if pass_changes == 0:
                break

        return grid.to_annotations(), applied


def correct_file(
    pred_path: Union[str, Path],
    out_path: Union[str, Path],
    rules: list[CorrectionRule],
    max_passes: int = 1,
    dry_run: bool = False,
) -> list[Correction]:
    """Load predictions, spatially correct them, write the result as JSON in the input schema (unless ``dry_run``), and return the audit log."""
    loader = AnnotationDoc.load(pred_path)
    corrector = SpatialCorrector(rules, max_passes=max_passes)
    corrected, log = corrector.correct(loader.annotations)

    if dry_run:
        for corr in log:
            logger.info(
                "[dry-run] %s: cell %d %s -> %s (%s)",
                Path(pred_path).name, corr.cell_index, corr.old_label, corr.new_label, corr.reason,
            )
        logger.info("[dry-run] %s: %d changes, nothing written", pred_path, len(log))
        return log

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([ann.to_dict() for ann in corrected], f, indent=2)

    logger.info("Corrected %s -> %s (%d changes)", pred_path, out_path, len(log))
    return log
