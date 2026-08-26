"""Thin adapters exposing the pipeline's correction rules to the GUI as plain-data functions (labels/centers/neighbor maps in, ``{index: new_label}`` out)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from src.core.annotations import Annotation
from src.postprocess.grid import SequenceGrid
from src.postprocess.rules import (
    DEFAULT_MAX_GAP_HOURS,
    DEFAULT_MAX_ISLAND_SIZE,
    DEFAULT_MIN_BORDER_FRAC,
    ClusterMajorityRule,
    NeighborMajorityRule,
    TemporalTransitionRule,
)

DEFAULT_IGNORE = frozenset({"empty_cell", "unlabeled"})

class _GraphView:
    """Duck-typed stand-in for ``CombGrid`` (``n``, ``labels``, ``neighbors``) over the GUI's cached neighbor map, so rules run without recomputing geometry."""

    def __init__(self, labels: list[str], neighbors: dict[int, list[int]]) -> None:
        self.labels = list(labels)
        self.n = len(self.labels)
        self._neighbors = neighbors

    def neighbors(self, i: int) -> list[int]:
        return self._neighbors.get(i, [])

def cluster_majority(
    labels: list[str],
    neighbors: dict[int, list[int]],
    max_island_size: int = DEFAULT_MAX_ISLAND_SIZE,
    min_border_frac: float = DEFAULT_MIN_BORDER_FRAC,
    ignore: frozenset[str] = DEFAULT_IGNORE,
) -> dict[int, str]:
    """Absorb small mislabelled islands into their surrounding region (pipeline ``ClusterMajorityRule``)."""
    rule = ClusterMajorityRule(max_island_size=max_island_size, min_border_frac=min_border_frac, ignore=set(ignore))
    return {correction.cell_index: correction.new_label for correction in rule.apply(_GraphView(labels, neighbors))}

def _annotations(centers: np.ndarray, labels: list[str]) -> list[Annotation]:
    return [
        Annotation(f"c{i}", float(x), float(y), 0.0, label)
        for i, ((x, y), label) in enumerate(zip(np.asarray(centers, dtype=float).reshape(-1, 2), labels))
    ]
    
def transition_fixes(
    target_labels: list[str],
    target_centers: np.ndarray,
    next_labels: list[str],
    next_centers: np.ndarray,
    gap_hours: float,
    radius: float,
    transitions: Optional[dict[str, set[str]]] = None,
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
    ignore: frozenset[str] = DEFAULT_IGNORE,
) -> dict[int, str]:
    """Correct earlier-frame cells from the next frame via the pipeline's ``TemporalTransitionRule`` over a 2-frame ``SequenceGrid`` (Hungarian matching)."""
    t0 = datetime(2000, 1, 1)
    grid = SequenceGrid(
        [_annotations(target_centers, target_labels), _annotations(next_centers, next_labels)],
        match_radius=radius,
        timestamps=[t0, t0 + timedelta(hours=gap_hours)],
    )
    rule = TemporalTransitionRule(transitions=transitions, max_gap_hours=max_gap_hours, ignore=set(ignore))
    return {correction.cell_index: correction.new_label for t, correction in rule.apply_sequence(grid) if t == 0}


def temporal_majority_fixes(
    prev_labels: list[str],
    prev_centers: np.ndarray,
    target_labels: list[str],
    target_centers: np.ndarray,
    next_labels: list[str],
    next_centers: np.ndarray,
    radius: float,
    min_agree: int = 2,
) -> dict[int, str]:
    """Flip a middle-frame cell to the label its previous- and next-frame matches agree on (pipeline ``NeighborMajorityRule`` on the temporal axis, as in ``ccc postprocess correct-temporal``)."""
    grid = SequenceGrid(
        [
            _annotations(prev_centers, prev_labels),
            _annotations(target_centers, target_labels),
            _annotations(next_centers, next_labels),
        ],
        match_radius=radius,
    )
    rule = NeighborMajorityRule(min_agree=min_agree)
    return {correction.cell_index: correction.new_label for correction in rule.apply(grid.frame_view(1, "temporal"))}
