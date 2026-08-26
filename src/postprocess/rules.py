"""Registry of pluggable correction rules that read a cell's neighbor context and *propose* label changes, which the correctors then apply — rules never mutate the grid themselves, keeping them composable and order-independent within a pass."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.postprocess.grid import CombGrid, SequenceGrid, connected_components

logger = logging.getLogger(__name__)

# Single source of the correction defaults shared by CLI, TUI and napari.
DEFAULT_MAX_ISLAND_SIZE = 4
DEFAULT_MIN_BORDER_FRAC = 0.6
DEFAULT_MAX_GAP_HOURS = 36.0


@dataclass(frozen=True)
class Correction:
    """A single proposed label change for one cell."""
    cell_index: int
    old_label: str
    new_label: str
    reason: str

class CorrectionRule(ABC):
    """Base class for a correction rule: inspect ``grid`` read-only and return proposed corrections (empty list = no change)."""
    name: str = "base"

    @abstractmethod
    def apply(self, grid: CombGrid) -> list[Correction]:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


RULE_REGISTRY: dict[str, type[CorrectionRule]] = {}

def register(name: str) -> Callable[[type[CorrectionRule]], type[CorrectionRule]]:
    """Class decorator registering a rule under ``name`` for :func:`build_rule`."""

    def deco(cls: type[CorrectionRule]) -> type[CorrectionRule]:
        if name in RULE_REGISTRY:
            raise ValueError(f"Correction rule {name!r} is already registered.")
        cls.name = name
        RULE_REGISTRY[name] = cls
        return cls

    return deco

def build_rule(name: str, **cfg: Any) -> CorrectionRule:
    """Instantiate a registered rule by name with keyword config."""
    if name not in RULE_REGISTRY:
        raise ValueError(f"Unknown correction rule {name!r}. Available: {sorted(RULE_REGISTRY)}")
    return RULE_REGISTRY[name](**cfg)

def list_rules() -> list[str]:
    """Names of all registered rules."""
    return sorted(RULE_REGISTRY)

@register("neighbor_majority")
class NeighborMajorityRule(CorrectionRule):
    """Flip a cell to the most common neighbor label when at least ``min_agree`` neighbors share it and it differs from the cell's own label."""

    # min_agree=6 is the strict hard rule "all 6 ring neighbors are X -> the
    # cell is X"; 4-5 gives softer majority voting. ``exclude`` labels are never
    # proposed as replacement, e.g. to stop sparse combs collapsing into
    # ``empty_cell``.

    def __init__(self, min_agree: int = 6, exclude: Optional[set[str]] = None) -> None:
        self.min_agree = min_agree
        self.exclude = set(exclude or ())

    def apply(self, grid: CombGrid) -> list[Correction]:
        out: list[Correction] = []
        for i in range(grid.n):
            neigh = grid.neighbor_labels(i)
            # Border cells with fewer than min_agree neighbors can never reach
            # the vote threshold, so they are implicitly protected here.
            if len(neigh) < self.min_agree:
                continue
            label, count = Counter(neigh).most_common(1)[0]
            if count < self.min_agree:
                continue
            if label in self.exclude:
                continue
            if label != grid.labels[i]:
                reason=f"{count}/{len(neigh)} neighbors are {label!r}"
                out.append(Correction(cell_index=i, old_label=grid.labels[i], new_label=label, reason=reason))
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"NeighborMajorityRule(min_agree={self.min_agree}, exclude={self.exclude or set()})"

@register("cluster_majority")
class ClusterMajorityRule(CorrectionRule):
    """Flip a small connected island of same-labelled cells to the label that dominates its border, absorbing it into the surrounding region."""

    def __init__(self, max_island_size: int = DEFAULT_MAX_ISLAND_SIZE, min_border_frac: float = DEFAULT_MIN_BORDER_FRAC, ignore: Optional[set[str]] = None) -> None:
        self.max_island_size = max_island_size
        self.min_border_frac = min_border_frac
        self.ignore = set(ignore) if ignore is not None else {"empty_cell", "unlabeled"}

    def _components(self, grid: CombGrid) -> list[list[int]]:
        """Return connected components of the same-label hex graph, skipping ``ignore`` labels."""
        return connected_components(n=grid.n, neighbors=grid.neighbors, key=lambda i: None if grid.labels[i] in self.ignore else grid.labels[i]) # Falls Komponente Label in Ignore hat, dann wird au None gesetzt

    def apply(self, grid: CombGrid) -> list[Correction]:
        out: list[Correction] = []
        for comp in self._components(grid):
            if len(comp) > self.max_island_size:
                continue
            own_label = grid.labels[comp[0]]
            members = set(comp)
            # Labels of the cells ringing the island: outside it, not ignored, and not the island's own label
            border = [
                grid.labels[j]
                for i in comp
                    for j in grid.neighbors(i)
                        if j not in members and grid.labels[j] not in self.ignore and grid.labels[j] != own_label
            ]
            if not border: # skip component with neighbours with different non-ignored labels
                continue
            label, count = Counter(border).most_common(1)[0]
            if count / len(border) < self.min_border_frac:
                continue
            for i in comp:
                reason=f"island of {len(comp)} {own_label!r} ringed by {count}/{len(border)} {label!r}"
                out.append(Correction(cell_index=i, old_label=own_label, new_label=label, reason=reason))
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ClusterMajorityRule(max_island_size={self.max_island_size}, "
            f"min_border_frac={self.min_border_frac})"
        )

def default_transitions() -> dict[str, set[str]]:
    """Map each label to its biologically allowed predecessors one frame earlier."""
    # Read as "a cell that is X now could, one frame ago, only have been one of these"
    return {
        "capped_brood": {"capped_brood", "open_brood"},
        "open_brood": {"open_brood", "empty_cell"},
        "capped_honey": {"capped_honey", "open_honey"},
        "open_honey": {"open_honey", "empty_cell"},
    }


class TemporalTransitionRule:
    """Correct an earlier frame from a later one: if a cell's predecessor label is not a valid predecessor per the biology transition table, overwrite it with the successor's label."""

    name = "temporal_transition"

    def __init__(self, transitions: Optional[dict[str, set[str]]] = None, max_gap_hours: float = DEFAULT_MAX_GAP_HOURS, ignore: Optional[set[str]] = None) -> None:
        self.transitions = transitions if transitions is not None else default_transitions()
        self.max_gap_hours = max_gap_hours
        self.ignore = set(ignore) if ignore is not None else {"unlabeled"}

    def apply_sequence(self, grid: SequenceGrid) -> list[tuple[int, Correction]]:
        out: list[tuple[int, Correction]] = []
        # Latest -> earliest so a fixed predecessor can constrain its own predecessor within this single pass.
        for t in range(grid.n_frames - 1, 0, -1):
            gap = grid.calculate_gap_delta_in_hours(t)
            if gap is None or gap > self.max_gap_hours:
                continue
            for i in range(grid.grids[t].n):
                label = grid.labels(t)[i]
                allowed_labels = self.transitions.get(label)
                if not allowed_labels or label not in allowed_labels: # Only labels that may persist (label in its own predecessor set) can be projected backwards
                    continue
                p = grid.prev_cell(t, i)
                if p < 0: # -1 = no match, first frame, or breakpoint
                    continue
                prev_label = grid.labels(t - 1)[p]
                if prev_label in allowed_labels or prev_label in self.ignore or prev_label == label:
                    continue
                grid.set_label(t - 1, p, label)
                reason=f"invalid predecessor of {label!r} {gap:.1f}h later"
                out.append((t - 1, Correction(cell_index=p, old_label=prev_label, new_label=label, reason=reason)))
        return out

    def __repr__(self) -> str:
        return f"TemporalTransitionRule(max_gap_hours={self.max_gap_hours})"
