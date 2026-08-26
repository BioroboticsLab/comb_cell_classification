"""Temporal correction over the 3D (space + time) :class:`~src.postprocess.grid.SequenceGrid`, run after spatial correction and reusing the same rule classes tagged with an axis (``"spatial"`` = 6 in-frame neighbors, ``"temporal"`` = up to 2 neighbors along time)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from src.core.annotations import Annotation, AnnotationDoc, shift_point
from src.postprocess.grid import DEFAULT_RADIUS_SCALE, SequenceGrid
from src.postprocess.rules import (
    DEFAULT_MAX_GAP_HOURS,
    Correction,
    CorrectionRule,
    TemporalTransitionRule,
    build_rule,
    default_transitions,
)
from src.postprocess.sequence import parse_annotation_filename

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)

# A pipeline entry: which neighbor axis the rule reads, and the rule itself.
AxisRule = tuple[str, CorrectionRule]


def default_temporal_pipeline(min_agree: int = 2) -> list[AxisRule]:
    """Return the default temporal pipeline: flip a cell when its previous- and next-frame neighbors agree (``min_agree=2`` requires both)."""
    return [("temporal", build_rule("neighbor_majority", min_agree=min_agree))]


class SequenceCorrector:
    """Runs an ordered ``(axis, rule)`` pipeline over a :class:`SequenceGrid`, repeating up to ``max_passes`` times until a pass changes nothing."""

    def __init__(
        self,
        pipeline: Optional[list[AxisRule]] = None,
        max_passes: int = 1,
        match_radius: Optional[float] = None,
        radius_scale: float = DEFAULT_RADIUS_SCALE,
        neighbor_dist: Optional[float] = None,
        transitions: Optional[dict[str, set[str]]] = None,
        max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
        transition_ignore: Optional[set[str]] = None,
    ) -> None:
        self.pipeline = pipeline if pipeline is not None else default_temporal_pipeline()
        self.max_passes = max_passes
        self.match_radius = match_radius
        self.radius_scale = radius_scale
        self.neighbor_dist = neighbor_dist
        # transitions=None -> brood defaults; {} (empty) -> stage disabled.
        table = default_transitions() if transitions is None else transitions
        self.transition_rule = (
            TemporalTransitionRule(table, max_gap_hours=max_gap_hours, ignore=transition_ignore)
            if table
            else None
        )

    def correct(
        self,
        frames: list[list[Annotation]],
        breakpoints: Optional[list[bool]] = None,
        timestamps: Optional[list[datetime]] = None,
    ) -> tuple[list[list[Annotation]], list[tuple[int, Correction]]]:
        """Correct one camera's time-ordered ``frames`` (inputs untouched) and return ``(corrected_frames, log)`` with ``(frame_index, Correction)`` entries."""
        grid = SequenceGrid(
            frames,
            breakpoints=breakpoints,
            match_radius=self.match_radius,
            radius_scale=self.radius_scale,
            neighbor_dist=self.neighbor_dist,
            timestamps=timestamps,
        )
        applied: list[tuple[int, Correction]] = []

        # max_passes > 1 lets a stable label propagate frame-by-frame along the
        # time axis: each pass can fix a cell using neighbors fixed last pass.
        for pass_idx in range(self.max_passes):
            pass_changes = 0
            for axis, rule in self.pipeline:
                for t in range(grid.n_frames):
                    view = grid.frame_view(t, axis)
                    for corr in rule.apply(view):
                        # Re-check: an earlier rule/pass may already have set it.
                        if grid.labels(t)[corr.cell_index] == corr.new_label:
                            continue
                        grid.set_label(t, corr.cell_index, corr.new_label)
                        applied.append((t, corr))
                        pass_changes += 1
            logger.debug("Sequence pass %d applied %d changes", pass_idx, pass_changes)
            if pass_changes == 0:
                break

        # Directional biology stage: runs after the majority-vote pipeline and
        # only with timestamps, since its time gate needs real frame gaps.
        if self.transition_rule is not None and timestamps is not None:
            applied.extend(self.transition_rule.apply_sequence(grid))

        return grid.to_annotations(), applied


def _ordered_paths_by_cam(files: list[Path]) -> dict[str, list[Path]]:
    """Group annotation JSONs by camera, each list ordered by capture time, skipping unparseable filenames with a warning."""
    groups: dict[str, list[tuple[tuple[datetime, int], Path]]] = {}
    for p in files:
        try:
            meta = parse_annotation_filename(p.name)
        except ValueError as e:
            logger.warning("Skipping unparseable filename %s: %s", p.name, e)
            continue
        groups.setdefault(meta["cam"], []).append((meta["sort_key"], p))
    return {
        cam: [p for _, p in sorted(items, key=lambda t: (t[0], str(t[1])))]
        for cam, items in groups.items()
    }


def _stabilized(doc: AnnotationDoc) -> list[Annotation]:
    """Copies of ``doc``'s cells at stabilized (warped-display) coordinates, so cross-frame matching compares the same physical cell; returned unchanged when the file carries no corner pins."""
    pins = doc.pins()
    if not pins:
        return doc.annotations
    return [
        Annotation(a.id, *shift_point(a.center_x, a.center_y, pins), a.radius, a.label)
        for a in doc.annotations
    ]


def correct_dir(
    directory: Union[str, Path],
    corrector: Optional[SequenceCorrector] = None,
    cam: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Temporally correct every camera's frames in ``directory`` in place (breakpoints read from each frame's ``sequence_breakpoint`` flag, only changed frames rewritten, nothing written on ``dry_run``) and return the total number of label changes."""
    directory = Path(directory)
    corrector = corrector or SequenceCorrector()

    by_cam = _ordered_paths_by_cam(sorted(directory.glob("*.json")))
    if cam:
        cam = cam if cam.startswith("cam-") else f"cam-{cam}"
        by_cam = {cam: by_cam[cam]} if cam in by_cam else {}
        if not by_cam:
            logger.warning("No frames for %s in %s", cam, directory)

    total = 0
    for cam_id, paths in by_cam.items():
        loaders = [AnnotationDoc.load(p) for p in paths]
        # Matching runs on stabilized coordinates; only labels are copied back,
        # so the files keep their raw (source-space) centers.
        frames = [_stabilized(ld) for ld in loaders]
        breakpoints = [bool(ld.sequence_breakpoint) for ld in loaders]
        timestamps = [parse_annotation_filename(p.name)["timestamp"] for p in paths]
        n_blocks = 1 + sum(breakpoints[1:])  # a flagged frame (after frame 0) opens a block
        logger.info("%s: %d frame(s), %d block(s)", cam_id, len(loaders), n_blocks)
        unstabilized = [ld.path.name for ld in loaders if not ld.pins()]
        if unstabilized:
            logger.warning(
                "%s: %d frame(s) without corner pins, matched at raw coordinates (e.g. %s)",
                cam_id, len(unstabilized), unstabilized[0],
            )

        corrected, log = corrector.correct(frames, breakpoints, timestamps)

        changed_frames = {t for t, _ in log}
        for t in changed_frames:
            for ann, new in zip(loaders[t].annotations, corrected[t]):
                ann.label = new.label
            for corr in (c for ft, c in log if ft == t):
                logger.debug(
                    "%s: cell %d %s -> %s (%s)",
                    loaders[t].path.name, corr.cell_index, corr.old_label, corr.new_label, corr.reason,
                )
            if not dry_run:
                loaders[t].save()

        logger.info(
            "%s: %d correction(s) in %d frame(s)%s",
            cam_id, len(log), len(changed_frames), " (dry-run, not written)" if dry_run else "",
        )
        total += len(log)

    return total
