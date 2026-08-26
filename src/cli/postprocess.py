"""``ccc postprocess`` command — spatial and temporal correction of prediction JSONs, with backup/revert support."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.postprocess.api import correct_spatial, correct_temporal, default_rules, revert
from src.postprocess.rules import (
    DEFAULT_MAX_GAP_HOURS,
    DEFAULT_MAX_ISLAND_SIZE,
    DEFAULT_MIN_BORDER_FRAC,
    list_rules,
)
from src.postprocess.temporal_correction import SequenceCorrector, default_temporal_pipeline

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ccc postprocess", description="ccc post-processing (spatial correction + breakpoint tool).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Log would-be corrections without writing any files (correct-spatial only).")
    parser.add_argument("--revert", type=Path, metavar="PATH", help="Restore a file/dir from its bkp/ folder and exit.")
    sub = parser.add_subparsers(dest="command")

    p_correct = sub.add_parser("correct-spatial", help="Spatially correct a prediction JSON file or directory, in place.")
    p_correct.add_argument("input_path", type=Path, help="Prediction JSON file or directory of JSONs.")
    p_correct.add_argument("-b", "--backup", action="store_true", help="Save originals to a bkp/ folder next to the inputs first.")
    p_correct.add_argument("--max-island-size", type=int, default=DEFAULT_MAX_ISLAND_SIZE, help="Largest same-label island (cell count) absorbed into its surrounding region.")
    p_correct.add_argument("--min-border-frac", type=float, default=DEFAULT_MIN_BORDER_FRAC, help="Fraction of an island's border that must share one label for it to be absorbed.")
    p_correct.add_argument("--max-passes", type=int, default=1, help="Iterate the pipeline up to N times (stops at a fixpoint).")
    # SUPPRESS: don't clobber a --dry-run given before the subcommand.
    p_correct.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS, help="Log would-be corrections without writing any files.")

    p_temporal = sub.add_parser("correct-temporal", help="Temporally correct a directory of prediction JSONs (run after correct-spatial and after marking breakpoints).")
    p_temporal.add_argument("input_path", type=Path, help="Directory of frame JSONs (grouped per camera).")
    p_temporal.add_argument("-b", "--backup", action="store_true", help="Save originals to a bkp/ folder next to the inputs first.")
    p_temporal.add_argument("--cam", default=None, help="Restrict to one camera id, e.g. cam-0 or 0 (default: all cameras in the dir).")
    p_temporal.add_argument("--match-radius", type=float, default=None, help="Pixel radius for matching a cell across frames (default: auto from cell spacing).")
    p_temporal.add_argument("--min-agree", type=int, default=2, help="Temporal neighbors (prev/next frame) that must agree to flip a cell (default: 2 = both).")
    p_temporal.add_argument("--max-gap-hours", type=float, default=DEFAULT_MAX_GAP_HOURS, help="Max hours between frames for the biology transition constraint to apply (default: 36 = 1.5 days).")
    p_temporal.add_argument("--no-transitions", action="store_true", help="Disable the directional brood/honey transition constraint (keep only the majority vote).")
    p_temporal.add_argument("--max-passes", type=int, default=1, help="Iterate the pipeline up to N times (stops at a fixpoint); >1 propagates labels along time.")
    p_temporal.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS, help="Log would-be corrections without writing any files.")

    sub.add_parser("list-rules", help="List available correction rules.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None and args.revert is None:
        parser.error("a subcommand or --revert is required")

    from src.core.logs.setup import setup_run_logging

    run_dir = setup_run_logging("postprocess", level=getattr(logging, args.log_level))
    from src.core.logs.paths import run_log_file
    logger.info("Logging this run to %s", run_log_file(run_dir))

    try:
        if args.revert is not None:
            revert(args.revert)
        elif args.command == "correct-spatial":
            correct_spatial(
                args.input_path,
                default_rules(args.max_island_size, args.min_border_frac),
                max_passes=args.max_passes,
                dry_run=args.dry_run,
                backup=args.backup,
            )
        elif args.command == "correct-temporal":
            correct_temporal(
                args.input_path,
                SequenceCorrector(
                    default_temporal_pipeline(min_agree=args.min_agree),
                    max_passes=args.max_passes,
                    match_radius=args.match_radius,
                    transitions={} if args.no_transitions else None,
                    max_gap_hours=args.max_gap_hours,
                ),
                cam=args.cam,
                dry_run=args.dry_run,
                backup=args.backup,
            )
        elif args.command == "list-rules":
            print("\n".join(list_rules()))
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
