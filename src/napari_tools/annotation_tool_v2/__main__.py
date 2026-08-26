"""Entry point: parse CLI args, build the viewer, optionally pre-load, run."""

from __future__ import annotations

import argparse
import logging
import sys

def main(argv = None) -> int:
    """Entry point for ``ccc gui-tools annotation-tool_v2``."""
    parser = argparse.ArgumentParser(prog="annotation-tool_v2", description="napari tool for annotating a time-ordered image sequence.")
    parser.add_argument("folder", help="Flat folder of images to annotate (e.g. .../data/annotated/cam-1).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import napari
    from .gui.app import AnnotationApp

    app = AnnotationApp()
    app.load_folder(args.folder)
    napari.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())
