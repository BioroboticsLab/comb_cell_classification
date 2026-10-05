"""Entry point: parse CLI args, build the viewer, optionally pre-load, run."""

from __future__ import annotations

import argparse
import logging
import sys

def _repair_qt_plugin_path() -> None:
    """Undo opencv-python's hijack of QT_QPA_PLATFORM_PLUGIN_PATH.

    On Linux, `import cv2` from the full (non-headless) opencv-python wheel points that variable at cv2's
    vendored Qt plugins, whose libqxcb.so will not load against PyQt5's Qt, so napari dies with "no Qt
    platform plugin could be initialized". Setting the variable beforehand does not help: cv2 overwrites
    it unconditionally. So import cv2 here and, if it took the variable over, point it back at PyQt5's
    own plugins before any Qt object exists; the later `import cv2` inside napari/ccc is then a no-op and
    cannot reset it. Where cv2 leaves the variable alone (macOS, headless cv2) this does nothing.

    The full wheel can't simply be dropped for opencv-python-headless: the honeybee pipeline packages in
    external/ require it, and both wheels install into the same cv2/ directory.
    """
    import os
    from pathlib import Path

    try:
        import cv2  # noqa: F401  -- imported for the side effect being undone
    except ImportError:
        return
    current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if not current or not Path(current).resolve().is_relative_to(Path(cv2.__file__).resolve().parent):
        return
    try:
        import PyQt5
    except ImportError:
        return
    plugins = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
    if (plugins / "platforms").is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins)

def main(argv = None) -> int:
    """Entry point for ``ccc gui-tools annotation-tool_v2``."""
    # Must run before anything creates a Qt object; see the docstring. Deleting it breaks the tool on Linux.
    _repair_qt_plugin_path()
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
