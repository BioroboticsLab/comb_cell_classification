"""``hbcsp`` console-script entry point that dispatches to the honeybee_cell_segmentation_pipeline's own console scripts by ``exec``-ing them."""

from __future__ import annotations

import os
import shutil
import sys
from typing import Optional

_COMMANDS = {
    "cell-finder": "Detect cells in comb images.",
    "performance-validation": "Validate cell-finder output against ground truth.",
    "mask-writer": "Convert JSON annotations to PNG segmentation masks.",
    "frame-extractor": "Extract frames from video files.",
    "annotation-tool": "napari GUI for labeling comb images.",
    "background-generator": "Generate bee-free background images (Linux only).",
}


# Run straight out of the external checkout instead of a console script: upstream's
# pyproject for these two declares no importable package, so an install ships no modules.
_CHECKOUT_SCRIPTS = {
    "mask-writer": "tools/mask_writer/src/mask_writer.py",
    "frame-extractor": "tools/frame_extractor/src/main.py",
}


def _usage() -> None:
    print("usage: hbcsp <command> [options]\n")
    print("honeybee_cell_segmentation_pipeline commands:")
    width = max(len(c) for c in _COMMANDS)
    for cmd, desc in _COMMANDS.items():
        print(f"  {cmd:<{width}}  {desc}")
    print("\nRun `hbcsp <command> --help` for command-specific options.")


def script_path(name: str) -> str:
    """Find the console script, preferring the venv bin next to this interpreter."""
    # Under `uv tool install` the tool's bin dir may not be on PATH, so look beside sys.executable first, then PATH, then let execvp resolve the bare name.
    beside = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.exists(beside):
        return beside
    return shutil.which(name) or name


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        return

    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        _usage()
        sys.exit(f"\nhbcsp: unknown command {cmd!r}")

    if cmd == "cell-finder":
        # Through our wrapper, not the console script: upstream loads the comb
        # segmentation checkpoint in a way CPU-only torch cannot (see
        # src/cli/cell_finder_shim). Same CLI, same exit code.
        os.execv(sys.executable, [sys.executable, "-m", "src.cli.cell_finder_shim", *rest])

    if cmd in _CHECKOUT_SCRIPTS:
        from src.core.utils import PROJECT_ROOT

        script = PROJECT_ROOT / "external/honeybee_cell_segmentation_pipeline" / _CHECKOUT_SCRIPTS[cmd]
        if not script.is_file():
            sys.exit(f"hbcsp {cmd}: {script} is missing. Run ./setup.sh to clone the honeybee pipeline.")
        # The script's own directory lands on sys.path, which is what its flat imports need.
        os.execv(sys.executable, [sys.executable, str(script), *rest])

    target = script_path(cmd)
    # Replace this process with the tool so its CLI/exit code/GUI are unchanged.
    os.execvp(target, [cmd, *rest])


if __name__ == "__main__":
    main()
