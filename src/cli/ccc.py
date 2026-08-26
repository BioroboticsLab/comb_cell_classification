"""``ccc`` console-script entry point that dispatches to the project's commands, forwarding the remaining argv verbatim."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Optional

_COMMANDS = {
    "training": "Launch/manage a training run in the background. (Linux only)",
    "classify": "Run a trained model and write predicted labels into annotation JSONs (or --reset labels back to unlabeled). (Linux only)",
    "postprocess": "Spatial and temporal correction of prediction JSONs.",
}

def _usage() -> None:
    print("usage: ccc <command> [options]\n")
    print("Commands:")
    width = max(len(c) for c in _COMMANDS)
    for cmd, desc in _COMMANDS.items():
        print(f"  {cmd:<{width}}  {desc}")
    print("\nRun `ccc <command> --help` for command-specific options.")
    # The napari tools are their own console scripts, not ccc subcommands.
    print("The napari annotation tool is its own command: `annotation-tool_v2 --help`.")

def _resolve(cmd: str) -> Callable[[Optional[list[str]]], None]:
    """Import and return the handler ``main(argv)`` for ``cmd``, lazily so `ccc --help` never pays heavy import costs (TensorFlow, wandb)."""
    # Training and model inference need tensorflow[and-cuda] + an NVIDIA GPU, which pyproject installs on Linux only!
    if cmd in ("training", "classify") and sys.platform != "linux":
        sys.exit(f"ccc {cmd} runs on Linux only (needs tensorflow[and-cuda] and a NVIDIA GPU).")
    if cmd == "training":
        from src.cli.train import main
    elif cmd == "classify":
        from src.cli.classify import main
    elif cmd == "postprocess":
        from src.cli.postprocess import main
    else:
        raise KeyError(cmd)
    return main


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        return
    
    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        _usage()
        sys.exit(f"\nccc: unknown command {cmd!r}")

    _resolve(cmd)(rest)


if __name__ == "__main__":
    main()
