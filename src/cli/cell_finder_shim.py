"""Run upstream's ``cell-finder`` with the fixes and options this project needs because of dependency differences.

Three changes:

1. ``torch.load`` defaults to ``map_location="cpu"``. The comb segmentation checkpoint was saved on CUDA and this project installs CPU-only torch on
   Linux on purpose (TF[and-cuda] owns the CUDA stack), so bare ``torch.load`` upstream uses raises and ``graph_building`` dies.
2. Each image's JSON is written the moment that image is finished, instead of every JSON at the very end. A cancelled or crashed run then keeps the work it already did.
3. ``--skip-existing`` drops images that already have a JSON, so an interrupted run can be resumed.

Usage — upstream's own arguments, plus the flag above::
    python -m src.cli.cell_finder_shim <input_dir> [--output_path DIR] <method> [--curve_aware] [--skip-existing]
"""

from __future__ import annotations

import sys
from typing import Optional

SHIM_FLAGS = ("--skip-existing",)

_state: dict = {"finder": None, "suppress": False}


def install_cpu_map_location() -> None:
    """Make ``torch.load`` default to the CPU, so CUDA-saved checkpoints load anywhere."""
    import torch

    original = torch.load
    if getattr(original, "_ccc_cpu_default", False):
        return

    def load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original(*args, **kwargs)

    load._ccc_cpu_default = True
    torch.load = load


class WritingResults(dict):
    """Results dict that writes an image's JSON as soon as its entry is assigned.
    ``run_with_graph_building`` stores each image's finished cells back into results dict right after growing that image's graph, so
    assignment is exactly the moment the image is done."""

    def __init__(self, finder, data) -> None:
        super().__init__(data)
        self._finder = finder

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        image_path, _duration, _count, _image, matches = value
        self._finder._save_cells_to_json(matches, image_path)


def install_incremental_json() -> None:
    """Write each image's JSON when that image finishes, not all of them at the end."""
    from cell_finder import cell_finder as upstream

    if getattr(upstream.CellFinder.run, "_ccc_incremental", False):
        return

    original_run = upstream.CellFinder.run
    original_tqdm = upstream.tqdm

    def run(self, *args, **kwargs):
        # The tqdm hook needs the instance to reach _save_cells_to_json, and the returned dict carries the per-image write for the graph-building loop.
        _state["finder"] = self
        try:
            return WritingResults(self, original_run(self, *args, **kwargs))
        finally:
            _state["finder"] = None

    def writing_tqdm(iterable, **kwargs):
        for future in original_tqdm(iterable, **kwargs):
            finder = _state["finder"]
            if finder is not None and not _state["suppress"]:
                try:
                    image_path, _duration, _count, _image, matches = future.result()
                    finder._save_cells_to_json(matches, image_path)
                except Exception:
                    pass  # the real failure surfaces where run() reads the future
            yield future

    def suppressed(original):
        """Wrap a composite method whose inner run() results are only intermediate."""

        def wrapper(self, *args, **kwargs):
            _state["suppress"] = True
            try:
                return original(self, *args, **kwargs)
            finally:
                _state["suppress"] = False

        return wrapper

    run._ccc_incremental = True
    upstream.CellFinder.run = run
    upstream.CellFinder.run_with_graph_building = suppressed(upstream.CellFinder.run_with_graph_building)
    upstream.CellFinder.run_hybrid_detection = suppressed(upstream.CellFinder.run_hybrid_detection)
    upstream.tqdm = writing_tqdm


def install_skip_existing() -> None:
    """Drop images that already have a JSON next to them, so a run can be resumed."""
    from cell_finder import cell_finder as upstream

    original = upstream.CellFinder._find_images

    def find_images(self):
        images = original(self)
        kept = [p for p in images if not (self.output_path / f"{p.stem}.json").exists()]
        if len(kept) != len(images):
            print(f"--skip-existing: skipping {len(images) - len(kept)} image(s) that already have a JSON.")
        return kept

    upstream.CellFinder._find_images = find_images


def main(argv: Optional[list[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    skip_existing = "--skip-existing" in args
    # Upstream's parser would reject our own flags, so hand it only its own.
    sys.argv = [sys.argv[0], *(a for a in args if a not in SHIM_FLAGS)]

    install_cpu_map_location()
    install_incremental_json()
    if skip_existing:
        install_skip_existing()

    # Absolute import: this resolves to the third-party top-level `cell_finder` package, never to this module, despite the similar name.
    from cell_finder.cell_finder import main as cell_finder_main

    cell_finder_main()


if __name__ == "__main__":
    main()
