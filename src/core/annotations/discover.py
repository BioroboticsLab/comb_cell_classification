"""Lightweight (TensorFlow-free) image/annotation discovery shared by inference and reset."""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

UNLABELED = "unlabeled"

def is_fully_labeled(annotation_path: Path, unlabeled: str = UNLABELED) -> bool:
    """True when *annotation_path* holds cells and none of them is still unlabelled.

    Partially labelled files count as unlabelled. Skipping one would leave its remaining cells unlabelled with no way to notice.
    """
    try:
        with annotation_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return False
    entries = data.get("annotations", []) if isinstance(data, dict) else data
    if not entries:
        return False
    return all(entry.get("label", unlabeled) != unlabeled for entry in entries)

def resolve_image_dirs(pattern: Union[str, Path]) -> list[Path]:
    """Expand *pattern* (a plain directory path or a glob like ``.../cam-*``) into sorted list of existing directories, raising if nothing matches."""
    pattern = str(pattern)
    direct = Path(pattern)
    if direct.is_dir():
        return [direct]

    dirs = sorted(p for p in (Path(m) for m in glob.glob(pattern)) if p.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No directory matches {pattern!r} (no such directory and no glob matches)."
        )
    return dirs

def pair_images_with_annotations(images_dir: Path, image_suffixes: str = ".jpeg, .png", annotation_suffix: str = ".json") -> list[tuple[Path, Path]]:
    """Pair each image under *images_dir* with its same-directory sibling ``<stem>.json``, skipping (with a warning) images that have none."""
    suffixes = tuple(s.strip().lower() for s in image_suffixes.split(",") if s.strip())
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or not image_path.name.lower().endswith(suffixes):
            continue
        annotation_path = image_path.with_name(image_path.stem + annotation_suffix)
        if annotation_path.exists():
            pairs.append((image_path, annotation_path))
        else:
            logger.warning(
                "No annotation for %s (expected %s); skipping.", image_path, annotation_path
            )
    return pairs
