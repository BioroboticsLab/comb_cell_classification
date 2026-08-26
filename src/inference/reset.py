"""Reset annotation labels back to ``unlabeled`` in place, without needing a backup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from src.core.annotations import AnnotationDoc
from src.core.annotations.discover import pair_images_with_annotations, resolve_image_dirs

logger = logging.getLogger(__name__)

# annotation tool treats this label as "no class assigned" (see annotation_tool.annotation_controller's "unlabeled" -> transparent colour).
UNLABELED_LABEL = "unlabeled"

def reset_labels(
    images_dir: Union[str, Path],
    label: str = UNLABELED_LABEL,
    image_suffixes: str = ".jpeg, .png",
    annotation_suffix: str = ".json",
    backup_dirname: str = "unlabeled_jsons",
) -> None:
    """Set every annotation label under *images_dir* (a directory or glob) to *label*, in place."""
    dirs = resolve_image_dirs(images_dir)
    total_files = total_anns = 0

    for directory in dirs:
        backup_dir = directory / backup_dirname
        pairs = pair_images_with_annotations(directory, image_suffixes, annotation_suffix)
        # Never touch the pristine backup copies themselves.
        pairs = [p for p in pairs if backup_dir not in p[1].parents]

        for _, annotation_path in pairs:
            loader = AnnotationDoc.load(annotation_path)
            if len(loader) == 0:
                continue
            for ann in loader.annotations:
                ann.label = label
            loader.save()
            total_files += 1
            total_anns += len(loader)

    logger.info("Reset %d annotation(s) across %d file(s) in %d director(y/ies) to %r.", total_anns, total_files, len(dirs), label)
