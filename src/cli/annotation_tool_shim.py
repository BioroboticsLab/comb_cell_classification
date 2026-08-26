"""Run the honeybee pipeline's ``annotation-tool`` (v1) straight out of the external checkout."""

from __future__ import annotations

import sys
import traceback
from typing import Optional

from src.core.annotations.annotations import Annotation, AnnotationDoc
from src.core.utils import PROJECT_ROOT

ANNOTATION_TOOL = PROJECT_ROOT / "external/honeybee_cell_segmentation_pipeline/tools/annotation_tool"
LABEL_CLASSES = ANNOTATION_TOOL / "data" / "label_classes_extended_version.json"


def tool_argv(args: list[str]) -> list[str]:
    """Fill in ``--config`` with the checkout's label classes unless one was given."""
    if any(a == "--config" or a.startswith("--config=") for a in args):
        return args
    return ["--config", str(LABEL_CLASSES), *args]


def install_wrapped_json() -> None:
    """Read and write this project's annotation JSONs, keeping the non-cell keys."""
    import numpy as np
    from skimage.io import imread

    import data_loader as upstream

    def load_data(self, image_idx: int):
        image_path = self.image_paths[image_idx]
        self.logger.info(f"loading image {image_path}")
        label_path = self.label_paths.get(image_path.stem)
        if label_path and label_path.exists():
            cells = AnnotationDoc.load(label_path).annotations
        else:
            cells = []
            self.logger.warning(f"No labels json found for {image_path.stem}")

        labels = [c.label for c in cells]
        self._validate_labels(labels, image_path.stem)
        return (
            imread(str(image_path)),
            image_path.stem,
            [c.id for c in cells],
            np.array([[c.center_y, c.center_x] for c in cells]),
            np.array([c.radius * 2 for c in cells], dtype=float),
            labels,
        )

    def export_annotated_cells(self, image_idx, ids, points, point_diameters, labels) -> None:
        path = self.data_dir / f"{self.image_paths[image_idx].stem}.json"
        # load and replace only annotation data (to preserve corner pin and breakpoints data)
        doc = AnnotationDoc.load(path) if path.exists() else AnnotationDoc(path=path)
        doc.annotations = [
            Annotation(id=id, center_x=x, center_y=y, radius=d / 2, label=label)
            for id, (y, x), d, label in zip(ids, points, point_diameters, labels)
        ]
        doc.save()
        self.logger.info(f"Exporting {len(doc)} cells to {path}")
        self.label_paths[path.stem] = path

    upstream.DataLoader.load_data = load_data
    upstream.DataLoader.export_annotated_cells = export_annotated_cells


def main(argv: Optional[list[str]] = None) -> None:
    src = ANNOTATION_TOOL / "src"
    if not src.is_dir():
        sys.exit(f"annotation-tool: {src} is missing. Run ./setup.sh to clone the honeybee pipeline.")

    sys.argv = [sys.argv[0], *tool_argv(list(sys.argv[1:] if argv is None else argv))]
    sys.path.append(str(src))

    from annotation_controller import main as annotation_tool_main

    install_wrapped_json()
    try:
        annotation_tool_main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
