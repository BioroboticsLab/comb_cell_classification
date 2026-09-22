from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import cv2

from src.core.annotations import Annotation

if TYPE_CHECKING:
    from src.core.config.config import BeeCombConfig

logger = logging.getLogger(__name__)

def apply_clahe(img: npt.NDArray[np.uint8], clip_limit: float=2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> npt.NDArray[np.uint8]:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image."""
    clahe = cv2.createCLAHE(clip_limit, tile_grid_size)

    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return clahe.apply(img).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR).astype(np.uint8)

def preprocess_image(img: npt.NDArray[np.uint8],
                     ann: Annotation,
                     roi_size: int = 0,
                     outer_layer:int = 0,
                     border: int = 12,
                     radius_extra: int = 20,
                     enable_clahe: bool = True,
                     resize_to_roi: bool = False,
                     fixed_radius: int = -1
                     ) -> npt.NDArray[np.uint8]:
    """Crop the image around the annotated cell (plus optional neighbour-cell context) and optionally resize, pad, and CLAHE the result."""
    x, y, r = ann.center_x, ann.center_y, ann.radius
    if fixed_radius > 0:
        r = fixed_radius
    diameter = 2 * r
    z = (outer_layer * (2* diameter) + diameter + radius_extra) // 2

    if resize_to_roi:
        scale_factor = (roi_size - 2*border) / (2 * z)
        crop_border = int(border / scale_factor)
    else:
        crop_border = border

    x1 = max(0, x - z - crop_border)
    y1 = max(0, y - z - crop_border)
    x2 = min(img.shape[1], x + z + crop_border)
    y2 = min(img.shape[0], y + z + crop_border)

    cropped_img = img[y1:y2, x1:x2]

    if cropped_img.shape[0] < 2 * (z + crop_border) or cropped_img.shape[1] < 2 * (z + crop_border):
        pad_top = max(0, (z + crop_border) - (y - y1))
        pad_bottom = max(0, (z + crop_border) - (img.shape[0] - y2))
        pad_left = max(0, (z + crop_border) - (x - x1))
        pad_right = max(0, (z + crop_border) - (img.shape[1] - x2))
        cropped_img = cv2.copyMakeBorder(cropped_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    if resize_to_roi:
        cropped_img = cv2.resize(cropped_img, (roi_size, roi_size))
    if enable_clahe:
        cropped_img = apply_clahe(cropped_img.astype(np.uint8))

    return cropped_img


def crop_cell(img: npt.NDArray[np.uint8], ann: Annotation, cfg: BeeCombConfig) -> npt.NDArray[np.uint8]:
    """Cut one cell's crop at native resolution with CLAHE, exactly as the training dataset producer stores it; training and inference both call this, then resize to the model input (``resize_to_input``)."""
    return preprocess_image(
        img,
        ann,
        outer_layer=cfg.dataset.cell_outer_layer,
        border=12,
        radius_extra=20,
        enable_clahe=cfg.dataset.apply_clahe,
        fixed_radius=cfg.dataset.fixed_radius,
    )
