"""Per-image annotation JSONs: cells, corner pins and the breakpoint flag."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

UNLABELED = "unlabeled"


@dataclass
class Annotation:
    """One annotated cell, as stored in the per-image JSON."""
    id: str
    center_x: float
    center_y: float
    radius: float
    label: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "center_x": int(round(self.center_x)),
            "center_y": int(round(self.center_y)),
            "radius": int(round(self.radius)),
            "label": self.label,
        }


def pins_from_corner_pin(corner_pin: Optional[dict]) -> list[dict]:
    """Convert a per-image ``corner_pin`` data into display pins."""
    pins: list[dict] = []
    for half in ((corner_pin or {}).get("halves") or {}).values():
        frame = half.get("frame") or {}
        homography_matrix = frame.get("homography_3x3")
        if not half.get("has_corner_pin") or not homography_matrix:
            continue
        crop = half.get("source_crop") or {}
        rect = (
            int(crop.get("x", 0)), int(crop.get("y", 0)),
            int(crop.get("w", 0)), int(crop.get("h", 0)),
        )
        pins.append({"crop": rect, "H": np.array(homography_matrix, np.float32)})
    return pins


def _pin_for_point(x: float, y: float, pins: list[dict]) -> Optional[dict]:
    """The pin whose crop rect (halves) contains ``(x, y)``, or a full-frame pin."""
    return_value = None
    for pin in pins:
        crop_x, crop_y, crop_width, crop_height = pin["crop"]
        if crop_width <= 0 or crop_height <= 0:
            return_value = pin
            continue
        if crop_x <= x < crop_x + crop_width and crop_y <= y < crop_y + crop_height:
            return pin
    return return_value

def _apply_homography_matrix_to_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply a 3x3 homography to one (x, y) point (homogeneous divide)."""
    vec = matrix @ np.array([x, y, 1.0], dtype=np.float64)
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])

def shift_point(x: float, y: float, pins: list[dict], to_source: bool = False) -> tuple[float, float]:
    """Project a point between source and warped-display coordinates."""
    pin = _pin_for_point(x, y, pins)
    if pin is None:
        return x, y
    cx, cy = pin["crop"][0], pin["crop"][1]
    h_mat = np.asarray(pin["H"], dtype=np.float64)
    if not to_source:
        h_mat = np.linalg.inv(h_mat)
    lx, ly = _apply_homography_matrix_to_point(h_mat, x - cx, y - cy)
    return lx + cx, ly + cy
@dataclass
class AnnotationDoc:
    """One per-image annotation JSON, loaded for editing and round-tripping."""
    path: Path
    annotations: list[Annotation] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnnotationDoc":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            entries = data.get("annotations", [])
            extras = {k: v for k, v in data.items() if k != "annotations"}
        else:
            entries, extras = data, {}
        annotations = [ Annotation( id=e["id"], center_x=e["center_x"], center_y=e["center_y"], radius=e["radius"], label=e["label"], ) for e in entries ]
        return cls(path=path, annotations=annotations, extras=extras)

    def __len__(self) -> int:
        return len(self.annotations)

    @property
    def corner_pin(self) -> Optional[dict]:
        return self.extras.get("corner_pin")

    @corner_pin.setter
    def corner_pin(self, value: Optional[dict]) -> None:
        if value is None:
            self.extras.pop("corner_pin", None)
        else:
            self.extras["corner_pin"] = value

    @property
    def sequence_breakpoint(self) -> bool:
        return bool(self.extras.get("sequence_breakpoint"))

    @sequence_breakpoint.setter
    def sequence_breakpoint(self, value: bool) -> None:
        self.extras["sequence_breakpoint"] = bool(value)

    def pins(self) -> list[dict]:
        """Display pins from this file's ``corner_pin`` payload (may be [])."""
        return pins_from_corner_pin(self.corner_pin)

    def payload(self) -> Union[list, dict]:
        """The JSON-ready document: wrapped dict when extras exist, else a bare list."""
        cells = [ann.to_dict() for ann in self.annotations]
        if self.extras:
            return {**self.extras, "annotations": cells}
        return cells

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Write the document back to JSON, defaulting to its source file."""
        return dump_data(path if path is not None else self.path, self.payload())


def dump_data(path: Union[str, Path], payload: Union[list, dict]) -> Path:
    """Write a JSON data with the project's formatting (UTF-8, indent=2), atomically."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def write_breakpoint(path: Union[str, Path], flag: bool) -> Path:
    """Set ``sequence_breakpoint`` in one image's JSON, creating it if missing."""
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            data = {"sequence_breakpoint": bool(flag), "annotations": data}
        else:
            data["sequence_breakpoint"] = bool(flag)
    else:
        data = {"sequence_breakpoint": bool(flag), "annotations": []}
    return dump_data(path, data)

