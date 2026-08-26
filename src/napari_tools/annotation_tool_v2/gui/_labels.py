"""Label classes for :class:`AnnotationApp` (a mixin): the colour map, the Annotations dock and applying a label to the selection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.core.annotations import UNLABELED
from src.core.utils import PROJECT_ROOT

from ._prefs import load_prefs

if TYPE_CHECKING:
    import napari

FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
]


def hex_to_rgba(color: str) -> tuple[float, float, float, float]:
    """Parse ``#RRGGBB`` / ``#RRGGBBAA`` into an RGBA tuple in 0..1."""
    c = color.lstrip("#")
    if len(c) == 6:
        c += "ff"
    r, g, b, a = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6))
    return (r, g, b, a)


class LabelsMixin:
    """The label -> colour map, the Annotations dock (combo + legend) and labelling."""

    def _init_label_state(self) -> None:
        self.label_map: dict[str, str] = {UNLABELED: "#00000000"}
        self._warned_unclassified = False
        self._build_annotation_controls()

    def _load_label_map(self, folder: Path) -> None:
        """The configured label_classes.json if set, else one next to the data, else colours from data."""
        import json

        label_map = {UNLABELED: "#00000000"}
        # The label_classes_file pref wins: it is explicit configuration (resolved
        # against the project root when it is not absolute). Only without it do we
        # fall back to a file sitting in the image folder itself or beside it.
        candidates = []
        configured = load_prefs().get("label_classes_file")
        if configured:
            path = Path(configured)
            candidates.append(path if path.is_absolute() else PROJECT_ROOT / path)
        candidates += [folder / "label_classes.json", folder.parent / "label_classes.json"]
        config = next((c for c in candidates if c.exists()), None)
        if config is not None:
            try:
                with config.open(encoding="utf-8") as fh:
                    entries = json.load(fh)
                label_map.update({e["name"]: e["color"] for e in entries})
                self.log(f"Loaded {len(entries)} label classes from {config}.")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.log(f"Could not read {config}: {exc}")
        self.label_map = label_map
        self._sync_label_widgets()

    def _ensure_labels_known(self) -> None:
        """Extend the label map with colours for labels found only in the data."""
        seen: set[str] = set()
        for doc in self.docs:
            if doc is not None:
                seen.update(ann.label for ann in doc.annotations)
        missing = sorted(seen - set(self.label_map))
        for i, label in enumerate(missing):
            self.label_map[label] = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        if missing:
            self._sync_label_widgets()

    def _face_colors(self, labels: list[str]) -> np.ndarray:
        rgba = {lbl: hex_to_rgba(color) for lbl, color in self.label_map.items()}
        default = hex_to_rgba(FALLBACK_COLORS[0])
        return np.array([rgba.get(lbl, default) for lbl in labels], dtype=float)

    def _build_annotation_controls(self) -> None:
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(QLabel("Label for selected points:"))
        self.label_combo = QComboBox()
        layout.addWidget(self.label_combo)

        apply_btn = QPushButton("Apply label to selection (j)")
        apply_btn.clicked.connect(lambda: self._apply_label_to_selected())
        layout.addWidget(apply_btn)

        self.lock_checkbox = QCheckBox("Lock labeled points (Ctrl+L)")
        self.lock_checkbox.toggled.connect(self._on_lock_toggled)
        layout.addWidget(self.lock_checkbox)

        save_btn = QPushButton("Save annotations now")
        save_btn.clicked.connect(lambda: self._save_annotations(reason="manual save"))
        layout.addWidget(save_btn)

        classify_btn = QPushButton("Classify sequence…")
        classify_btn.clicked.connect(lambda: self._open_classify_dialog())
        if sys.platform != "linux":
            classify_btn.setEnabled(False)
            classify_btn.setToolTip("Classification runs on Linux only")
        layout.addWidget(classify_btn)

        layout.addWidget(QLabel("Legend:"))
        self._legend_box = QVBoxLayout()
        legend_host = QWidget()
        legend_host.setLayout(self._legend_box)
        layout.addWidget(legend_host)
        layout.addStretch(1)
        self._legend_rows: list[QWidget] = []

        self._ann_dock = self._add_dock(widget, "Annotations")
        self._sync_label_widgets()

        @self.viewer.bind_key("j", overwrite=True)
        def _apply_key(_v: napari.Viewer) -> None:
            self._apply_label_to_selected()

    def _sync_label_widgets(self) -> None:
        """Refill the label combo box and the colour legend from the label map."""
        from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

        if not hasattr(self, "label_combo"):
            return
        current = self.label_combo.currentText()
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        self.label_combo.addItems(list(self.label_map))
        idx = self.label_combo.findText(current)
        if idx >= 0:
            self.label_combo.setCurrentIndex(idx)
        self.label_combo.blockSignals(False)

        for row in self._legend_rows:
            self._legend_box.removeWidget(row)
            row.deleteLater()
        self._legend_rows = []
        for label, color in self.label_map.items():
            row = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            box = QLabel()
            box.setFixedSize(14, 14)
            box.setStyleSheet(f"background-color: {color[:7]}; border: 1px solid #888;")
            row_layout.addWidget(box)
            row_layout.addWidget(QLabel(label))
            row_layout.addStretch(1)
            row.setLayout(row_layout)
            self._legend_box.addWidget(row)
            self._legend_rows.append(row)

        # Legend rows change the panel's natural height — re-cap the dock.
        dock = getattr(self, "_ann_dock", None)
        if dock is not None:
            self._refresh_dock_height(dock)

    def _current_menu_label(self) -> str:
        text = self.label_combo.currentText() if hasattr(self, "label_combo") else ""
        return text or UNLABELED

    def _apply_label_to_selected(self) -> None:
        if self.ann_layer is None or not self.ann_layer.selected_data:
            self.log("No annotation points selected.")
            return
        label = self._current_menu_label()
        indices = list(self.ann_layer.selected_data)
        self.ann_layer.features.loc[indices, "label"] = label
        self._refresh_annotation_colors()
        self._ann_dirty = True
        self.log(f"Labeled {len(indices)} cells as '{label}'.")

    def _maybe_warn_unclassified(self) -> None:
        """Pop up once per app run if unclassified cells remain."""
        if self._warned_unclassified:
            return
        count = sum(1 for doc in self.docs if doc is not None for ann in doc.annotations if ann.label == UNLABELED)
        if not count:
            return
        self._warned_unclassified = True
        from qtpy.QtWidgets import QMessageBox

        QMessageBox.information(
            getattr(self, "widget", None),
            "Unclassified annotations",
            f"There are still {count} cells labeled '{UNLABELED}' in this folder's "
            "annotations.\n\nSelect points and apply a label (j) to classify them. "
            "This notice shows once per session.",
        )
