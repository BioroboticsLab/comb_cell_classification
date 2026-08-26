"""Post-processing dock for :class:`AnnotationApp` (a mixin)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from src.postprocess.grid import DEFAULT_RADIUS_SCALE, connected_components, estimate_neighbor_dist
from src.postprocess.rules import (
    DEFAULT_MAX_GAP_HOURS as MAX_GAP_HOURS,
    DEFAULT_MAX_ISLAND_SIZE,
    DEFAULT_MIN_BORDER_FRAC,
)
from ..core.postprocess import cluster_majority, temporal_majority_fixes, transition_fixes
from ._prefs import load_prefs, save_prefs


class PostprocessMixin:
    """The Post-process dock: selection helpers + the two Apply buttons."""

    def _build_selection_controls(self) -> None:
        """A standalone Selection dock, tabbed beside Annotations on the right."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self._sel_toggle_btn = QPushButton("Select all (this frame)")
        self._sel_toggle_btn.setCheckable(True)
        self._sel_toggle_btn.toggled.connect(self._on_select_toggle)
        layout.addWidget(self._sel_toggle_btn, alignment=Qt.AlignLeft)

        self._sel_island_btn = QPushButton("Select Island from selection")
        self._sel_island_btn.clicked.connect(self._select_island_from_selection)
        layout.addWidget(self._sel_island_btn, alignment=Qt.AlignLeft)

        gap_tip = (
            "Add unselected cells enclosed by the selection. The slider caps the "
            "size of an enclosed gap/island that gets filled — e.g. 4 fills holes "
            "of up to 4 cells; larger regions connected to the outside are left."
        )
        self._gap_thresh_label = QLabel("Max enclosed gap size: 4")
        self._gap_thresh_label.setToolTip(gap_tip)
        layout.addWidget(self._gap_thresh_label)
        self._gap_thresh_slider = QSlider(Qt.Horizontal)
        self._gap_thresh_slider.setRange(1, 20)
        self._gap_thresh_slider.setValue(4)
        self._gap_thresh_slider.setToolTip(gap_tip)
        self._gap_thresh_slider.valueChanged.connect(
            lambda v: self._gap_thresh_label.setText(f"Max enclosed gap size: {v}")
        )
        layout.addWidget(self._gap_thresh_slider)
        self._sel_closed_btn = QPushButton("Fill enclosed gaps")
        self._sel_closed_btn.setToolTip(gap_tip)
        self._sel_closed_btn.clicked.connect(self._fill_enclosed_gaps)
        layout.addWidget(self._sel_closed_btn, alignment=Qt.AlignLeft)

        layout.addStretch(1)
        self._update_selection_buttons()

        dock = self._add_dock(widget, "Selection")
        ann_dock = getattr(self, "_ann_dock", None)
        if ann_dock is not None:
            self.viewer.window._qt_window.splitDockWidget(dock, ann_dock, Qt.Vertical)

    def _build_postprocess_controls(self) -> None:
        from qtpy.QtWidgets import (
            QDoubleSpinBox,
            QLabel,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(QLabel("Max island size (cells):"))
        self.pp_island_spin = QSpinBox()
        self.pp_island_spin.setRange(1, 50)
        self.pp_island_spin.setValue(DEFAULT_MAX_ISLAND_SIZE)
        layout.addWidget(self.pp_island_spin)

        layout.addWidget(QLabel("Min border fraction:"))
        self.pp_border_spin = QDoubleSpinBox()
        self.pp_border_spin.setRange(0.0, 1.0)
        self.pp_border_spin.setSingleStep(0.05)
        self.pp_border_spin.setDecimals(2)
        self.pp_border_spin.setValue(DEFAULT_MIN_BORDER_FRAC)
        layout.addWidget(self.pp_border_spin)

        spatial_btn = QPushButton("Apply Spatial Rule to Selection")
        spatial_btn.clicked.connect(self._apply_spatial_rule_to_selection)
        layout.addWidget(spatial_btn)

        temporal_btn = QPushButton("Apply Temporal Rule to Selection")
        temporal_btn.clicked.connect(self._apply_temporal_rule_to_selection)
        layout.addWidget(temporal_btn)

        layout.addStretch(1)
        dock = self._add_dock(widget, "Post-process")
        ann_dock = getattr(self, "_ann_dock", None)
        if ann_dock is not None:
            self.viewer.window._qt_window.tabifyDockWidget(ann_dock, dock)
            ann_dock.raise_()

    def _frame_rows(self, frame: int) -> np.ndarray:
        """Layer row indices belonging to ``frame`` (ascending)."""
        data = np.asarray(self.ann_layer.data)
        return np.where(np.round(data[:, 0]).astype(int) == frame)[0]

    def _on_select_toggle(self, checked: bool) -> None:
        """One button toggles between select-all (this frame) and select-none."""
        self._sel_toggle_btn.setText("Select none" if checked else "Select all (this frame)")
        if self.ann_layer is None or len(self.ann_layer.data) == 0:
            return
        if checked:
            rows = self._frame_rows(self._current_index())
            self.ann_layer.selected_data = set(rows.tolist())
        else:
            self.ann_layer.selected_data = set()
        self.ann_layer.refresh(force=True)

    def _update_selection_buttons(self) -> None:
        """Grey out the derived-selection buttons when nothing is selected."""
        has_sel = self.ann_layer is not None and bool(self.ann_layer.selected_data)
        for name in ("_sel_island_btn", "_sel_closed_btn"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(has_sel)

    def _current_frame_graph(self) -> Optional[tuple[np.ndarray, list[str], dict[int, list[int]]]]:
        """``(rows, labels, neighbors)`` for the current frame, or None if too few cells."""
        if self.ann_layer is None or len(self.ann_layer.data) == 0:
            return None
        rows = self._frame_rows(self._current_index())
        if len(rows) < 2:
            return None
        feats = self.ann_layer.features
        centers = np.asarray(self.ann_layer.data)[rows, 1:3]
        labels = [str(feats["label"].iloc[r]) for r in rows]
        nb = self._frame_neighbor_map(centers)
        return rows, labels, nb

    def _select_island_from_selection(self) -> None:
        """Grow the selection to the full same-label island(s) it touches."""
        if self.ann_layer is None or not self.ann_layer.selected_data:
            return
        graph = self._current_frame_graph()
        if graph is None:
            self.log("Not enough cells on this frame.")
            return
        rows, labels, nb = graph
        selected = set(self.ann_layer.selected_data)
        seeds = {i for i, r in enumerate(rows) if int(r) in selected}
        # Keep every same-label component that a selected cell sits in.
        seen = {
            i
            for comp in connected_components(len(rows), lambda i: nb.get(i, ()), labels.__getitem__)
            if seeds.intersection(comp)
            for i in comp
        }
        self.ann_layer.selected_data = {int(rows[i]) for i in seen}
        self.ann_layer.refresh(force=True)
        self.log(f"Selected island(s): {len(seen)} cell(s).")

    def _fill_enclosed_gaps(self) -> None:
        """Add unselected islands enclosed by the selection, up to the slider size."""
        if self.ann_layer is None or not self.ann_layer.selected_data:
            return
        graph = self._current_frame_graph()
        if graph is None:
            self.log("Not enough cells on this frame.")
            return
        rows, _labels, nb = graph
        selected = set(self.ann_layer.selected_data)
        sel_local = {i for i, r in enumerate(rows) if int(r) in selected}
        threshold = self._gap_thresh_slider.value()

        added: set[int] = set()
        # Components of the unselected cells: constant key, so only sel_local splits them.
        gaps = connected_components(
            len(rows),
            lambda i: nb.get(i, ()),
            lambda i: None if i in sel_local else "gap",
        )
        for comp in gaps:
            if len(comp) > threshold:
                continue
            if any(j in sel_local for i in comp for j in nb.get(i, ())):
                added.update(int(rows[i]) for i in comp)
        if not added:
            self.log(f"No enclosed gaps ≤ {threshold} cells to add.")
            return
        self.ann_layer.selected_data = selected | added
        self.ann_layer.refresh(force=True)
        self.log(f"Added {len(added)} enclosed cell(s) to the selection.")

    def _apply_spatial_rule_to_selection(self) -> None:
        if self.ann_layer is None or not self.ann_layer.selected_data:
            self.log("No annotation points selected.")
            return
        t = self._current_index()
        rows = self._frame_rows(t)
        if len(rows) < 2:
            self.log("Not enough cells on this frame to correct.")
            return
        feats = self.ann_layer.features
        centers = np.asarray(self.ann_layer.data)[rows, 1:3]
        labels = [str(feats["label"].iloc[r]) for r in rows]
        nb = self._frame_neighbor_map(centers)
        fixes = cluster_majority(
            labels,
            nb,
            max_island_size=int(self.pp_island_spin.value()),
            min_border_frac=float(self.pp_border_spin.value()),
        )
        selected = set(self.ann_layer.selected_data)
        applied = self._write_fixes({int(rows[i]): new for i, new in fixes.items()}, selected)
        self.log(f"Spatial rule: corrected {applied} selected cell(s).")

    def _apply_temporal_rule_to_selection(self) -> None:
        if self.ann_layer is None or not self.ann_layer.selected_data:
            self.log("No annotation points selected.")
            return
        t = self._current_index()
        if t + 1 >= len(self.entries):
            self.log("Temporal rule: no later frame to correct from.")
            return
        if (t + 1) in self.breakpoints:
            self.log("Temporal rule: a breakpoint separates this frame from the next.")
            return
        gap = self._frame_gap_hours(t, t + 1)
        if gap is None:
            self.log("Temporal rule: cannot read the capture-time gap between frames.")
            return

        data = np.asarray(self.ann_layer.data)
        feats = self.ann_layer.features
        rows_t = self._frame_rows(t)
        rows_n = self._frame_rows(t + 1)
        if len(rows_t) < 2 or len(rows_n) == 0:
            self.log("Temporal rule: not enough cells to match across frames.")
            return
        # Majority vote needs the previous frame too; unavailable at the first
        # frame of a sequence/block, where only the transition stage runs.
        has_prev = t > 0 and t not in self.breakpoints
        rows_p = self._frame_rows(t - 1) if has_prev else np.array([], dtype=int)
        frames_used = ([t - 1] if len(rows_p) else []) + [t, t + 1]
        if not self._confirm_unstabilized_temporal(frames_used):
            self.log("Temporal rule: cancelled (no stabilization data).")
            return
        centers_t = data[rows_t, 1:3]
        centers_n = data[rows_n, 1:3]
        labels_t = [str(feats["label"].iloc[r]) for r in rows_t]
        labels_n = [str(feats["label"].iloc[r]) for r in rows_n]
        radius = DEFAULT_RADIUS_SCALE * estimate_neighbor_dist(centers_t)
        maj: dict[int, str] = {}
        if len(rows_p):
            labels_p = [str(feats["label"].iloc[r]) for r in rows_p]
            maj = temporal_majority_fixes(
                labels_p, data[rows_p, 1:3], labels_t, centers_t, labels_n, centers_n, radius,
            )
            # As in the CLI pipeline: the transition stage runs on the voted labels.
            for i, new in maj.items():
                labels_t[i] = new
        fixes = transition_fixes(
            labels_t, centers_t, labels_n, centers_n, gap, radius,
            max_gap_hours=MAX_GAP_HOURS,
        )
        fixes = {**maj, **fixes}
        selected = set(self.ann_layer.selected_data)
        applied = self._write_fixes({int(rows_t[i]): new for i, new in fixes.items()}, selected)
        self.log(f"Temporal rule: corrected {applied} selected cell(s) from the neighboring frame(s).")

    def _confirm_unstabilized_temporal(self, frames: list[int]) -> bool:
        """Ask before matching cells across frames whose coordinates are not stabilized. The popup can be switched off for good."""
        # Same source as the annotation layer's coordinates (_annotation_layer_arrays):
        # empty pins there means the centers being compared are the raw ones.
        unstabilized = [f for f in frames if not self._effective_pins(f)]
        if not unstabilized or not load_prefs().get("warn_temporal_unstabilized", True):
            return True

        from qtpy.QtWidgets import QCheckBox, QMessageBox

        reason = (
            "Warping is toggled off (w), so the annotations sit at their raw coordinates."
            if not getattr(self, "warp_enabled", True)
            else "No corner pins on frame(s) " + ", ".join(str(f) for f in unstabilized) + "."
        )
        box = QMessageBox(
            QMessageBox.Warning,
            "No stabilization data",
            f"{reason}\n\nThe temporal rule matches cells between two frames by "
            "position, so without stabilization it can carry labels over to the "
            "wrong cells wherever the comb shifted between the frames.\n\n"
            "Apply it anyway?",
            QMessageBox.Yes | QMessageBox.No,
            getattr(self, "widget", None),
        )
        box.setDefaultButton(QMessageBox.No)
        never_again = QCheckBox("Do not warn about this again")
        box.setCheckBox(never_again)
        if box.exec_() != QMessageBox.Yes:
            return False
        # Only honoured on Yes: remembering a No would silently apply the rule on
        # every later click, the opposite of what the checkbox promises.
        if never_again.isChecked():
            save_prefs(warn_temporal_unstabilized=False)
        return True

    def _write_fixes(self, row_fixes: dict[int, str], selected: set[int]) -> int:
        """Apply ``{layer_row: new_label}`` to selected rows; recolor + mark dirty."""
        feats = self.ann_layer.features
        applied = 0
        for row, new in row_fixes.items():
            if row not in selected:
                continue
            feats.loc[row, "label"] = new
            applied += 1
        if applied:
            self._refresh_annotation_colors()
            self._ann_dirty = True
        return applied

    def _frame_gap_hours(self, a: int, b: int) -> Optional[float]:
        """Capture-time gap in hours between frames ``a`` and ``b`` (None if unparseable)."""
        try:
            ta = datetime.strptime(self.entries[a].timestamp, "%Y%m%dT%H%M%S")
            tb = datetime.strptime(self.entries[b].timestamp, "%Y%m%dT%H%M%S")
        except (ValueError, AttributeError, IndexError):
            return None
        return abs((tb - ta).total_seconds()) / 3600.0
