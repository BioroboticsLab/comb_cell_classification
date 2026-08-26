"""Hex-grid overlay for :class:`AnnotationApp` (a mixin): a raster with patch updates, computed on demand per frame."""

from __future__ import annotations

import numpy as np

from src.postprocess.grid import edge_pairs, estimate_neighbor_dist, neighbor_map, neighbor_map_from_pairs

GRID_LAYER_NAME = "grid"

GRID_WIDTH_PX = 4  # line width in full-resolution pixels

# Two lazy caches, both keyed (frame, dist), filled on first render of a frame:
#   _grid_cache        -> edge_pairs() output (pairs, lengths, centers); only
#                         pairs + centers are read here — they feed rendering
#                         AND the postprocess tools' neighbor maps. Invalidated
#                         on every annotation edit (_invalidate_grid_cache), so
#                         the edges can never outlive the vertex positions they
#                         were computed from.
#   _grid_raster_cache -> (H, W) uint8 mask, the rendered grid image
# The raster is a single-channel mask displayed through a yellow colormap with
# an alpha ramp — a quarter of the memory of RGBA, and patch updates stay
# simple 2D slice writes.


class GridMixin:
    """The grid raster layer, its dock controls, caches and rebuild scheduling."""

    def _init_grid_state(self) -> None:
        from qtpy.QtCore import QTimer

        self.grid_layer = None
        self._grid_frame = -1
        # Edges currently drawn into the layer raster, keyed by their (rounded,
        # sorted) endpoint coordinates — stable across point index shifts.
        self._grid_drawn: dict = {}
        self._grid_cache: dict[tuple[int, float], tuple[list[tuple[int, int]], np.ndarray, np.ndarray]] = {}
        self._grid_raster_cache: dict[tuple[int, float], np.ndarray] = {}
        self._grid_timer = QTimer()
        self._grid_timer.setSingleShot(True)
        self._grid_timer.timeout.connect(self._rebuild_grid)
        self._build_grid_controls()

    def _build_grid_controls(self) -> None:
        import math

        from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel, QVBoxLayout, QWidget

        class AutoStepSpinBox(QDoubleSpinBox):
            """At 0 ("auto"), +/- steps from the estimated auto value to the next multiple of the step, not from 0."""

            auto_value = 0.0

            def stepBy(self, steps: int) -> None:
                if self.value() == 0.0 and self.auto_value > 0 and steps:
                    step = self.singleStep()
                    base = self.auto_value / step
                    ticks = math.floor(base) if steps > 0 else math.ceil(base)
                    self.setValue(max(0.0, (ticks + steps) * step))
                else:
                    super().stepBy(steps)

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.grid_checkbox = QCheckBox("Show grid")
        self.grid_checkbox.toggled.connect(self._on_show_grid_toggled)
        layout.addWidget(self.grid_checkbox)

        layout.addWidget(QLabel("Neighbor dist (px, 0 = auto):"))
        self.grid_dist_spin = AutoStepSpinBox()
        self.grid_dist_spin.setRange(0.0, 5000.0)
        self.grid_dist_spin.setSingleStep(5.0)
        self.grid_dist_spin.setDecimals(1)
        # At 0 the spinbox shows the estimated auto cutoff instead of "0.0";
        # refreshed per frame in _rebuild_grid.
        self.grid_dist_spin.setSpecialValueText("auto")
        self.grid_dist_spin.valueChanged.connect(self._on_grid_dist_changed)
        layout.addWidget(self.grid_dist_spin)
        layout.addStretch(1)

        self._add_dock(widget, "Grid")

    def _show_grid(self) -> bool:
        return self.grid_checkbox.isChecked()

    def _grid_dist(self) -> float:
        return float(self.grid_dist_spin.value())

    def _on_show_grid_toggled(self, checked: bool) -> None:
        if self.grid_layer is not None and self.grid_layer in self.viewer.layers:
            self.grid_layer.visible = checked
        if checked:
            self._rebuild_grid()

    def _on_grid_dist_changed(self, _value: float) -> None:
        self._invalidate_grid_cache()
        self._schedule_grid_rebuild()

    def _grid_key(self, frame: int) -> tuple[int, float]:
        return (frame, round(self._grid_dist(), 3))

    def _invalidate_grid_cache(self) -> None:
        """Drop all cached edges/rasters."""
        self._grid_cache.clear()
        self._grid_raster_cache.clear()

    def _schedule_grid_rebuild(self) -> None:
        """Coalesce bursts of scrub/edit events into one rebuild per event-loop pass."""
        if self.entries:
            self._grid_timer.start(0)

    def _frame_neighbor_map(self, centers: np.ndarray) -> dict[int, list[int]]:
        """Current frame's neighbor map — from the grid cache when fresh, else computed synchronously."""
        cached = self._grid_cache.get(self._grid_key(self._current_index()))
        if cached is not None and len(cached[2]) == len(centers):
            return neighbor_map_from_pairs(cached[0])
        return neighbor_map(centers, self._grid_dist())

    def _update_grid_dist_hint(self) -> None:
        """Show the current frame's estimated auto cutoff in the spinbox's 0 position."""
        est = 0.0
        if self.entries and self.ann_layer is not None and len(self.ann_layer.data):
            data = np.asarray(self.ann_layer.data)
            pts = data[np.round(data[:, 0]).astype(int) == self._current_index(), 1:3]
            est = estimate_neighbor_dist(pts)
        self.grid_dist_spin.auto_value = est
        self.grid_dist_spin.setSpecialValueText(f"auto ({est:.1f} px)" if est else "auto")

    def _rebuild_grid(self) -> None:
        """Render the current frame's grid (computing its edges/raster if needed)."""
        self._update_grid_dist_hint()
        if not self.entries or self.ann_layer is None or len(self.ann_layer.data) == 0:
            self._remove_layer("grid_layer")
            self._grid_frame = -1
            self._grid_drawn = {}
            return
        self._render_current_grid()

    # -- raster rendering ---------------------------------------------------

    def _edge_dict(self, pairs: list[tuple[int, int]], centers: np.ndarray) -> dict:
        new_edges: dict = {}
        for i, j in pairs:
            p1 = (float(centers[i, 0]), float(centers[i, 1]))  # (y, x) full-res
            p2 = (float(centers[j, 0]), float(centers[j, 1]))
            key = tuple(sorted(((round(p1[0], 1), round(p1[1], 1)), (round(p2[0], 1), round(p2[1], 1)))))
            new_edges[key] = (p1, p2)
        return new_edges

    def _render_current_grid(self) -> None:
        """Show the current frame's grid raster (patching only the changed region when possible)."""
        frame = self._current_index()
        key = self._grid_key(frame)
        cached = self._grid_cache.get(key)
        if cached is None:
            data = np.asarray(self.ann_layer.data)
            pts = data[np.round(data[:, 0]).astype(int) == frame, 1:3]
            cached = edge_pairs(pts, self._grid_dist())
            self._grid_cache[key] = cached
        pairs, _, centers = cached
        f = self._display_factor()
        h, w = self._display_shape()
        new_edges = self._edge_dict(pairs, centers)

        layer_ok = (
            self.grid_layer is not None
            and self.grid_layer in self.viewer.layers
            and self.grid_layer.data.shape[:2] == (h, w)
        )
        if layer_ok and self._grid_frame == frame and self._grid_drawn:
            self._patch_grid_raster(new_edges)
        else:
            mask = self._grid_raster_cache.get(key)
            # Shape guard: a stale raster from before a display-scale change.
            if mask is None or mask.shape != (h, w):
                mask = np.zeros((h, w), dtype=np.uint8)
                self._draw_edges(mask, new_edges.values(), f)
                self._grid_raster_cache[key] = mask
            # Copy so later in-place patching never mutates the cache entry.
            raster = mask.copy()
            if layer_ok:
                self.grid_layer.data = raster
            else:
                self._remove_layer("grid_layer")
                self.grid_layer = self._add_grid_layer(raster, f)
                # Keep the grid raster BELOW the annotation points so the
                # label colours stay clearly visible (image < grid < points).
                self._fix_overlay_layer_order()
        self.grid_layer.visible = self._show_grid()
        self._grid_frame = frame
        self._grid_drawn = new_edges

    def _add_grid_layer(self, raster: np.ndarray, f: int):
        from napari.utils import Colormap

        return self.viewer.add_image(
            raster,
            name=GRID_LAYER_NAME,
            colormap=Colormap([[1, 1, 0, 0], [1, 1, 0, 1]], name="grid-yellow"),
            contrast_limits=[0, 255],
            opacity=0.8,
            scale=(f, f),
            visible=self._show_grid(),
        )

    def _draw_edges(self, mask: np.ndarray, edges, f: int) -> None:
        import cv2

        lw = max(1, round(GRID_WIDTH_PX / f))
        for p1, p2 in edges:
            cv2.line(
                img=mask,
                pt1=(int(round(p1[1] / f)), int(round(p1[0] / f))),
                pt2=(int(round(p2[1] / f)), int(round(p2[0] / f))),
                color=255, thickness=lw, lineType=cv2.LINE_AA,
            )

    def _patch_grid_raster(self, new_edges: dict) -> None:
        """Re-render only the region whose edges changed since the last draw."""
        old = self._grid_drawn
        changed = old.keys() ^ new_edges.keys()
        if not changed:
            return
        f = self._display_factor()
        raster = np.asarray(self.grid_layer.data)
        h, w = raster.shape[:2]
        # ponytail: one merged dirty box; switch to per-edge boxes if edits
        # ever span the whole frame and full redraw becomes the bottleneck.
        pts = np.array(
            [p for key in changed for p in (old.get(key) or new_edges[key])], dtype=float
        ) / f
        pad = max(1, round(GRID_WIDTH_PX / f)) + 2
        y0 = max(0, int(pts[:, 0].min()) - pad)
        y1 = min(h, int(pts[:, 0].max()) + pad + 1)
        x0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(w, int(pts[:, 1].max()) + pad + 1)
        raster[y0:y1, x0:x1] = 0
        # Redraw every edge touching the cleared box in full — overdrawing
        # outside the box is idempotent (constant colour).
        touching = [
            (p1, p2)
            for p1, p2 in new_edges.values()
            if not (
                max(p1[0], p2[0]) / f < y0
                or min(p1[0], p2[0]) / f > y1
                or max(p1[1], p2[1]) / f < x0
                or min(p1[1], p2[1]) / f > x1
            )
        ]
        self._draw_edges(raster, touching, f)
        self.grid_layer.refresh()
