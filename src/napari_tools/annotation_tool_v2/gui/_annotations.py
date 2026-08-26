"""Annotation points layer and brush selection for :class:`AnnotationApp` (a mixin)."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.core.annotations import UNLABELED, shift_point

if TYPE_CHECKING:
    import napari
    from napari.layers import Points
    from napari.utils.events import Event
    from vispy.app.canvas import MouseEvent

ANNOTATION_LAYER_NAME = "annotations"
BRUSH_LAYER_NAME = "brush select"
ANNOTATION_OPACITY = 0.75

_POINTS_HIGHLIGHT_PATCHED = False

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _restrict_brush_layer_tools():
    """v1's control-hiding helper, loaded from the external checkout by file path.

    annotation-tool (v1) is no longer an installed package — it runs straight out
    of ``external/`` via :mod:`src.cli.annotation_tool_shim`, so ``import
    annotation_tool.utils`` no longer resolves. Its ``utils.py`` is a flat
    top-level module that imports nothing but qtpy/napari/stdlib, so loading it by
    path works and keeps the generic name ``utils`` off ``sys.path`` (the shim can
    afford that only because it owns the process).

    Returns None when the checkout is missing (``./setup.sh`` not run): hiding the
    controls is a guard rail, not worth taking the whole tool down for.
    """
    import importlib.util

    from src.cli.annotation_tool_shim import ANNOTATION_TOOL

    path = ANNOTATION_TOOL / "src" / "utils.py"
    if not path.exists():
        logger.warning("annotation-tool (v1) checkout missing at %s — napari's layer controls stay unrestricted.", path)
        return None
    spec = importlib.util.spec_from_file_location("annotation_tool_v1_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.restrict_brush_layer_tools


def _patch_points_highlight() -> None:
    """Patches napari's ``Points._set_highlight`` to survive stale hover indices."""
    global _POINTS_HIGHLIGHT_PATCHED
    if _POINTS_HIGHLIGHT_PATCHED:
        return
    import napari.layers.points.points as napari_points

    original = napari_points.Points._set_highlight

    def safe_set_highlight(self: Points, force: bool = False) -> None:
        try:
            return original(self, force)
        except ValueError as exc:
            if "None is not in list" not in str(exc):
                raise
            self._value = None
            self._highlight_index = []
            return self.refresh()
        except IndexError:
            self._highlight_index = []
            return self.refresh()

    napari_points.Points._set_highlight = safe_set_highlight
    _POINTS_HIGHLIGHT_PATCHED = True


def _combine_selection(current: set[int], painted: set[int], mods) -> set[int]:
    """Merge a brush stroke into the selection: Ctrl+Shift removes, Shift adds, plain replaces."""
    from qtpy.QtCore import Qt

    # Ctrl+Shift carries ShiftModifier too — this branch has to come first.
    if mods & Qt.ControlModifier and mods & Qt.ShiftModifier:
        return current - painted
    if mods & Qt.ShiftModifier:
        return current | painted
    return painted


class AnnotationsMixin:
    """The annotation points layer, the brush-select layer and the selection helpers."""

    def _init_annotation_state(self) -> None:
        self.ann_layer = None
        self.brush_layer = None
        self._lock_labeled = False
        _patch_points_highlight()
        self._bind_annotation_keys()

    def _show_annotations(self) -> None:
        """Annotation, brush and grid layers + status/popup, after the images exist."""
        self._rebuild_annotation_layer()
        self._rebuild_brush_layer()
        self._invalidate_grid_cache()
        self._rebuild_grid()
        self._refresh_breakpoint_status()
        # Last, because _rebuild_grid's add_image would otherwise leave the grid
        # overlay as the active layer — napari selects whatever was added last.
        self._activate_brush_layer()
        self._maybe_warn_unclassified()

    def _annotation_layer_arrays(self) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """(coords, sizes, labels, ids) across all frames, display-shifted."""
        coords, sizes, labels, ids = [], [], [], []
        for idx, doc in enumerate(self.docs):
            if doc is None:
                continue
            pins = self._effective_pins(idx)
            for ann in doc.annotations:
                x, y = ann.center_x, ann.center_y
                if pins:
                    x, y = shift_point(x, y, pins)
                coords.append([idx, y, x])
                sizes.append(max(1.0, ann.radius * 2.0))
                labels.append(ann.label)
                ids.append(ann.id)
        return (np.array(coords, dtype=float).reshape(-1, 3), np.array(sizes, dtype=float), labels, ids)

    def _rebuild_annotation_layer(self) -> None:
        """Drop and re-add the annotation points layer for the current camera."""
        if self.ann_layer is not None and self.ann_layer in self.viewer.layers:
            self.viewer.layers.remove(self.ann_layer)
        self.ann_layer = None
        if not any(doc is not None for doc in self.docs):
            return
        self._ensure_labels_known()
        coords, sizes, labels, ids = self._annotation_layer_arrays()
        layer = self.viewer.add_points(
            coords,
            name=ANNOTATION_LAYER_NAME,
            features={"label": labels, "id": ids},
            face_color=self._face_colors(labels) if len(labels) else "white",
            size=sizes if len(sizes) else 10.0,
            border_color="black",
            opacity=ANNOTATION_OPACITY,
            out_of_slice_display=False,
        )
        self.ann_layer = layer
        layer.events.data.connect(self._on_ann_data_changed)
        layer.selected_data.events.items_changed.connect(self._on_ann_selection_changed)
        layer.mouse_wheel_callbacks.append(self._on_scroll_resize_points)
        self._fix_overlay_layer_order()
        self._restrict_layer_controls()
        self.log(f"Annotations: {len(labels)} cells on {sum(d is not None for d in self.docs)} frames.")

    def _on_ann_data_changed(self, event: Optional[Event] = None) -> None:
        action = getattr(event, "action", None)
        if action == "adding":
            defaults = self.ann_layer.feature_defaults
            defaults["id"] = str(uuid.uuid4())
            defaults["label"] = self._current_menu_label()
            self.ann_layer.feature_defaults = defaults
            return
        if action in {"added", "changed", "removed"}:
            self._ann_dirty = True
            if action in {"added", "removed"}:
                self._refresh_annotation_colors()
            self._invalidate_grid_cache()
            self._schedule_grid_rebuild()

    def _refresh_annotation_colors(self) -> None:
        if self.ann_layer is None or len(self.ann_layer.data) == 0:
            return
        labels = list(self.ann_layer.features["label"])
        self.ann_layer.face_color = self._face_colors(labels)

    def _before_pins_change(self) -> None:
        if self._ann_dirty:
            self._sync_layer_to_docs()

    def _after_pins_change(self) -> None:
        self._rebuild_annotation_layer()
        self._invalidate_grid_cache()
        self._rebuild_grid()

    def _rebuild_brush_layer(self) -> None:
        """A throwaway 2D paint mask over the frame: painting selects points."""
        if self.brush_layer is not None and self.brush_layer in self.viewer.layers:
            self.viewer.layers.remove(self.brush_layer)
        self.brush_layer = None
        if not self.entries:
            return
        h, w = self._display_shape()
        f = self._display_factor()
        layer = self.viewer.add_labels(
            np.zeros((h, w), dtype=np.uint8),
            name=BRUSH_LAYER_NAME,
            scale=(f, f),
            opacity=0.4,
        )
        layer.mode = "paint"
        layer.brush_size = 70 // self._display_factor()
        layer.events.paint.connect(self._on_brush_painted)
        self.brush_layer = layer
        self._fix_overlay_layer_order()
        self._restrict_layer_controls()

    def _activate_brush_layer(self) -> None:
        """Make the brush layer the active one, in paint mode (the 'b' key, and the state the tool starts in)."""
        if self.brush_layer is not None and self.brush_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.brush_layer
            self.brush_layer.mode = "paint"

    def _restrict_layer_controls(self) -> None:
        """Hide the napari controls that would break the annotation workflow using hbcsp annotation_tool function.
        """
        restrict_brush_layer_tools = _restrict_brush_layer_tools()
        if restrict_brush_layer_tools is None:
            return
        if self.ann_layer is not None and self.brush_layer is not None:
            restrict_brush_layer_tools(self.viewer, self.brush_layer, self.ann_layer)

    def _on_brush_painted(self, _event: Optional[Event] = None) -> None:
        """Select the current frame's points under the painted mask, then wipe it."""
        if self.ann_layer is None or self.brush_layer is None:
            return
        mask = np.asarray(self.brush_layer.data)
        f = self._display_factor()
        frame = self._current_index()
        data = np.asarray(self.ann_layer.data)
        painted: set[int] = set()
        # filtering for cells in only current frame
        for i in np.where(np.round(data[:, 0]).astype(int) == frame)[0]:
            # checks if point lays in mask
            my, mx = int(data[i, 1] / f), int(data[i, 2] / f)
            if 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1] and mask[my, mx]:
                painted.add(int(i))

        from qtpy.QtWidgets import QApplication

        selected = _combine_selection(set(self.ann_layer.selected_data), painted, QApplication.keyboardModifiers())
        self.ann_layer.selected_data = selected
        self.ann_layer.refresh(force=True)
        self.brush_layer.events.paint.disconnect(self._on_brush_painted)
        self.brush_layer.data = np.zeros_like(mask)
        self.brush_layer.events.paint.connect(self._on_brush_painted)

    def _bind_annotation_keys(self) -> None:
        """annotation-tool's cursor workflow keys (b/c/h, Ctrl+L, Ctrl+I).

        Only letters napari leaves free on *every* layer type. napari resolves a
        key against the active layer's keymap before the viewer's, and
        ``bind_key(..., overwrite=True)`` reaches only the viewer's own keymap —
        so a letter the Labels/Points/Image class already claims never gets here.
        Taken (napari/utils/shortcuts.py): p (paint + add point), x (swap label),
        y (plane normal), e, f, l, z, m, s, a, v, r, t, d, i, o.
        """
        viewer = self.viewer

        @viewer.bind_key("b", overwrite=True)
        def _brush_mode(_v: napari.Viewer) -> None:
            self._activate_brush_layer()

        @viewer.bind_key("c", overwrite=True)
        def _points_mode(_v: napari.Viewer) -> None:
            if self.ann_layer is not None and self.ann_layer in viewer.layers:
                viewer.layers.selection.active = self.ann_layer
                self.ann_layer.mode = "select"

        @viewer.bind_key("h", overwrite=True)
        def _points_toggle(_v: napari.Viewer) -> None:
            if self.ann_layer is not None:
                self.ann_layer.opacity = 0.0 if self.ann_layer.opacity else ANNOTATION_OPACITY

        for combo in ("Control-l", "Meta-l"):

            @viewer.bind_key(combo, overwrite=True)
            def _lock(_v: napari.Viewer) -> None:
                self.lock_checkbox.toggle()

        for combo in ("Control-i", "Meta-i"):

            @viewer.bind_key(combo, overwrite=True)
            def _inverse(_v: napari.Viewer) -> None:
                self._delete_inverse_selection()

    def _on_lock_toggled(self, checked: bool) -> None:
        self._lock_labeled = checked
        self.log(f"Lock labeled points: {'on' if checked else 'off'}.")
        self._filter_locked_selection()

    def _on_ann_selection_changed(self, _event: Optional[Event] = None) -> None:
        self._filter_locked_selection()
        self._update_selection_borders()
        if hasattr(self, "_update_selection_buttons"):
            self._update_selection_buttons()

    def _filter_locked_selection(self) -> None:
        """With lock-labeled on, only 'unlabeled' points stay selectable."""
        if not self._lock_labeled or self.ann_layer is None:
            return
        features = self.ann_layer.features
        keep = {
            i for i in self.ann_layer.selected_data if features["label"].iloc[i] == UNLABELED
        }
        if keep != set(self.ann_layer.selected_data):
            self.ann_layer.selected_data = keep

    def _update_selection_borders(self) -> None:
        """White borders on selected points, black otherwise."""
        layer = self.ann_layer
        if layer is None or len(layer.data) == 0:
            return
        borders = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(layer.data), 1))
        for i in layer.selected_data:
            if 0 <= i < len(borders):
                borders[i] = (1.0, 1.0, 1.0, 1.0)
        layer.border_color = borders

    def _delete_inverse_selection(self) -> None:
        """Keep only the selected points on the *current frame* (Ctrl+I)."""
        if self.ann_layer is None:
            return
        selected = set(self.ann_layer.selected_data)
        if not selected:
            self.log("No points selected, nothing to delete.")
            return
        data = np.asarray(self.ann_layer.data)
        frame = self._current_index()
        frame_rows = set(np.where(np.round(data[:, 0]).astype(int) == frame)[0].tolist())
        inverse = frame_rows - selected
        if not inverse:
            self.log("No unselected points on this frame.")
            return
        self.ann_layer.selected_data = inverse
        self.ann_layer.remove_selected()
        self.log(f"Deleted {len(inverse)} unselected points on frame {frame}.")

    def _on_scroll_resize_points(self, layer: Points, event: MouseEvent) -> None:
        """Alt + scroll resizes the selected points (annotation-tool port)."""
        if "Alt" not in event.modifiers:
            return
        selected = list(layer.selected_data)
        if not selected:
            return
        d = event.delta[0] if event.delta[0] else event.delta[1]
        delta = 2.0 if d > 0 else -2.0
        sizes = np.asarray(layer.size, dtype=float).reshape(-1).copy()
        for idx in selected:
            sizes[idx] = min(100.0, max(1.0, sizes[idx] + delta))
        layer.size = sizes
        layer.selected_data = set(selected)
        self._ann_dirty = True
