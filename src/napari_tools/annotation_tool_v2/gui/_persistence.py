"""Reading and writing the per-image annotation JSONs for :class:`AnnotationApp` (a mixin)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.core.annotations import Annotation, AnnotationDoc, dump_data, shift_point

if TYPE_CHECKING:
    from qtpy.QtGui import QCloseEvent

logger = logging.getLogger(__name__)


class PersistenceMixin:
    """Per-frame ``AnnotationDoc`` state and the background JSON writer."""

    def _init_persistence_state(self) -> None:
        self.docs: list[Optional[AnnotationDoc]] = []
        self._ann_dirty = False
        self._io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="json-write")
        self._hook_close_handlers()

    def _load_docs(self) -> list[Optional[AnnotationDoc]]:
        """Load each frame's sibling ``<stem>.json``, if present."""
        docs: list[Optional[AnnotationDoc]] = []
        for entry in self.entries:
            json_path = entry.path.with_suffix(".json")
            doc = None
            if json_path.exists():
                try:
                    doc = AnnotationDoc.load(json_path)
                except (OSError, ValueError, KeyError) as exc:
                    self.log(f"Could not read {json_path.name}: {exc}")
            docs.append(doc)
        return docs

    def _activate_annotations(self) -> None:
        """Annotation state for the loaded folder: docs, per-image pins, breakpoints."""
        self.docs = self._load_docs()
        overridden = 0
        for idx, doc in enumerate(self.docs):
            pins = doc.pins() if doc is not None else []
            if pins:
                self.pins[idx] = pins
                overridden += 1
        if overridden:
            self.log(f"Corner pins from per-image JSONs on {overridden} frames.")
        self.breakpoints = {
            idx for idx, doc in enumerate(self.docs) if doc is not None and doc.sequence_breakpoint
        }
        self._ann_dirty = False

    def _sync_layer_to_docs(self) -> None:
        """Rebuild each doc's annotations from the points layer rows."""
        if self.ann_layer is None:
            return
        data = np.asarray(self.ann_layer.data)
        sizes = np.asarray(self.ann_layer.size, dtype=float).reshape(-1)
        feats = self.ann_layer.features
        per_frame: dict[int, list[Annotation]] = {}
        for row in range(len(data)):
            frame_idx = int(round(data[row, 0]))
            y, x = float(data[row, 1]), float(data[row, 2])
            pins = self._effective_pins(frame_idx)
            if pins:
                x, y = shift_point(x, y, pins, to_source=True)
            radius = float(sizes[row]) / 2.0
            per_frame.setdefault(frame_idx, []).append(
                Annotation(
                    id=str(feats["id"].iloc[row]),
                    center_x=x,
                    center_y=y,
                    radius=radius,
                    label=str(feats["label"].iloc[row]),
                )
            )
        for idx, doc in enumerate(self.docs):
            if doc is not None:
                doc.annotations = per_frame.get(idx, [])

    def _save_annotations(self, reason: str = "autosave") -> None:
        """Sync the points layer into the docs and write them (background)."""
        if self.ann_layer is None or not self._ann_dirty:
            return
        self._sync_layer_to_docs()
        written = 0
        for doc in self.docs:
            if doc is None:
                continue
            payload = doc.payload()
            self._submit_write(
                lambda p=doc.path, d=payload: dump_data(p, d),
                f"save {doc.path.name}",
            )
            written += 1
        self._ann_dirty = False
        self.log(f"Saving annotations to {written} JSONs in the background ({reason}).")

    def _hook_close_handlers(self) -> None:
        """Save the dock layout and flush unsaved annotation edits when the napari window is closed."""
        qt_window = self.viewer.window._qt_window
        original_close_event = qt_window.closeEvent

        def _close(event: QCloseEvent) -> None:
            self._save_layout()  # while the window is still intact
            self._save_annotations("window close")
            original_close_event(event)

        qt_window.closeEvent = _close

    def _submit_write(self, fn: Callable[[], object], desc: str) -> None:
        """Run a JSON write on the background worker, logging failures."""

        def _run() -> None:
            try:
                fn()
            except Exception:
                logger.exception("Background write failed: %s", desc)

        self._io_pool.submit(_run)
