"""The annotation tool application object."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import napari
import numpy as np
from napari.utils.events import Event

from ..core.frames import FrameStore
from ..core.parsing import ImageEntry, find_images_and_sort
from ._annotations import AnnotationsMixin
from ._breakpoints import BreakpointsMixin
from ._classify import ClassifyMixin
from ._controls import ControlsMixin
from ._display import DisplayMixin
from ._grid import GridMixin
from ._labels import LabelsMixin
from ._persistence import PersistenceMixin
from ._postprocess import PostprocessMixin
from ._prefs import load_prefs, save_prefs

PRELOAD_RADIUS = 3


class AnnotationApp(
    ControlsMixin,
    DisplayMixin,
    AnnotationsMixin,
    LabelsMixin,
    PersistenceMixin,
    BreakpointsMixin,
    ClassifyMixin,
    GridMixin,
    PostprocessMixin,
):
    """Owns the napari viewer, the dock widget, and the per-frame perspective state."""

    def __init__(self, viewer: Optional[napari.Viewer] = None) -> None:
        from qtpy.QtCore import QTimer

        self.viewer = viewer if viewer is not None else napari.Viewer()

        self.folder: Optional[Path] = None
        self.entries: list[ImageEntry] = []
        self.store = FrameStore([])
        self.pins: list[list[dict]] = []
        self.warp_enabled = True
        self._warp_cache: dict[int, np.ndarray] = {}

        self.image_layer = None

        self._build_log()
        self._build_widget()
        self._build_file_info()
        self._bind_keys()
        self._init_persistence_state()
        self._init_annotation_state()
        
        # Remaining _init/_build calls each add their dock, in on-screen order.
        self._init_label_state()
        self._init_breakpoint_state()
        self._init_grid_state()
        self._build_selection_controls()
        self._build_postprocess_controls()
        
        # Queued, so it runs after every dock's own singleShot height re-measure.
        QTimer.singleShot(0, self._restore_layout)
        self.viewer.dims.events.current_step.connect(self._on_dims_changed)

        self._enforcing_2d = False
        self.viewer.dims.ndisplay = 2
        self.viewer.dims.events.ndisplay.connect(self._enforce_2d)

    def _add_dock(self, widget, name: str, area: str = "right", scroll: bool = True):
        """Add a dock panel: scroll-wrapped so it stays usable at small window sizes, and listed in the 'Panels' menu so a hidden dock can be reopened."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QScrollArea

        if scroll:
            content = widget
            wrapper = QScrollArea()
            wrapper.setWidget(content)
            wrapper.setWidgetResizable(True)
            wrapper.setFrameShape(QScrollArea.NoFrame)
            wrapper.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # never scrolling horizontally
            wrapper.setMaximumHeight(content.sizeHint().height() + 8)
            widget = wrapper
        dock = self.viewer.window.add_dock_widget(widget, name=name, area=area)
        # restoreState matches docks by object name; without one the saved
        # arrangement cannot be put back.
        dock.setObjectName(name)
        if scroll:
            # For _refresh_dock_height when a panel's content grows later.
            dock._ccc_scroll = wrapper
            dock._ccc_content = content
            # napari's stylesheet(padding, fonts) applies only once the event loop runs
            # re-measure afterwards, otherwise cap clips panel
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, lambda: self._refresh_dock_height(dock))
        if not hasattr(self, "_panels_menu"):
            self._panels_menu = self.viewer.window._qt_window.menuBar().addMenu("Panels")
        # toggleViewAction: checkable Qt action that hides/restores the dock.
        self._panels_menu.addAction(dock.toggleViewAction())
        return dock

    @staticmethod
    def _refresh_dock_height(dock) -> None:
        """Re-cap a scroll-wrapped dock to its content height after the content changed size (e.g. the label legend grew)."""
        scroll = getattr(dock, "_ccc_scroll", None)
        content = getattr(dock, "_ccc_content", None)
        if scroll is not None and content is not None:
            scroll.setMaximumHeight(content.sizeHint().height() + 8)

    def _save_layout(self) -> None:
        """Remember how the docks are arranged and how big the window is."""
        window = self.viewer.window._qt_window
        save_prefs(window_geometry=bytes(window.saveGeometry().toBase64()).decode("ascii"), window_state=bytes(window.saveState().toBase64()).decode("ascii"))

    def _restore_layout(self) -> None:
        """Put back the saved arrangement, matched dock by dock via object name."""
        from qtpy.QtCore import QByteArray

        prefs = load_prefs()
        window = self.viewer.window._qt_window
        for key, restore in (("window_geometry", window.restoreGeometry), ("window_state", window.restoreState)):
            blob = prefs.get(key)
            # A layout saved by an older version simply does not apply: Qt
            # reports False and leaves the freshly built default alone.
            if blob:
                restore(QByteArray.fromBase64(blob.encode("ascii")))

    def _build_log(self) -> None:
        """The timestamped Log dock (left)."""
        from qtpy.QtWidgets import QListWidget

        self.log_widget = QListWidget()
        self.log_widget.setAlternatingRowColors(True)
        # A list widget scrolls itself — no scroll wrapper.
        self._add_dock(self.log_widget, "Log", area="left", scroll=False)

    def log(self, message: str) -> None:
        """Append a timestamped line to the scrollable Log dock."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_widget.addItem(f"[{timestamp}]  {message}")
        self.log_widget.scrollToBottom()

    def _on_dims_changed(self, _event: Optional[Event] = None) -> None:
        if self.entries:
            self.store.prefetch(self._current_index())
            self._refresh_breakpoint_status()
            self._schedule_grid_rebuild()
            self._update_file_info()

    def _enforce_2d(self, _event: Optional[Event] = None) -> None:
        """Snap the viewer back to 2D if it switches to (crash-prone) 3D rendering."""
        if self._enforcing_2d or self.viewer.dims.ndisplay == 2:
            return
        self._enforcing_2d = True
        try:
            self.viewer.dims.ndisplay = 2
        finally:
            self._enforcing_2d = False
        self.log("Kept the viewer in 2D — 3D rendering of large image stacks is unsupported.")

    def load_folder(self, folder: Union[str, Path]) -> None:
        """Discover, parse and sort the images in ``folder``, then show them."""
        folder = Path(folder)
        entries = find_images_and_sort(folder)
        if not entries:
            self.log("No matching .png files found in the selected folder.")
            return
        self.folder = folder
        self.entries = entries
        self._warned_unclassified = False
        self._load_label_map(folder)

        self.pins = [[] for _ in entries]
        self.store = self._new_store()
        self.store.prefetch(0)
        self._activate_annotations()
        self._rebuild_view()
        self._show_annotations()
        self.log(f"Loaded {len(entries)} frames from {folder.name}.")

    def _new_store(self) -> FrameStore:
        """A FrameStore for the current entries, using the JPEG-proxy view scale."""
        return FrameStore(self.entries, radius=PRELOAD_RADIUS, proxy_scale=self._display_downsample())
