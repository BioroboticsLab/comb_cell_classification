"""Input + UI for :class:`annotation_tool_v2.app.AnnotationApp` (a mixin)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


class ControlsMixin:
    """Dock widget and key bindings for the corner-pin viewer."""

    def _build_widget(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(QLabel("Display scale (performance):"))
        self.downsample_combo = QComboBox()
        self._downsample_options = [("Full", 1), ("Half (1/2)", 2), ("Quarter (1/4)", 4)]
        for label, _factor in self._downsample_options:
            self.downsample_combo.addItem(label)
        self.downsample_combo.currentIndexChanged.connect(self._on_downsample_changed)
        layout.addWidget(self.downsample_combo)

        copy_btn = QPushButton("Copy image path")
        copy_btn.setToolTip("Copy the current image's absolute path to the clipboard.")
        copy_btn.clicked.connect(self._copy_current_path)
        layout.addWidget(copy_btn)

        self.warp_btn = QPushButton("Warp: stabilized (w)")
        self.warp_btn.setCheckable(True)
        self.warp_btn.setChecked(True)
        self.warp_btn.toggled.connect(self._on_warp_toggled)
        layout.addWidget(self.warp_btn)

        layout.addStretch(1) # stretchable empty space with factor = 1

        self.widget = widget

        self._add_dock(widget, "Controls")

    def _build_file_info(self) -> None:
        """Current image name in the window's bottom status bar (no extra panel)."""
        self._file_name_label = QLabel("—")
        self.viewer.window._qt_window.statusBar().addPermanentWidget(self._file_name_label)
        self._update_file_info()

    def _current_entry_path(self) -> Optional[Path]:
        """Absolute path of the current frame's image, or None."""
        if not self.entries:
            return None
        return self.entries[self._current_index()].path.resolve()

    def _update_file_info(self) -> None:
        if getattr(self, "_file_name_label", None) is None:
            return
        path = self._current_entry_path()
        self._file_name_label.setText("—" if path is None else path.name)

    def _copy_current_path(self) -> None:
        from qtpy.QtWidgets import QApplication

        path = self._current_entry_path()
        if path is None:
            return
        QApplication.clipboard().setText(str(path))
        self.log(f"Copied path: {path}")

    def _bind_keys(self) -> None:
        viewer = self.viewer

        for combo, delta in (
            ("Meta-Left", -1),
            ("Control-Left", -1),
            ("Meta-Right", 1),
            ("Control-Right", 1),
        ):
            @viewer.bind_key(combo, overwrite=True)
            def _step(_v: napari.Viewer, _d: int = delta) -> None:
                self._step_frame(_d)

        @viewer.bind_key("w", overwrite=True)
        def _warp_key(_v: napari.Viewer) -> None:
            self.warp_btn.toggle()

        @viewer.bind_key("g", overwrite=True)
        def _grid_key(_v: napari.Viewer) -> None:
            self.grid_checkbox.toggle()

    def _step_frame(self, delta: int) -> None:
        """Move the dimension slider by ``delta`` frames (clamped to the stack)."""
        if not self.entries:
            return
        self._set_current_index(self._current_index() + delta)

    def _on_warp_toggled(self, checked: bool) -> None:
        """Switch between stabilized (corner pins applied) and raw frames."""
        if checked == self.warp_enabled:
            return
        self._before_pins_change()
        self.warp_enabled = checked
        self.warp_btn.setText("Warp: stabilized (w)" if checked else "Warp: raw (w)")
        if self.entries:
            self._rebuild_view()
        self._after_pins_change()
        self.log(f"Warp {'enabled (stabilized)' if checked else 'disabled (raw frames)'}.")

    def _display_downsample(self) -> int:
        """The proxy-scale factor currently selected in the Display scale combo."""
        return self._downsample_options[max(0, self.downsample_combo.currentIndex())][1]

    def _on_downsample_changed(self, index: int) -> None:
        if index < 0:
            return
        factor = self._downsample_options[index][1]
        if factor == self.store.proxy_scale:
            return
        if self.entries:
            self.store = self._new_store()
            self.store.prefetch(self._current_index())
        self._rebuild_view()
        self._rebuild_brush_layer()
        suffix = "full resolution" if factor == 1 else f"1/{factor} (JPEG proxy, view only)"
        self.log(f"Display scale set to {suffix}.")
