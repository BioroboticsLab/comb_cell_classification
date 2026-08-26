"""Sequence breakpoints for :class:`AnnotationApp` (a mixin): the dock, the toggle and its status line."""

from __future__ import annotations

from src.core.annotations import write_breakpoint


class BreakpointsMixin:
    """The frames marked as the start of a new sequence, and their dock."""

    def _init_breakpoint_state(self) -> None:
        self.breakpoints: set[int] = set()
        self._build_breakpoint_controls()

    def _build_breakpoint_controls(self) -> None:
        from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        toggle = QPushButton("Toggle breakpoint @ current frame")
        toggle.clicked.connect(lambda: self._toggle_breakpoint())
        layout.addWidget(toggle)
        self.bp_status = QLabel("")
        self.bp_status.setWordWrap(True)
        layout.addWidget(self.bp_status)
        self._add_dock(widget, "Breakpoints")

    def _toggle_breakpoint(self) -> None:
        """Flip the current frame's breakpoint and write its JSON immediately."""
        if not self.entries:
            return
        idx = self._current_index()
        flag = idx not in self.breakpoints
        self.breakpoints.symmetric_difference_update({idx})
        doc = self.docs[idx] if idx < len(self.docs) else None
        if doc is not None:
            doc.sequence_breakpoint = flag
            json_path = doc.path
        else:
            json_path = self.entries[idx].path.with_suffix(".json")
        self._submit_write(lambda p=json_path, f=flag: write_breakpoint(p, f), f"breakpoint {json_path.name}")
        self._refresh_breakpoint_status()
        self.log(f"Breakpoint {'set' if flag else 'cleared'} @ frame {idx} (writing {json_path.name} in the background).")

    def _refresh_breakpoint_status(self) -> None:
        if not hasattr(self, "bp_status") or not self.entries:
            return
        idx = self._current_index()
        here = idx in self.breakpoints
        self.bp_status.setText(f"Breakpoint here: {here}  |  {len(self.breakpoints)} total (saved to the per-image JSONs automatically)")
