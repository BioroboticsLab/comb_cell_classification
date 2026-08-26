"""Classify the loaded sequence with a trained model (popup dialog + worker thread)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.core.annotations import dump_data

from ._prefs import load_prefs, save_prefs

if TYPE_CHECKING:
    from .app import AnnotationApp

BATCH_SIZE = 64


class ClassifyMixin:
    """The "Classify sequence…" popup (Linux only — TF inference)."""

    def _open_classify_dialog(self) -> None:
        if not self.entries:
            self.log("Load a folder before classifying.")
            return
        ClassifyDialog(self).exec_()


class ClassifyDialog(QDialog):
    """Model-folder input, Start, progress bar, Abort; closes itself when done."""

    # Emitted from the worker thread; queued connections marshal to the UI thread.
    _progress = Signal(int, int)
    _finished = Signal(str)  # "" = success, "aborted …" = user abort, "error: …" = failure

    def __init__(self, app: AnnotationApp) -> None:
        super().__init__(app.viewer.window._qt_window)
        self.app = app
        self._abort = threading.Event()
        self._running = False

        self.setWindowTitle("Classify sequence")
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path_edit = QLineEdit(str(load_prefs().get("classify_model_dir", "")))
        self.path_edit.setPlaceholderText("Model folder (.keras + model_info.json)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        layout.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        layout.addWidget(self.bar)
        self.status = QLabel("")
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        abort_btn = QPushButton("Abort")
        abort_btn.clicked.connect(self.reject)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(abort_btn)
        layout.addLayout(buttons)

        self._progress.connect(self._on_progress)
        self._finished.connect(self._on_finished)

    def _browse(self) -> None:
        picked = QFileDialog.getExistingDirectory(
            self, "Model folder", self.path_edit.text() or str(Path.home())
        )
        if picked:
            self.path_edit.setText(picked)

    def _start(self) -> None:
        model_dir = Path(self.path_edit.text()).expanduser()
        if not model_dir.is_dir():
            QMessageBox.warning(self, "Classify", f"Not a folder: {model_dir}")
            return
        save_prefs(classify_model_dir=str(model_dir))
        self._running = True
        self.start_btn.setEnabled(False)
        self.path_edit.setEnabled(False)
        self.status.setText("Loading model…")
        self.bar.setRange(0, 0)  # busy indicator while TF loads the checkpoint
        self.app._save_annotations("before classify")
        threading.Thread(target=self._work, args=(model_dir,), daemon=True).start()

    def _work(self, model_dir: Path) -> None:
        try:
            # TF import lives here so opening the dialog stays instant.
            import cv2

            from src.core.config.config import BeeCombConfig
            from src.inference.inference import (
                build_crops,
                load_model_and_classes,
                predict_labels,
                sync_cfg_to_model,
            )

            model, class_names, h, w, c = load_model_and_classes(model_dir)
            cfg = BeeCombConfig()
            sync_cfg_to_model(cfg, model_dir, h, w, c)
            read_flag = cv2.IMREAD_GRAYSCALE if c == 1 else cv2.IMREAD_COLOR

            todo = [
                (entry, doc)
                for entry, doc in zip(self.app.entries, self.app.docs)
                if doc is not None and len(doc.annotations)
            ]
            for done, (entry, doc) in enumerate(todo):
                if self._abort.is_set():
                    self._finished.emit(f"aborted after {done}/{len(todo)} frames")
                    return
                self._progress.emit(done, len(todo))
                img = cv2.imread(str(entry.path), read_flag)
                if img is None:
                    continue
                crops = build_crops(img, doc.annotations, cfg, h, w, c)
                for ann, label in zip(
                    doc.annotations, predict_labels(model, crops, class_names, BATCH_SIZE)
                ):
                    ann.label = label
                dump_data(doc.path, doc.payload())
            self._finished.emit("")
        except ImportError as exc:
            # TF is a Linux-only dependency (see pyproject); the rest of the
            # annotation tool still runs on macOS, only classify is unavailable.
            self._finished.emit(f"classification needs TensorFlow — Linux only ({exc})")
        except Exception as exc:  # anything (bad checkpoint, OOM, …) surfaces in the dialog
            self._finished.emit(f"error: {exc}")

    def _on_progress(self, done: int, total: int) -> None:
        self.bar.setRange(0, total)
        self.bar.setValue(done)
        self.status.setText(f"Classifying frame {done + 1}/{total}…")

    def _on_finished(self, message: str) -> None:
        self._running = False
        self.app._ann_dirty = False 
        self.app._rebuild_annotation_layer()
        if message.startswith("error: "):
            QMessageBox.warning(self, "Classify", message[len("error: "):])
            self.start_btn.setEnabled(True)
            self.path_edit.setEnabled(True)
            self.bar.setRange(0, 1)
            self.bar.setValue(0)
            self.status.setText("")
            return
        self.app.log(f"Classification {message or 'finished'}.")
        self.accept()

    def reject(self) -> None:  # Abort button, Esc and the window's close button
        if self._running:
            self._abort.set()
            self.status.setText("Aborting…")
            return  # stay open; the worker emits _finished at the next frame boundary
        super().reject()
