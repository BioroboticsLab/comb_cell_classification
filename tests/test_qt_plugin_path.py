"""annotation-tool_v2 points QT_QPA_PLATFORM_PLUGIN_PATH back at PyQt5 when opencv-python has taken it over, and leaves it alone otherwise."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
PyQt5 = pytest.importorskip("PyQt5")

from src.napari_tools.annotation_tool_v2.__main__ import _repair_qt_plugin_path

VAR = "QT_QPA_PLATFORM_PLUGIN_PATH"
PYQT_PLUGINS = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"


@pytest.mark.skipif(not (PYQT_PLUGINS / "platforms").is_dir(), reason="this PyQt5 ships no platform plugins")
def test_cv2_plugin_path_is_pointed_back_at_pyqt5(monkeypatch):
    # What the full opencv-python wheel sets at import on Linux.
    monkeypatch.setenv(VAR, str(Path(cv2.__file__).resolve().parent / "qt" / "plugins"))
    _repair_qt_plugin_path()
    assert Path(os.environ[VAR]) == PYQT_PLUGINS


def test_other_plugin_path_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setenv(VAR, str(tmp_path))
    _repair_qt_plugin_path()
    assert os.environ[VAR] == str(tmp_path)


def test_unset_stays_unset(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    _repair_qt_plugin_path()
    assert VAR not in os.environ
