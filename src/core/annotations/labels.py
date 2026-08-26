"""Shared helpers for the ``index<TAB>name`` label-map format used by both training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Union

def read_label_map(path: Union[str, Path]) -> list[str]:
    """Parse a ``label_map.txt`` (``index<TAB>name`` per line) into a list where position equals the label index the model emits."""
    index_to_label: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, label = line.split("\t", 1)
            index_to_label[int(idx_str)] = label
    return [index_to_label[i] for i in sorted(index_to_label)]

def write_label_map(class_names: list[str], path: Union[str, Path]) -> None:
    """Write an index-ordered class-name list as ``index<TAB>name`` per line (inverse of :func:`read_label_map`)."""
    lines = [f"{idx}\t{name}" for idx, name in enumerate(class_names)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
