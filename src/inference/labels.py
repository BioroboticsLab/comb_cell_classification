"""Resolve the model's ordered output class names from a companion file in its folder."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from src.core.annotations import read_label_map
from src.core.model_info import class_names_from_info

def read_class_names(model_dir: Union[str, Path]) -> list[str]:
    """Return the index-ordered class names from the model folder."""
    model_dir = Path(model_dir)

    names = class_names_from_info(model_dir)
    if names:
        return names

    label_maps = sorted(model_dir.glob("*label_map.txt"))
    if label_maps:
        return read_label_map(label_maps[0])

    raise FileNotFoundError(
        f"No class-name source in {model_dir}. Add a 'model_info.json' with a 'class_names' list or a 'label_map.txt' (index<TAB>name per line)."
    )
