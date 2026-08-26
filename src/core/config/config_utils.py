import copy
import itertools
from typing import Any, Optional

from src.core.config.config import BeeCombConfig
from src.core.project_settings import project_settings

# Fixed slot+letter per augmentation for the run-name/tag code: each slot shows
# its letter when the augmentation is enabled, "0" when not — e.g. rotate_60
# alone is "r000", rotate_60+flip "rf00", all four "rfzw", none "0000".
AUGMENTATION_CODE_SLOTS: list[tuple[str, str]] = [
    ("rotate_60", "r"),
    ("flip", "f"),
    ("zoom", "z"),
    ("wiggle_around_center", "w"),
]


def augmentation_code(augmentations: Optional[list[str]]) -> str:
    """4-char code identifying a run's augmentation combination (see AUGMENTATION_CODE_SLOTS)."""
    enabled = set(augmentations or [])
    return "".join(letter if name in enabled else "0" for name, letter in AUGMENTATION_CODE_SLOTS)


def all_augmentation_combinations() -> list[list[str]]:
    """Every on/off subset (2**4 = 16, "0000" through "rfzw") of the augmentations in AUGMENTATION_CODE_SLOTS, for factorial sweeps."""
    names = [name for name, _ in AUGMENTATION_CODE_SLOTS]
    return [
        [name for name, keep in zip(names, mask) if keep]
        for mask in itertools.product([False, True], repeat=len(names))
    ]


def build_cfg_variations(
    base_cfg: BeeCombConfig,
    spec: dict[str, tuple[Optional[str], list[Any]]],
) -> list[BeeCombConfig]:
    """Resolve each ``spec`` entry (dotted config path -> ``(project_settings attribute, defaults)``) to the CLI-override list if truthy, else the defaults, and build the cartesian product of configs. ``attr=None`` means no CLI counterpart (YAML-only sweep dim)."""
    resolved = {
        path: ((attr and getattr(project_settings, attr)) or default)
        for path, (attr, default) in spec.items()
    }
    return create_config_variations(base_cfg, **resolved)

def create_config_variations(
    base_cfg: BeeCombConfig,
    **variation_params: list[Any]
) -> list[BeeCombConfig]:
    """Generate one deep-copied config per combination in the cartesian product of the given value lists, each keyed by a dotted config path (e.g. ``"training.learning_rate"``)."""
    if not variation_params:
        return [copy.deepcopy(base_cfg)]

    keys = list(variation_params.keys())
    value_lists = list(variation_params.values())

    for key, values in zip(keys, value_lists):
        if not isinstance(values, list):
            raise TypeError(
                f"Expected a list for parameter '{key}', got {type(values).__name__}"
            )
        if len(values) == 0:
            raise ValueError(f"Parameter '{key}' has an empty list of values.")

    cfgs = []
    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(base_cfg)

        for key, value in zip(keys, combo):
            _set_nested_attr(cfg, key, value)

        cfgs.append(cfg)

    return cfgs


def _set_nested_attr(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a nested attribute via a dotted path (``"training.learning_rate"`` == ``cfg.training.learning_rate``), raising ``AttributeError`` if any part is missing."""
    parts = dotted_path.split(".")

    target = obj
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise AttributeError(
                f"'{type(target).__name__}' has no attribute '{part}' "
                f"(in path '{dotted_path}')"
            )
        target = getattr(target, part)

    final_attr = parts[-1]
    if not hasattr(target, final_attr):
        raise AttributeError(
            f"'{type(target).__name__}' has no attribute '{final_attr}' "
            f"(in path '{dotted_path}')"
        )
    setattr(target, final_attr, value)
