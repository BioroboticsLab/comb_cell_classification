from dataclasses import dataclass
from typing import Optional

@dataclass
class ProjectSettings:
    debug: bool = False
    interactive: bool = False
    silent: bool = False
    log_level: str = "INFO"
    models: Optional[list[str]] = None
    learning_rates: Optional[list[float]] = None
    unfreeze_after: Optional[list[int]] = None
    finetune_lrs: Optional[list[float]] = None
    use_adaptive_lr: Optional[list[bool]] = None
    cell_outer_layers: Optional[list[int]] = None
    apply_clahes: Optional[list[bool]] = None
    use_class_weights: Optional[list[bool]] = None
    oversamples: Optional[list[bool]] = None
    # Augmentation-combination sweep arms as augmentation-name lists, parsed from 4-char codes on CLI ("rf00" -> ["rotate_60", "flip"]).
    augmentation_combos: Optional[list[list[str]]] = None

    def __post_init__(self) -> None:
        if self.interactive and self.silent:
            raise ValueError("interactive and silent cannot both be True.")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")

project_settings = ProjectSettings()

