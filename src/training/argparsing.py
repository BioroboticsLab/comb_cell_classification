from argparse import ArgumentParser, ArgumentTypeError

from typing import Callable, Optional, TypeVar, Union
import src.core.project_settings as settings
from src.core.config.config import BeeCombConfig
from src.core.config.config_utils import AUGMENTATION_CODE_SLOTS

T = TypeVar("T")


def _csv(item_parser: Callable[[str], T]) -> Callable[[str], list[T]]:
    """argparse ``type=`` factory that splits on commas and parses each item."""
    def parse(raw: str) -> list[T]:
        return [item_parser(x.strip()) for x in raw.split(",") if x.strip()]
    return parse


def _to_bool(raw: str) -> bool:
    v = raw.lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ArgumentTypeError(f"Expected bool (true/false/1/0/yes/no), got {raw!r}")


def _augmentation_combo(raw: str) -> list[str]:
    """Parse a 4-char augmentation code ("rf00") back into the augmentation-name list it stands for."""
    code = raw.strip().lower()
    if len(code) != len(AUGMENTATION_CODE_SLOTS):
        raise ArgumentTypeError(
            f"Expected a {len(AUGMENTATION_CODE_SLOTS)}-char augmentation code like 'rf00', got {raw!r}"
        )
    names = []
    for ch, (name, letter) in zip(code, AUGMENTATION_CODE_SLOTS):
        if ch == letter:
            names.append(name)
        elif ch != "0":
            raise ArgumentTypeError(
                f"Bad augmentation code {raw!r}: slot for {name} must be {letter!r} or '0', got {ch!r}"
            )
    return names


def _lr_value(raw: str) -> Union[float, str]:
    if raw.strip().lower() == "adaptive":
        return "adaptive"
    try:
        v = float(raw)
    except ValueError:
        raise ArgumentTypeError(f"Expected a learning rate (positive float) or 'adaptive', got {raw!r}")
    if v <= 0:
        raise ArgumentTypeError(f"Learning rate must be positive or 'adaptive', got {v}")
    return v

help_text = """
This script trains a model on the bee comb dataset.
Usage (the training driver, run detached by `ccc training`):
    python -m src.cli.train _train --config config.yaml
    python -m src.cli.train _train --interactive
    python -m src.cli.train _train --non-interactive
    python -m src.cli.train _train --debug
    python -m src.cli.train _train --silent
    python -m src.cli.train _train --log-level DEBUG
    python -m src.cli.train _train --max-worker-threads 4
    python -m src.cli.train _train --models
    python -m src.cli.train _train --learning-rates
    python -m src.cli.train _train --unfreeze-after
    python -m src.cli.train _train --finetune-lrs
    python -m src.cli.train _train --cell-outer-layers
    python -m src.cli.train _train --apply-clahes
    python -m src.cli.train _train --augmentation-combos
"""

def parse_args_to_project(cfg: BeeCombConfig, argv: Optional[list[str]] = None) -> None:
    """Parse CLI args into ``project_settings`` and, when debug mode is active, rewrite the wandb project and data paths on ``cfg`` to their debug variants."""
    parser = ArgumentParser(description="Train a model on the bee comb dataset.", add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to the configuration file.")

    # Interactive and non-interactive are mutually exclusive; debug is orthogonal.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--interactive", action="store_true", default=None, help="Run in interactive mode.")
    mode_group.add_argument("--non-interactive", action="store_true", default=None, help="Run in non-interactive mode (default when launched via script).")

    parser.add_argument("--debug", action="store_true", default=None, help="Run in debug mode (uses ccc_debug wandb project and data/annotated_DEBUG dataset).")

    parser.add_argument("--silent", action="store_true", default=None, help="Run in silent mode.")
    parser.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    parser.add_argument("--max-worker-threads", type=int, default=None, help="Maximum number of worker threads for data loading and processing.")

    parser.add_argument("--models", type=_csv(str), default=None, help="Name of all models to train. Multiple values can be passed as comma-separated list, e.g. --models InceptionV3,ResNet50V2")
    parser.add_argument("--learning-rates", type=_csv(_lr_value), default=None, help="Learning rates for the optimizer. Use the literal 'adaptive' for adaptive learning rate. Multiple values can be passed as comma-separated list, e.g. --learning-rates 0.001,adaptive,0.0001")
    parser.add_argument("--unfreeze-after", type=_csv(int), default=None, help="Number of head warm-up epochs before unfreezing the backbone (0 = unfrozen: backbone trainable from epoch 1, -1 = frozen: backbone stays frozen for the whole run). Multiple values can be passed as comma-separated list, e.g. --unfreeze-after 3,5")
    parser.add_argument("--finetune-lrs", type=_csv(float), default=None, help="Learning rate for the unfrozen fine-tune phase after the warm-up. Multiple values can be passed as comma-separated list, e.g. --finetune-lrs 3e-5,1e-5")
    parser.add_argument("--cell-outer-layers", type=_csv(int), default=None, help="Size of the outer layer of cells. Multiple values can be passed as comma-separated list, e.g. --cell-outer-layers 0,1")
    parser.add_argument("--apply-clahes", type=_csv(_to_bool), default=None, help="Apply CLAHE to the images. Multiple values can be passed as comma-separated list, e.g. --apply-clahes True,False")
    parser.add_argument("--augmentation-combos", type=_csv(_augmentation_combo), default=None, help="Augmentation combinations as 4-char codes — one slot per augmentation (r=rotate_60, f=flip, z=zoom, w=wiggle_around_center), '0' = off. Multiple values can be passed as comma-separated list, e.g. --augmentation-combos 0000,r000,rfzw")
    parser.add_argument("--help", "-h", action="help", help=help_text)


    # Not implemented yet
    parser.add_argument("--use-oversampling", type=_to_bool, default=None, help="Whether to oversample minority classes to address class imbalance. This is a shortcut for setting both --use-class-weights and --oversamples to the same value, which is a common pattern when addressing class imbalance. Setting this flag will override any individual settings for --use-class-weights and --oversamples.")
    parser.add_argument("--use-class-weights", type=_to_bool, default=None, help="Whether to use class weights to address class imbalance.")
    

    args = parser.parse_args(argv)

    if args.non_interactive:
        args.interactive = False
    del args.non_interactive  # Remove so it doesn't confuse update_dataclass_from_dict

    update_dataclass_from_dict(settings.project_settings, vars(args))

    if settings.project_settings.debug:
        cfg.apply_debug_overrides()

def update_dataclass_from_dict(instance: object, updates: dict, ignore_value_eq_none: bool = True) -> None:
    for key, value in updates.items():
        if hasattr(instance, key) and not (value is None and ignore_value_eq_none):
            setattr(instance, key, value)
