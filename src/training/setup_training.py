import logging
from typing import Optional

from src.training.models import ModelFactory
from src.training.argparsing import parse_args_to_project
from src.core.config.config import BeeCombConfig

def setup_training_project(cfg: BeeCombConfig, argv: Optional[list[str]] = None) -> None:
    """Use once at startup for configuring logging, applying CLI overrides to ``cfg`` in place and load the model registry."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parse_args_to_project(cfg, argv)
    ModelFactory.load_registry(cfg.paths.models_yaml_path)