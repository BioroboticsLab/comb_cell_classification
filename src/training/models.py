from __future__ import annotations

import logging
import os

import tensorflow as tf
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

from src.core.config.config_utils import augmentation_code

# ImageNet channel statistics on [0, 1] pixels — Keras preprocess_input mode "torch" 
# Same values DINOv3 pretrains with (facebookresearch/dinov3, dinov3/data/transforms.py).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Type-only import (avoids a circular import with config at runtime) used by generate_wandb_checkpoint_path's signature.
    from src.core.config.config import BeeCombConfig

def _load_models_yaml(path: Path) -> dict:
    """Parse the model registry yaml; converts input-shape lists to tuples."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for entry in raw.values():
        if "default_input_shape" in entry:
            entry["default_input_shape"] = tuple(entry["default_input_shape"])
    return raw


class ModelFactory:
    """Stateless namespace for the model registry and model construction — never instantiated, all methods are called on the class itself."""
    # Shared registry, populated once from models.yaml
    MODELS: dict = {}

    @classmethod
    def load_registry(cls, models_yaml_path: Path) -> None:
        """Populate the shared MODELS registry from the config-supplied yaml path (call once at startup)."""
        cls.MODELS = _load_models_yaml(models_yaml_path)

    @classmethod
    def lookup_property_for_model(cls, name: str, property_name: str) -> Any:
        """Return ``property_name`` for model ``name`` from the shared registry."""
        if not cls.MODELS:
            raise RuntimeError(
                "ModelFactory.MODELS is empty — call "
                "ModelFactory.load_registry(cfg.paths.models_yaml_path) first."
            )
        if name not in cls.MODELS:
            raise ValueError(f"Model {name} is not supported.")
        try:
            return cls.MODELS[name][property_name]
        except KeyError:
            raise KeyError(
                f"Registry entry {name!r} has no {property_name!r} key — add it "
                "to models.yaml (see the header there for the allowed values)."
            ) from None

    @classmethod
    def is_keras_hub_model(cls, name: str) -> bool:
        """True when the registry entry is a KerasHub preset (``keras_hub_preset``
        key, e.g. the DinoV3 family) rather than a tf.keras.applications class."""
        # Reuse lookup_property_for_model's registry/name validation.
        cls.lookup_property_for_model(name, "default_input_shape")
        return "keras_hub_preset" in cls.MODELS[name]

    @classmethod
    def ensure_model_dependencies(cls, name: str) -> None:
        """Raise early when ``name`` needs keras-hub and it is missing, so a sweep fails in seconds instead of after dataset generation."""
        if not cls.is_keras_hub_model(name):
            return
        try:
            import keras_hub 
        except ImportError as exc:
            raise RuntimeError(
                f"Model {name} is a KerasHub preset but keras-hub is not installed — run `uv sync`."
            ) from exc

    @staticmethod
    def generate_run_name(cfg: "BeeCombConfig", added_manual_name: Optional[str] = None) -> str:
        """Build the run name encoding the run's hyperparameters (shared by the checkpoint dir and the W&B run)."""
        components = [cfg.training.model]

        if added_manual_name is not None:
            components.append(f"{added_manual_name}")

        components.append(augmentation_code(cfg.training.augmentation))
        components.append(f"col{cfg.dataset.cell_outer_layer}")
        components.append(f"lr{cfg.training.learning_rate}")
        if cfg.training.use_adaptive_lr:
            components.append("adap-lr")

        if cfg.training.unfreeze_after_epochs:
            components.append(f"wu{cfg.training.unfreeze_after_epochs}")
            components.append(f"ftlr{cfg.training.finetune_learning_rate}")
        elif not cfg.training.freeze_backbone:
            components.append("unfrozen")
        else:
            components.append("frozen")

        components.append(cfg.training.gray_to_rgb)

        if cfg.training.mixed_precision:
            components.append("mp16")

        if hasattr(cfg.training, "epochs"):
            components.append(f"te{cfg.training.epochs}")

        components.append(
            {(True, False): "cw",
             (False, True): "os",
             (False, False): "none",
             (True, True): "both"}[(cfg.training.use_class_weight, cfg.training.oversample_minority_classes)]
        )

        return "_".join(components)

    @staticmethod
    def generate_wandb_checkpoint_path(cfg: "BeeCombConfig", mk_dir: bool = True, include_date: bool = True, added_manual_name: Optional[str] = None) -> tuple[str, str]:
        """Generate a unique checkpoint path for the per-epoch ModelCheckpoint and the matching W&B run name, both encoding the run's hyperparameters."""
        model_root_path = cfg.paths.model_save_path

        run_name = ModelFactory.generate_run_name(cfg, added_manual_name)

        dir_name = run_name
        file_name = f"{run_name}_epoch-{{epoch:02d}}.keras"

        if include_date:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dir_name = f"{timestamp}_pid{os.getpid()}_{run_name}"

        model_dir = model_root_path / dir_name

        if mk_dir:
            model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = model_dir / file_name

        return str(checkpoint_path), run_name



    @classmethod
    def _apply_preprocess(cls, x, mode: str):
        """Map the pipeline's float32 [0, 1] pixels to the range the backbone's
        pretrained weights expect, per the registry's ``preprocess`` key."""
        if mode == "tf":
            return tf.keras.layers.Rescaling(2.0, offset=-1.0, name="normalize")(x)
        if mode == "torch":
            # normalization layer takes variance, not std
            return tf.keras.layers.Normalization(
                mean=IMAGENET_MEAN,
                variance=[s**2 for s in IMAGENET_STD],
                name="normalize",
            )(x)
        if mode == "raw255": # carry own rescaling/normalization
            return tf.keras.layers.Rescaling(255.0, name="normalize")(x)
        raise ValueError(
            f"Unknown preprocess mode {mode!r} in models.yaml — expected one of "
            "'tf', 'torch', 'raw255'."
        )

    @classmethod
    def _create_keras_hub_backbone(cls, name: str, input_shape: tuple[int, ...]) -> tf.keras.Model:
        """Build the KerasHub DINOv3 ViT backbone alone — ``create`` adds the shared head."""
        # Deferred import: keras-hub is a Linux-only dependency (see pyproject).
        # ensure_model_dependencies turns a missing package into a clear error
        # for standalone create() calls that skipped the batch_train-time check.
        cls.ensure_model_dependencies(name)
        import keras_hub

        preset = cls.lookup_property_for_model(name, "keras_hub_preset")
        h, w = input_shape[0], input_shape[1]
        # Always 3 channels — the grayscale→RGB adapter runs before the backbone.
        return keras_hub.models.DINOV3Backbone.from_preset(preset, image_shape=(h, w, 3))

    @staticmethod
    def set_backbone_trainable(model: tf.keras.Model, trainable: bool) -> None:
        """Toggle the nested backbone sub-model (tf.keras.applications base model or KerasHub DINOV3Backbone) of a classifier built by ``create``"""
        # Takes effect only AFTER a model.compile(), keras snapshots trainable-variable set when train function is built changing flag mid-fit would be silently ignored
        for layer in model.layers:
            # only backbone is nested model, rest are layers
            if isinstance(layer, tf.keras.Model):
                layer.trainable = trainable
                logger.info(
                    "Set backbone %s trainable=%s (%s params).",
                    layer.name, trainable, f"{layer.count_params():,}",
                )
                return
        raise ValueError("No nested backbone Model found — was the model built by ModelFactory.create?")

    @classmethod
    def _create_basemodel(cls, name: str, input_shape: tuple[int, ...], pretrained: bool) -> tuple[tf.keras.Model, int]:
        # lookup_property_for_model validates that `name` is registered.
        model_func = getattr(tf.keras.applications, cls.lookup_property_for_model(name, "model"))

        # Pretrained ImageNet models require 3-channel input.
        base_channels = input_shape[-1] if len(input_shape) == 3 else 3
        base_input_shape = (*input_shape[:-1], 3) if base_channels == 1 else input_shape

        model = model_func(include_top=False, weights="imagenet" if pretrained else None, input_shape=base_input_shape)
        return model, base_channels
    
    @staticmethod
    def _set_batchnorm_momentum(base_model: tf.keras.Model, momentum: float) -> None:
        """Override BatchNorm momentum on every BN layer of a pretrained backbone."""
        n = 0
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = momentum
                n += 1
        logger.info("Set BatchNorm momentum=%.4g on %d layer(s) of backbone.", momentum, n)

    @staticmethod
    def _adapt_gray_to_rgb(x, mode: str):
        """Map the 1-channel input to the 3 channels the pretrained backbone expects, per ``training.gray_to_rgb``."""
        if mode == "replicate":
            return tf.keras.layers.Concatenate(axis=-1, name="gray_to_rgb")([x, x, x])
        if mode == "conv1x1":
            # Initialized to exact replication (w=1, b=0), means trainable variant starts identical to "replicate" and only then adapts
            return tf.keras.layers.Conv2D(3, (1, 1), padding="same", name="gray_to_rgb", kernel_initializer="ones", bias_initializer="zeros")(x)
        raise ValueError(f"training.gray_to_rgb must be 'replicate' or 'conv1x1', got {mode!r}")

    @classmethod
    def create(cls, name: str, input_shape: tuple[int, ...], num_of_classes: int, batchnorm_momentum: Optional[float] = None, freeze_backbone: bool = True, gray_to_rgb: str = "replicate") -> tf.keras.Model:
        is_keras_hub = cls.is_keras_hub_model(name)
        
        if is_keras_hub:
            backbone = cls._create_keras_hub_backbone(name, input_shape)
            base_channels = input_shape[-1] if len(input_shape) == 3 else 3
        else:
            backbone, base_channels = cls._create_basemodel(name, input_shape, pretrained=True)
            if batchnorm_momentum is not None:
                cls._set_batchnorm_momentum(backbone, batchnorm_momentum)

        # Frozen backbone, so train only fresh head
        backbone.trainable = not freeze_backbone
        logger.info("Built %s (%s backbone, %s params).", name, "frozen" if freeze_backbone else "trainable", f"{backbone.count_params():,}")

        model_input = tf.keras.layers.Input(shape=input_shape, name="input")
        x = model_input

        if base_channels == 1:
            x = cls._adapt_gray_to_rgb(x, gray_to_rgb)

        x = cls._apply_preprocess(x, cls.lookup_property_for_model(name, "preprocess"))


        ### Model head
        if is_keras_hub:
            preset = cls.lookup_property_for_model(name, "keras_hub_preset")
            
            if preset.startswith("dinov3"):
                tokens = backbone({"pixel_values": x})  # (batch, 1 CLS + registers + H/16*W/16 patches, hidden)
                x = tokens[:, 0, :] # just the CLS-Token
            else:
                raise ValueError(
                    f"Model {name!r} uses KerasHub preset {preset!r}, but only DINOv3 family is supported (CLS-token head) yet. "
                    f"Add a branch for it in {__file__}."
                )
        else:
            x = backbone(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x) # collapses the CNN's spatial feature map to one vector per image


        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        logits = tf.keras.layers.Dense(num_of_classes, dtype='float32', name='logits')(x)
        model = tf.keras.Model(inputs=model_input, outputs=logits)

        return model


def sync_roi_to_model(cfg: "BeeCombConfig") -> None:
    """Resize ``cfg.dataset.roi_size_tuple`` to match the current model's default input shape. Writes in place."""
    # Must run before dataset generation so pipeline tensors match the backbone (otherwise gradient all-reduce fails with a shape mismatch).
    h, w, _ = ModelFactory.lookup_property_for_model(name=cfg.training.model, property_name="default_input_shape")
    cfg.dataset.roi_size_tuple = (h, w)


def sync_batch_size_to_model(cfg: "BeeCombConfig") -> None:
    """Overwrite ``cfg.dataset.batch_size`` with the model's recommended per-replica batch (no-op when ``auto_batch_size`` is False). Writes in place."""
    if not cfg.dataset.auto_batch_size:
        return
    cfg.dataset.batch_size = ModelFactory.lookup_property_for_model(name=cfg.training.model, property_name="recommended_batch_size")
