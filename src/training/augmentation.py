from __future__ import annotations

import logging
import math

import tensorflow as tf

from src.core.config.config import BeeCombConfig

logger = logging.getLogger(__name__)


def _rotation_transforms(angles, height, width):
    """Build the 8-param affine transforms `tf.raw_ops.ImageProjectiveTransformV3` expects, one row per angle."""
    cos_a, sin_a = tf.cos(angles), tf.sin(angles)
    x_offset = ((width - 1) - (cos_a * (width - 1) - sin_a * (height - 1))) / 2.0
    y_offset = ((height - 1) - (sin_a * (width - 1) + cos_a * (height - 1))) / 2.0
    zeros = tf.zeros_like(angles)
    return tf.stack([cos_a, -sin_a, x_offset, sin_a, cos_a, y_offset, zeros, zeros], axis=1)


class DiscreteRotation(tf.keras.layers.Layer):
    """Rotates each image by a random multiple of `step_degrees` (n in [0, num_steps))."""
    def __init__(self, step_degrees: float = 60, num_steps: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.step_radians = step_degrees * (math.pi / 180)
        self.num_steps = num_steps

    def call(self, images, training=True):
        if not training:
            return images
        unbatched = images.shape.rank == 3 # ImageProjectiveTransformV3 requires a rank-4 [batch, H, W, C] tensor, but the dataset applies augmentation per image before batching -> wrapping image into batch of 1 and unwrap result
        if unbatched:
            images = images[tf.newaxis, ...]
        batch_size = tf.shape(images)[0]
        height = tf.cast(tf.shape(images)[1], tf.float32)
        width = tf.cast(tf.shape(images)[2], tf.float32)
        n = tf.random.uniform((batch_size,), 0, self.num_steps, dtype=tf.int32)
        angles = tf.cast(n, tf.float32) * self.step_radians
        rotated = tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            transforms=_rotation_transforms(angles, height, width),
            output_shape=tf.shape(images)[1:3],
            fill_value=0.0,
            interpolation="BILINEAR",
            fill_mode="REFLECT",
        )
        return rotated[0] if unbatched else rotated


def data_augmentation_transform(cfg: BeeCombConfig) -> tf.keras.Sequential | None:
    """Build the train-only augmentation pipeline from the augmentation names listed in the config."""
    transform_list = []

    augmentations = cfg.training.augmentation if cfg.training.augmentation is not None else []
    if not augmentations:
        return None

    aug_cfg = cfg.augmentation
    for aug in augmentations:
        match aug:
            case 'flip':
                transform_list.append(tf.keras.layers.RandomFlip(aug_cfg.flip_mode))
            case 'rotate_60':
                transform_list.append(DiscreteRotation(step_degrees=aug_cfg.rotate_step_degrees, num_steps=aug_cfg.rotate_num_steps))
            case 'zoom':
                transform_list.append(tf.keras.layers.RandomZoom(height_factor=aug_cfg.zoom_factor, width_factor=None))
            case 'wiggle_around_center':
                transform_list.append(tf.keras.layers.RandomTranslation(height_factor=aug_cfg.wiggle_factor, width_factor=aug_cfg.wiggle_factor))
            case _:
                logger.warning("Transform '%s' is not implemented yet. Skipping.", aug)

    return tf.keras.Sequential(transform_list)
