"""Inference crops (``build_crops``) must equal, byte for byte, what the model saw in training: producer crop -> TFRecord example -> ``parse_fn``."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import tensorflow as tf

from src.core.annotations import Annotation, AnnotationDoc
from src.core.config.config import BeeCombConfig
from src.inference.inference import build_crops
from src.training.dataset.producer import BeeCombDatasetProducer
from src.training.dataset.utils import _dataset_generation_worker, make_parse_fn

# Centre cells, cells whose crop needs border padding, and a second radius (a second resize group).
CELLS = [(300, 300, 24), (410, 170, 24), (20, 300, 24), (590, 585, 24), (250, 450, 26)]


def _training_crops(image_path, annotations, cfg, h, w):
    """Run each annotation through the real training path: dataset worker, tf.train.Example, parse_fn."""
    doc = AnnotationDoc(path=image_path.with_suffix(".json"), annotations=annotations)
    parse_fn = make_parse_fn(image_size=(h, w))
    crops = []
    for crop, label in _dataset_generation_worker((image_path, doc), cfg):
        example = BeeCombDatasetProducer._make_tf_example(crop, label, 0)
        image, _ = parse_fn(tf.constant(example.SerializeToString()))
        crops.append(image.numpy())
    return np.stack(crops)


@pytest.mark.parametrize("col", [0, 1])
@pytest.mark.parametrize("size", [224, 299])
def test_build_crops_matches_training_pipeline(tmp_path, col, size):
    rng = np.random.default_rng(col * 1000 + size)
    # Smooth random texture, so both CLAHE and the bilinear resize have real work to do.
    img = cv2.resize(rng.integers(0, 256, (60, 60), dtype=np.uint8), (600, 600), interpolation=cv2.INTER_CUBIC)
    image_path = tmp_path / "comb.png"
    cv2.imwrite(str(image_path), img)

    cfg = BeeCombConfig()
    cfg.dataset.cell_outer_layer = col
    annotations = [Annotation(id=str(i), center_x=x, center_y=y, radius=r, label="empty_cell") for i, (x, y, r) in enumerate(CELLS)]

    expected = _training_crops(image_path, annotations, cfg, size, size)
    actual = build_crops(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE), annotations, cfg, size, size, 1)

    assert actual.shape == (len(CELLS), size, size, 1)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
