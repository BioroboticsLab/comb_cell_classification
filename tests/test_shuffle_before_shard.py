"""``dataset.shuffle_before_shard`` decides whether a class is sharded in image order or in a seeded random order.

Without it, ``BeeCombDataset.split`` cuts each class by position, so with two sources (say one camera's images sorted
after another's) the second source lands entirely in val/eval. With it, both sources are spread across the splits.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import tensorflow as tf

from src.core.annotations import Annotation, AnnotationDoc
from src.core.config.config import BeeCombConfig
from src.training.dataset.producer import BeeCombDatasetProducer
from src.training.dataset.utils import make_parse_fn

IMAGES = 6            # "sources" are just the first and last three, as file names sort
CELLS_PER_IMAGE = 40


def _fixture(tmp_path):
    """One flat-grey image per source image, so every crop taken from it carries that image's grey value as its id."""
    for i in range(IMAGES):
        stem = f"img{i}"
        cv2.imwrite(str(tmp_path / f"{stem}.png"), np.full((400, 400), 30 + 20 * i, dtype=np.uint8))
        cells = [Annotation(id=f"{i}-{c}", center_x=60 + 40 * (c % 7), center_y=60 + 40 * (c // 7), radius=24, label="empty_cell")
                 for c in range(CELLS_PER_IMAGE)]
        AnnotationDoc(path=tmp_path / f"{stem}.json", annotations=cells).save()


def _cfg(tmp_path, shuffle: bool, out="dataset") -> BeeCombConfig:
    cfg = BeeCombConfig()
    cfg.paths.raw_data_path = tmp_path
    cfg.paths.raw_annotations_path = tmp_path
    cfg.paths.produced_dataset_save_path = tmp_path / out
    cfg.dataset.apply_clahe = False      # keep each crop flat, so its grey value still identifies the image
    cfg.dataset.shuffle_before_shard = shuffle
    return cfg


def _stream_image_ids(save_path) -> list[int]:
    """The grey value of every stored crop, in the order the split would read them."""
    parse = make_parse_fn(image_size=(64, 64))
    files = sorted(save_path.glob("empty_cell*.tfrecord"))
    return [int(parse(record)[0].numpy()[32, 32, 0]) for f in files for record in tf.data.TFRecordDataset(str(f))]


@pytest.mark.parametrize("shuffle", [False, True])
def test_cache_name_marks_shuffling(tmp_path, shuffle):
    name = BeeCombDatasetProducer(tmp_path, _cfg(tmp_path, shuffle)).readable_custom_name
    assert ("shuffled" in name) is shuffle


def test_unshuffled_splits_by_image_and_shuffled_mixes(tmp_path):
    _fixture(tmp_path)
    order = {}
    for shuffle in (False, True):
        producer = BeeCombDatasetProducer(tmp_path, _cfg(tmp_path, shuffle, out=f"dataset_{shuffle}"))
        order[shuffle] = _stream_image_ids(producer.toTFRecord())

    total = IMAGES * CELLS_PER_IMAGE
    assert len(order[False]) == total
    tail = slice(int(total * 0.8), None)     # the 20% that split() hands to val + eval
    # Unshuffled: the tail is the last image or two — with two sources, the later-sorted one is the whole val/eval set.
    assert len(set(order[False][tail])) <= 2
    # Shuffled: the tail draws on most images instead.
    assert len(set(order[True][tail])) >= IMAGES - 1
    # Same seed, same order.
    repeat = _stream_image_ids(BeeCombDatasetProducer(tmp_path, _cfg(tmp_path, True, out="dataset_repeat")).toTFRecord())
    assert repeat == order[True]
