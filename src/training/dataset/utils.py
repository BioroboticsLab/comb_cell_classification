from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Union
import multiprocessing
import os
import threading
import tensorflow as tf

import cv2
import numpy as np
from tqdm import tqdm
from src.core.config.config import BeeCombConfig
from src.core.annotations import AnnotationDoc
from src.core.image_processing import crop_cell
from pathlib import Path
from multiprocessing.sharedctypes import Synchronized

_shared_counter: Optional[Synchronized[int]] = None

def _init_worker(counter: Synchronized[int]) -> None:
    """Pool initialiser that stores the shared progress counter in each worker forces to use CPU by hiding GPUs"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    global _shared_counter
    _shared_counter = counter

def _dataset_generation_worker(image_ann_pair: tuple[Path, AnnotationDoc], cfg: BeeCombConfig) -> list[tuple[np.ndarray, str]]:
    """Worker that crops and preprocesses every annotated cell of one image, returning (sample, label) pairs."""
    global _shared_counter
    image_path, anns = image_ann_pair
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    preprocessed_samples = []
    for ann in anns.annotations:
        preprocessed_img = crop_cell(img, ann, cfg)
        # label_merge renames at crop time, so merged cells land in their target class's shards and get split like any other cell of that class.
        preprocessed_samples.append((preprocessed_img, cfg.dataset.label_merge.get(ann.label, ann.label)))

        if _shared_counter is not None:
            with _shared_counter.get_lock():
                _shared_counter.value += 1

    return preprocessed_samples

def max_num_of_threads(manual_max_num_of_threads: int = -1) -> int:
    """Determine the maximum number of multiprocessing workers from the CPU core count and an optional manual limit."""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    if manual_max_num_of_threads > 0:
        return min(cpu_count - 1, manual_max_num_of_threads)
    return cpu_count - 1 

def create_dataset(image_ann_pairs: list[tuple[Path, AnnotationDoc]], cfg: BeeCombConfig) -> tuple[list[tuple[np.ndarray, str]], dict[str, int]]:
    """Process all image/annotation pairs into a list of (sample, label) tuples plus per-class sample counts, using CPU multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor

    n = len(image_ann_pairs)
    total_annotations = sum(len(anns.annotations) for _, anns in image_ann_pairs)
    complete_dataset = []

    mp_ctx = multiprocessing.get_context("spawn")  # "spawn" gives each child a fresh interpreter, so workers never inherit the parent's TensorFlow/CUDA GPU context.
    num_workers = min(max_num_of_threads(), n)

    counter = mp_ctx.Value('i', 0)
    pbar = tqdm(total=total_annotations, desc="Creating dataset (CPU workers)", unit="sample", dynamic_ncols=True)
    stop_event = threading.Event()

    def _poll_progress() -> None:
        while not stop_event.is_set():
            pbar.n = counter.value
            pbar.refresh()
            stop_event.wait(0.1)
        # Final sync
        pbar.n = counter.value
        pbar.refresh()

    poll_thread = threading.Thread(target=_poll_progress, daemon=True)
    poll_thread.start()

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx, initializer=_init_worker, initargs=(counter,)) as executor:
        results = list(executor.map(_dataset_generation_worker, image_ann_pairs, [cfg]*n))

    stop_event.set()
    poll_thread.join()
    pbar.close()

    samples_per_class: dict[str, int] = {}
    for preprocessed_samples in results:
        for sample in preprocessed_samples:
            label = sample[1]
            samples_per_class[label] = samples_per_class.get(label, 0) + 1
        complete_dataset.extend(preprocessed_samples)

    return complete_dataset, samples_per_class


def make_parse_fn(image_size: tuple[int, int]) -> Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """Return a ``tf.data``-mappable function that parses one serialised TFRecord example (raw uint8 layout written by ``toTFRecord``) into a resized (image, label) pair."""
    h, w = image_size

    feature_spec = {
        "image/encoded":     tf.io.FixedLenFeature([], tf.string),
        "image/height":      tf.io.FixedLenFeature([], tf.int64),
        "image/width":       tf.io.FixedLenFeature([], tf.int64),
        "image/channels":    tf.io.FixedLenFeature([], tf.int64),
        "image/label_index": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_fn(serialised: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        parsed = tf.io.parse_single_example(serialised, feature_spec)
        stored_h = tf.cast(parsed["image/height"], tf.int32)
        stored_w = tf.cast(parsed["image/width"], tf.int32)
        stored_c = tf.cast(parsed["image/channels"], tf.int32)
        image = tf.io.decode_raw(parsed["image/encoded"], tf.uint8)
        image = tf.reshape(image, [stored_h, stored_w, stored_c])
        image = resize_to_input(image, h, w)
        label = tf.cast(parsed["image/label_index"], tf.int32)
        return image, label

    return parse_fn


def resize_to_input(image: Union[tf.Tensor, np.ndarray], h: int, w: int) -> tf.Tensor:
    """Resize a uint8 crop, or a batch of equally sized crops, to the model input as training does: bilinear ``tf.image.resize``, then a truncating cast back to uint8."""
    return tf.cast(tf.image.resize(image, [h, w]), tf.uint8)