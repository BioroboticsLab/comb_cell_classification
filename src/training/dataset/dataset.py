import logging
from collections.abc import Iterable
from functools import reduce
from pathlib import Path
from typing import Literal, Optional, Union

import tensorflow as tf
import yaml

from src.core.config.config import BeeCombConfig
from src.training.dataset.utils import make_parse_fn

logger = logging.getLogger(__name__)


class BeeCombDataset:
    """A ``tf.data``-backed classification dataset built from a directory of per-class TFRecord shards written by ``BeeCombDatasetProducer.toTFRecord``."""
    def __init__(self, root_path: Union[str, Path], cfg: Optional[BeeCombConfig] = None, transform: Optional[tf.keras.Sequential] = None) -> None:
        """Init from *root_path* (per-class TFRecord dir), with optional *cfg* and a train-only augmentation *transform*."""
        self.cfg = cfg
        self._transform = transform
        self.class_names: Optional[list[str]] = None
        self._samples_per_class: Optional[dict[str, int]] = None
        self._class_weights: Optional[dict[int, float]] = None
        self._per_class_streams: Optional[dict[str, tf.data.Dataset]] = None

        self.root_path = Path(root_path)
        self._load_from_path(self.root_path)

        if self.cfg.dataset.remap_to_other:
            self._remap_labels_to_other()

    @property
    def _num_channels(self) -> int:
        """Colour channels from cfg (default 3)."""
        return getattr(self.cfg.dataset, "num_channels", 3)

    def _load_from_path(self, root: Path) -> None:
        """Load a per-class TFRecord dataset dir (``properties.yaml`` + one or more shards per class, as written by ``toTFRecord``)."""
        self.properties_path = root / "properties.yaml"
        if not root.is_dir() or not self.properties_path.exists():
            raise ValueError(
                f"root_path must be a dataset directory containing properties.yaml: {root}"
            )
        props = self._load_properties_yaml(self.properties_path)
        if props.get("layout") != "per_class":
            raise ValueError(
                f"Unsupported dataset layout {props.get('layout')!r} at {root} — "
                "only 'per_class' (toTFRecord output) is supported."
            )
        self._update_dataset_from_properties_file(props)
        self._per_class_streams = self._load_per_class_streams(root)

    @staticmethod
    def _read_tfrecords(files: list[Path]) -> tf.data.Dataset:
        """Read one or many TFRecord shards (multi-shard via parallel interleave so reads don't bottleneck on a single thread)."""
        if len(files) == 1:
            return tf.data.TFRecordDataset(str(files[0]))
        file_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in files])
        return file_ds.interleave(
            lambda f: tf.data.TFRecordDataset(f),
            cycle_length=len(files),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    def _list_class_files(self, root: Path, cls: str) -> list[Path]:
        """TFRecord file(s) for one class: single ``{cls}.tfrecord`` or sharded ``{cls}-NNNNN-of-MMMMM.tfrecord`` (explicit patterns avoid prefix clashes)."""
        single = root / f"{cls}.tfrecord"
        if single.exists():
            return [single]
        return sorted(root.glob(f"{cls}-*-of-*.tfrecord"))

    def _load_per_class_streams(self, root: Path) -> dict[str, tf.data.Dataset]:
        """Build one parsed ``tf.data.Dataset`` per class."""
        if self.class_names is None:
            raise ValueError("class_names must be loaded before per-class streams.")

        files_by_class = {c: self._list_class_files(root, c) for c in self.class_names}
        parse_fn = make_parse_fn(image_size=self.cfg.dataset.roi_size_tuple)

        streams: dict[str, tf.data.Dataset] = {}
        for cls, files in files_by_class.items():
            if not files:
                raise FileNotFoundError(
                    f"No TFRecord file(s) found for class '{cls}' under {root}"
                )
            streams[cls] = self._read_tfrecords(files).map(
                parse_fn, num_parallel_calls=tf.data.AUTOTUNE
            )
        return streams

    def _update_dataset_from_properties_file(self, properties: dict) -> None:
        """Set class_names and sample counts from a properties dict."""
        if not isinstance(properties, dict):
            raise TypeError(f"Expected a dict, got {type(properties)}")
        labels = properties.get("labels")
        if not isinstance(labels, list):
            raise ValueError(
                f"'labels' must be a list in properties at {self.properties_path}, "
                f"got {type(labels)}"
            )
        self.class_names = labels
        counts = properties.get("samples_per_class", {})
        self._samples_per_class = {name: counts.get(name, 0) for name in labels}

    @staticmethod
    def _load_properties_yaml(root_path: Path) -> dict:
        """Load ``properties.yaml`` next to *root_path* (dir or .tfrecord file)."""
        directory = root_path.parent if root_path.is_file() else root_path
        with open(directory / "properties.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(self.class_names)

    @property
    def num_samples_per_class(self) -> dict[str, int]:
        """Per-class sample counts (from ``properties.yaml``)."""
        return self._samples_per_class

    @property
    def num_total_size(self) -> int:
        """Total sample count across all classes."""
        return sum(self.num_samples_per_class.values())

    @property
    def class_weights(self) -> dict[int, float]:
        """Class-index → weight, inversely proportional to class frequency."""
        if self._class_weights is None:
            logger.info("Compute class weights...")
            total, n = self.num_total_size, self.num_classes
            self._class_weights = {idx: total / (n * count) if count > 0 else 0.0 for idx, count in enumerate(self.num_samples_per_class.values())}
        return self._class_weights

    @staticmethod
    def _build_label_lookup(
        original_class_names: list[str], remap_set: set[str], remap_class_name: str
    ) -> tuple[tf.Tensor, list[str]]:
        """Build an old-index → new-index lookup tensor (for ``tf.gather``) that collapses *remap_set* into one *remap_class_name* class, plus new names."""
        new_class_names: list[str] = []
        mapping: list[int] = []
        for class_name in original_class_names:
            new_name = remap_class_name if class_name in remap_set else class_name
            if new_name not in new_class_names:
                new_class_names.append(new_name)
            mapping.append(new_class_names.index(new_name))
        return tf.constant(mapping, dtype=tf.int32), new_class_names

    def _remap_labels_to_other(self) -> None:
        """Collapse cfg-selected classes into a combined 'other' class across the per-class streams, sample counts, and weights."""
        logger.info(
            "Remapping classes %s to '%s' in the dataset",
            self.cfg.dataset.remap_classes_set,
            self.cfg.dataset.remap_class_name,
        )
        if self.class_names is None:
            raise ValueError("class_names must be initialized before remapping labels.")

        remap_set = self.cfg.dataset.remap_classes_set
        combined_name = self.cfg.dataset.remap_class_name
        lookup, new_class_names = self._build_label_lookup(
            self.class_names, remap_set, combined_name
        )
        remap = lambda x, y: (x, tf.gather(lookup, tf.cast(y, tf.int32)))

        remapped_counts: dict[str, int] = {}
        for name, count in self.num_samples_per_class.items():
            key = combined_name if name in remap_set else name
            remapped_counts[key] = remapped_counts.get(key, 0) + count

        self.class_names = new_class_names
        self._samples_per_class = {n: remapped_counts.get(n, 0) for n in new_class_names}
        self._class_weights = None  # recompute with new counts

        # Rebuild streams under the new class set so split() stays stratified.
        new_streams: dict[str, tf.data.Dataset] = {}
        for old_name, old_stream in self._per_class_streams.items():
            new_name = combined_name if old_name in remap_set else old_name
            remapped = old_stream.map(remap, num_parallel_calls=tf.data.AUTOTUNE)
            new_streams[new_name] = (
                remapped
                if new_name not in new_streams
                else new_streams[new_name].concatenate(remapped)
            )
        self._per_class_streams = new_streams

    @staticmethod
    def _concatenate(streams: Iterable[tf.data.Dataset]) -> tf.data.Dataset:
        """Concatenate an iterable of datasets into one."""
        return reduce(lambda a, b: a.concatenate(b), streams)

    def split(
        self,
        validation_split: float,
        eval_split: float,
        seed: int = 1,
        shuffle_training: bool = True,
        batch_size: int = 0,
        max_shuffle_samples: int = 5000,
        prefetch_size: int = tf.data.AUTOTUNE,
        oversample: bool = False,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Lazily split into (train, val, eval) streams where the augmentation transform hits train only and all splits are normalized to [0, 1], optionally batched, and prefetched."""
        if abs(eval_split + validation_split) >= 1:
            raise ValueError(
                f"validation_split ({validation_split}) + eval_split ({eval_split}) "
                f"must be < 1.0, got {eval_split + validation_split:.4f}."
            )

        return self._split_per_class(
            validation_split=validation_split,
            eval_split=eval_split,
            seed=seed,
            shuffle_training=shuffle_training,
            batch_size=batch_size,
            max_shuffle_samples=max_shuffle_samples,
            prefetch_size=prefetch_size,
            oversample=oversample
        )

    def _split_per_class(
        self,
        validation_split: float,
        eval_split: float,
        seed: int,
        shuffle_training: bool,
        batch_size: int,
        max_shuffle_samples: int,
        prefetch_size: int,
        oversample: bool,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Stratified split for per-class shards. Each class split exactly via take/skip, train assembled with ``sample_from_datasets`` (natural or balanced-oversampled weights) — no whole-set filter pass."""
        if self._per_class_streams is None or self.class_names is None:
            raise RuntimeError("_split_per_class called without per-class streams")

        counts = self.num_samples_per_class
        total = sum(counts.values())
        if total == 0:
            raise ValueError("Dataset is empty. Verify DATASET_PATH and dataset loading.")

        train_per_class: dict[str, tf.data.Dataset] = {}
        val_per_class: dict[str, tf.data.Dataset] = {}
        eval_per_class: dict[str, tf.data.Dataset] = {}
        train_n: dict[str, int] = {}

        for class_idx, cls in enumerate(self.class_names):
            n = counts[cls]
            if n == 0:
                continue
            cls_eval_n = int(round(n * eval_split))
            cls_val_n = int(round(n * validation_split))
            cls_train_n = n - cls_val_n - cls_eval_n
            train_n[cls] = cls_train_n

            # Deterministic per-class shuffle (seed+idx decorrelates orderings) so train/val/eval boundaries are stable across epochs.
            stream = self._per_class_streams[cls].shuffle(
                buffer_size=min(n, max_shuffle_samples),
                seed=seed + class_idx,
                reshuffle_each_iteration=False,
            )
            train_per_class[cls] = stream.take(cls_train_n)
            val_per_class[cls] = stream.skip(cls_train_n).take(cls_val_n)
            eval_per_class[cls] = stream.skip(cls_train_n + cls_val_n)

        class_order = [c for c in self.class_names if counts[c] > 0]

        val_total = sum(int(round(counts[c] * validation_split)) for c in class_order)
        eval_total = sum(int(round(counts[c] * eval_split)) for c in class_order)
        logger.info("Stratified split (per-class) over %s samples to train: %s  val: %s  eval: %s", total, sum(train_n.values()), val_total, eval_total)

        if oversample:
            # Balanced oversampling: infinite per-class streams drawn with equal probability 1/C, so minorities repeat until batches are balanced.
            train_streams = [train_per_class[c].repeat() for c in class_order]
            weights = [1.0 / len(class_order)] * len(class_order)
            logger.info("Oversampling enabled: balanced draw with p=%.3f per class (C=%s)", weights[0], len(class_order))
        else:
            # Draw each class proportionally to its size, reproducing an unstratified shuffle of the whole train set.
            train_streams = [train_per_class[c] for c in class_order]
            class_total = sum(train_n[c] for c in class_order)
            weights = [train_n[c] / class_total for c in class_order]

        train_ds = tf.data.Dataset.sample_from_datasets( train_streams, weights=weights, seed=seed)

        # Small shuffle distributes classes with fewer samples equally over batches
        if shuffle_training:
            train_ds = train_ds.shuffle(buffer_size=min(2_000, sum(train_n.values())), seed=seed)

        val_ds = self._concatenate(val_per_class[c] for c in class_order)
        eval_ds = self._concatenate(eval_per_class[c] for c in class_order)

        return self._finalize_splits(train_ds, val_ds, eval_ds, batch_size, prefetch_size)

    def _finalize_splits(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        eval_ds: tf.data.Dataset,
        batch_size: int,
        prefetch_size: int,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Apply transform (train only), normalize to [0,1], batch, prefetch."""
        if self._transform is not None:
            train_ds = train_ds.map(
                lambda x, y: (self._transform(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # uint8 [0, 255] to float32 [0, 1]. The models own Rescaling/Normalization layer maps onward to what each backbone expects.
        normalize = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
        train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        eval_ds = eval_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

        if batch_size > 0:
            logger.info("Batching splits with batch_size=%s...", batch_size)
            train_ds = train_ds.batch(batch_size, drop_remainder=False)
            val_ds = val_ds.batch(batch_size, drop_remainder=False)
            eval_ds = eval_ds.batch(batch_size, drop_remainder=False)

        train_ds = train_ds.prefetch(prefetch_size)
        val_ds = val_ds.prefetch(prefetch_size)
        eval_ds = eval_ds.prefetch(prefetch_size)

        logger.info("Done.")
        return train_ds, val_ds, eval_ds
