import logging
import math
import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

from src.core.config.config import BeeCombConfig
from src.core.annotations import AnnotationDoc
from src.core.annotations.labels import write_label_map
import src.training.utils as utils
import src.training.dataset.properties as dataset_properties

from src.training.dataset.utils import create_dataset

logger = logging.getLogger(__name__)

class BeeCombDatasetProducer:
    """Produces a dataset of per-cell image crops from annotated comb images and writes it to disk or memory."""

    def __init__(self, root_path: Union[str, Path], cfg: BeeCombConfig) -> None:
        """Collect all images under ``root_path`` and pair them with their annotation files."""
        self.root_path = Path(root_path)
        self.cfg = cfg
        self._image_path_list = self._load_file_paths(
            path=self.root_path,
            suffix_filter=self.cfg.import_config.image_suffixes,
            suffix_split=",",
        )

        self.image_ann_pairs = self._create_image_annotation_pairs()


    @property
    def readable_custom_name(self) -> str:
        """Generate a human-readable dataset name that encodes the generation parameters, e.g. ``outerlayers-2_clahe-False_fixed-r24``."""
        name_parts = [
            f"outerlayers-{self.cfg.dataset.cell_outer_layer}",
            f"clahe-{self.cfg.dataset.apply_clahe}",
            f"fixed-r{self.cfg.dataset.fixed_radius}" if self.cfg.dataset.fixed_radius > 0 else "fixed-rNone",
        ]
        # Both of these change the stored dataset, so each names its own cache (and leaves the default name unchanged).
        if self.cfg.dataset.shuffle_before_shard:
            name_parts.append("shuffled")
        if self.cfg.dataset.label_merge:
            name_parts.append("merge-" + "+".join(f"{src}-to-{dst}" for src, dst in sorted(self.cfg.dataset.label_merge.items())))
        return "_".join(name_parts).rstrip()

    def _create_image_annotation_pairs(self) -> list[tuple[Path, AnnotationDoc]]:
        """Pair each image path with its loaded annotations, matched via the same-named annotation file."""
        pairs = []
        for image_path in self._image_path_list:
            annotation_file = image_path.stem + self.cfg.import_config.annotation_suffix
            annotation_path = utils.resolve_to_root_path(self.cfg.paths.raw_annotations_path) / annotation_file
            if annotation_path.exists():
                anns = AnnotationDoc.load(annotation_path)
                pairs.append((image_path, anns))
            else:
                logger.warning(
                    "No annotation file found for %s. Expected at %s. Skipping this image.",
                    image_path, annotation_path,
                )
        return pairs

    def _load_file_paths(self, path: Path, suffix_filter: str, suffix_split: str = ",") -> list[Path]:
        """Recursively collect file paths under ``path`` whose extension matches one of the suffixes in ``suffix_filter``."""
        file_paths = []
        suffixes = utils.get_suffixes_as_tuple(suffix_filter, suffix_split)

        for root, _, files in os.walk(path):
            for filename in files:
                if utils.check_extension(filename, suffixes):
                    full_path = os.path.join(root, filename)
                    file_paths.append(Path(full_path))
        if self.cfg.import_config.sort_paths:
            file_paths.sort()
        return file_paths

    @staticmethod
    def _make_tf_example(
        image_data: np.ndarray,
        label_string: str,
        label_index: int,
    ) -> tf.train.Example:
        """Serialise a single (image, label) pair into a tf.train.Example."""
        if not isinstance(image_data, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(image_data)}")
        if image_data.ndim == 2:
            image_data = image_data[..., np.newaxis]
        h, w, c = image_data.shape
        img_bytes = image_data.tobytes()
        fmt = b"raw"

        feature = {
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[img_bytes])
            ),
            "image/format": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[fmt])
            ),
            "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            "image/channels": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[c])
            ),
            "image/label_index": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label_index])
            ),
            "image/label_string": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[label_string.encode("utf-8")])
            ),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def toTFRecord(
        self,
        save_path: Optional[Union[Path, str]] = None,
        samples_per_shard_per_class: int = 20_000,
        force_cpu: bool = True,
        skip_when_exists: bool = True,
    ) -> Path:
        """Generate the dataset and write it as one or more TFRecord shards per class, plus a label map and ``properties.yaml``."""
        def _fmt_size(b: int) -> str:
            if b >= 1 << 30:
                return f"{b / (1 << 30):.2f} GB"
            return f"{b / (1 << 20):.1f} MB"
        
        save_path = self.cfg.paths.produced_dataset_save_path / self.readable_custom_name

        # gets written as last step, so reliable for checking
        if skip_when_exists and (save_path / "properties.yaml").exists():
            logger.info("Dataset for %s already exists at %s. Skipping generation.", self.readable_custom_name, save_path)
            return save_path
        
        save_path.mkdir(parents=True, exist_ok=True)

        dataset, samples_per_class = create_dataset(self.image_ann_pairs, self.cfg)
        if not dataset:
            logger.warning("dataset is empty, nothing to write.")
            return

        # Sorted labels give a deterministic label→index mapping across runs.
        unique_labels = sorted({label for _, label in dataset})
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        label_map_path = save_path / f"{self.readable_custom_name}_label_map.txt"
        write_label_map(unique_labels, label_map_path)
        logger.info("Label map (%s classes) saved to %s", len(unique_labels), label_map_path)

        indexed_dataset = [(img, label, label_to_index[label]) for img, label in dataset]

        layout = "per_class"
        shards: list[list[tuple[Union[np.ndarray, bytes], str, int]]] = []
        shard_paths: list[str] = []

        # Group samples by class label. Iterate unique_labels (sorted) so the shard order is deterministic and matches label_to_index.
        samples_by_class: dict[str, list[tuple[Union[np.ndarray, bytes], str, int]]] = {cls: [] for cls in unique_labels}
        
        for sample in indexed_dataset:
            samples_by_class[sample[1]].append(sample)

        if self.cfg.dataset.shuffle_before_shard:
            # Shard order decides the split: BeeCombDataset.split takes the first 80% / next 10% / last 10% of each
            # class stream, and its shuffle buffer (5000) is far smaller than a large class, so an unshuffled class is
            # split by image order — the last images by file name become val/eval. Harmless for one homogeneous source,
            # wrong when the dataset mixes sources (one source then lands entirely in val/eval). Seeded, so runs repeat.
            rng = random.Random(self.cfg.dataset.seed)
            for class_samples in samples_by_class.values():
                rng.shuffle(class_samples)
            logger.info("Shuffled each class before sharding (seed %d), so the train/val/eval split is random.", self.cfg.dataset.seed)

        for class_name in unique_labels:
            class_samples = samples_by_class[class_name]
            n = len(class_samples)
            if n == 0:
                continue
            n_shards_this_class = max(1, math.ceil(n / samples_per_shard_per_class))

            class_shards: list[list[tuple[Union[np.ndarray, bytes], str, int]]] = [
                [] for _ in range(n_shards_this_class)
            ]
            # keeps shard-parts evenly spaced
            for i, sample in enumerate(class_samples):
                class_shards[i % n_shards_this_class].append(sample)

            # fewer samples for class than samples_per_shard_per_class
            if n_shards_this_class == 1:
                class_paths = [str(save_path / f"{class_name}.tfrecord")]
            else:
                # Starts with 1, better human readable
                class_paths = [str(save_path / f"{class_name}-{s + 1:05d}-of-{n_shards_this_class:05d}.tfrecord") for s in range(n_shards_this_class)]

            shards.extend(class_shards)
            shard_paths.extend(class_paths)

        total_examples = len(indexed_dataset)
        bytes_written = 0

        pbar = tqdm(
            total=total_examples,
            desc="Writing TFRecord",
            unit="ex",
            dynamic_ncols=True,
        )

        for shard, path in zip(shards, shard_paths):
            with tf.io.TFRecordWriter(path) as writer:
                for img_data, label_str, label_idx in shard:
                    example = self._make_tf_example(
                        img_data, label_str, label_idx
                    )
                    serialised = example.SerializeToString()
                    writer.write(serialised)
                    bytes_written += len(serialised)

                    # Estimate total size from average bytes per example so far
                    done = pbar.n + 1
                    avg_bytes = bytes_written / done
                    est_total = avg_bytes * total_examples
                    pbar.set_postfix_str(
                        f"{_fmt_size(bytes_written)} / ~{_fmt_size(int(est_total))}"
                    )
                    pbar.update(1)

        pbar.close()
        logger.info("Wrote %s examples (%s) across %s shard(s) [layout=%s] to %s", total_examples, _fmt_size(bytes_written), len(shard_paths), layout, save_path)

        dataset_properties.save_properties_yaml(
            save_path=save_path,
            total_examples=total_examples,
            samples_per_class=samples_per_class,
            output_name=self.readable_custom_name,
            cfg=self.cfg,
            label_to_index=label_to_index,
            layout=layout,
        )
         
        return save_path