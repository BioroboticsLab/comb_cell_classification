"""Run a trained model over images and their annotation JSONs, writing predicted labels back in place."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.core.annotations import Annotation, AnnotationDoc
from src.core.config.config import BeeCombConfig
from src.core.image_processing import preprocess_image
from src.core.annotations.discover import (
    is_fully_labeled,
    pair_images_with_annotations,
    resolve_image_dirs,
)
from src.inference.labels import read_class_names
from src.core.model_info import read_model_info

logger = logging.getLogger(__name__)


def _find_checkpoint(model_dir: Path) -> Path:
    """Return the single ``.keras`` checkpoint inside *model_dir*."""
    checkpoints = sorted(model_dir.glob("*.keras"))
    if not checkpoints:
        raise FileNotFoundError(f"No .keras checkpoint found in {model_dir}.")
    if len(checkpoints) > 1:
        logger.warning(
            "Multiple .keras checkpoints in %s; using %s.", model_dir, checkpoints[0].name
        )
    return checkpoints[0]


def _register_keras_hub_if_needed(checkpoint: Path) -> None:
    """Import keras_hub before deserializing checkpoints that contain its layers."""
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            config = archive.read("config.json").decode("utf-8", errors="replace")
    except (OSError, KeyError, zipfile.BadZipFile):
        return  # unreadable/odd checkpoint: let load_model raise its own error
    if "keras_hub" not in config:
        return
    try:
        import keras_hub
    except ImportError as exc:
        raise RuntimeError(f"{checkpoint.name} contains keras_hub layers (a DinoV3 model) but keras-hub is not installed — run `uv sync` on Linux to install it.") from exc


def _prepare_json_backups(pairs: list[tuple[Path, Path]], images_dir: Path, backup_dir: Path, restore: bool) -> None:
    """Back up the pristine (unlabeled) annotation JSONs before labelling them, restoring from backup first if *restore* is set."""
    backed_up = restored = 0
    for _, annotation_path in pairs:
        rel = annotation_path.relative_to(images_dir)
        backup_path = backup_dir / rel

        if restore and backup_path.exists():
            shutil.copy2(backup_path, annotation_path)
            restored += 1

        # Only back up files seen for the first time — never overwrite an existing backup, so it always holds the original unlabeled JSON even after re-runs.
        if not backup_path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(annotation_path, backup_path)
            backed_up += 1

    if restored:
        logger.info("Restored %d annotation file(s) from %s.", restored, backup_dir)
    if backed_up:
        logger.info("Backed up %d unlabeled annotation file(s) to %s.", backed_up, backup_dir)


def load_model_and_classes(model_dir: Path) -> tuple[tf.keras.Model, list[str], int, int, int]:
    """Load the checkpoint plus its index-ordered class names from *model_dir* and return ``(model, class_names, h, w, c)``."""
    checkpoint = _find_checkpoint(model_dir)
    class_names = read_class_names(model_dir)
    logger.info("Loading model %s (%d classes).", checkpoint, len(class_names))
    _register_keras_hub_if_needed(checkpoint)
    model = tf.keras.models.load_model(checkpoint)
    _, h, w, c = model.input_shape
    return model, class_names, h, w, c


def sync_cfg_to_model(cfg: BeeCombConfig, model_dir: Path, h: int, w: int, c: int) -> None:
    """Override the crop-pipeline config with the loaded model's own settings so inference crops match what it was trained on."""
    cfg.dataset.roi_size_tuple = (h, w)
    cfg.dataset.num_channels = c

    info = read_model_info(model_dir)
    hyperparameters = (info or {}).get("hyperparameters") or {}
    outer_layer = hyperparameters.get("cell_outer_layer")
    if outer_layer is None:
        logger.warning(
            "No cell_outer_layer in %s's model_info.json; falling back to col%d "
            "from config — crops may not match what the model was trained on.",
            model_dir.name, cfg.dataset.cell_outer_layer,
        )
    else:
        cfg.dataset.cell_outer_layer = outer_layer

    # Restore model's own label-remap settings
    remap_to_other = hyperparameters.get("remap_to_other")
    if remap_to_other is None:
        logger.warning(
            "No remap_to_other in %s's model_info.json; keeping the config "
            "default (remap_to_other=%s) — evaluation may remap ground-truth "
            "labels differently than the model was trained.",
            model_dir.name, cfg.dataset.remap_to_other,
        )
    else:
        cfg.dataset.remap_to_other = remap_to_other
        remap_class_name = hyperparameters.get("remap_class_name")
        remap_classes_set = hyperparameters.get("remap_classes_set")
        if remap_class_name is not None:
            cfg.dataset.remap_class_name = remap_class_name
        if remap_classes_set is not None:
            cfg.dataset.remap_classes_set = set(remap_classes_set)
        elif remap_to_other:
            # Older model_info.json: the flag was saved but not the class set.
            logger.warning(
                "%s's model_info.json predates remap_classes_set; using the "
                "config default %s.",
                model_dir.name, sorted(cfg.dataset.remap_classes_set),
            )

    logger.info(
        "Crop geometry from model: roi=%dx%d, channels=%d, col%d. Label remap "
        "from model: remap_to_other=%s (%s -> '%s').",
        h, w, c, cfg.dataset.cell_outer_layer, cfg.dataset.remap_to_other,
        sorted(cfg.dataset.remap_classes_set), cfg.dataset.remap_class_name,
    )


def build_crops(img: np.ndarray, annotations: list[Annotation], cfg: BeeCombConfig, h: int, w: int, c: int) -> np.ndarray:
    """Preprocess every annotation's ROI in *img* into a ``(N, h, w, c)`` uint8 batch."""
    # Crop pipeline (preprocess_image + these args) is intentionally identical to training's, so the model sees the exact same inputs it was trained on.
    crops = np.empty((len(annotations), h, w, c), dtype=np.uint8)
    for i, ann in enumerate(annotations):
        crop = preprocess_image(
            img,
            ann,
            roi_size=cfg.dataset.roi_size_tuple[0],
            outer_layer=cfg.dataset.cell_outer_layer,
            border=12,
            radius_extra=20,
            enable_clahe=cfg.dataset.apply_clahe,
            resize_to_roi=True,
            fixed_radius=cfg.dataset.fixed_radius,
        )
        if crop.ndim == 2:
            crop = crop[..., np.newaxis]
        crops[i] = crop
    return crops

def predict_labels(model: tf.keras.Model, crops: np.ndarray, class_names: list[str], batch_size: int) -> list[str]:
    """Normalise crops to [0, 1], predict logits, and map argmax to class names."""
    x = tf.cast(crops, tf.float32) / 255.0  # the model's own preprocessing layer maps [0,1] onward
    logits = model.predict(x, batch_size=batch_size, verbose=0)
    # argmax picks the highest-scoring class per crop; index i maps to class_names[i].
    indices = np.argmax(logits, axis=1)
    return [class_names[i] for i in indices]

def _classify_dir(
    images_dir: Path,
    model: tf.keras.Model,
    class_names: list[str],
    cfg: BeeCombConfig,
    h: int,
    w: int,
    c: int,
    batch_size: int,
    backup_jsons: bool,
    restore: bool,
    backup_dirname: str,
    labeled_copy_dir: Optional[Path] = None,
    skip_labeled: bool = False,
) -> int:
    """Label every annotation under a single *images_dir* and return the number of files written."""
    backup_dir = images_dir / backup_dirname
    pairs = pair_images_with_annotations(
        images_dir, cfg.import_config.image_suffixes, cfg.import_config.annotation_suffix
    )
    # Never treat the backup copies themselves as labelling targets.
    pairs = [p for p in pairs if backup_dir not in p[1].parents]
    if skip_labeled:
        # Filter before the count below, so the log line and the progress bar both describe the work that is actually going to happen.
        remaining = [p for p in pairs if not is_fully_labeled(p[1])]
        if len(remaining) != len(pairs):
            logger.info(
                "Skipping %d already-labelled annotation file(s).", len(pairs) - len(remaining)
            )
        pairs = remaining
    if not pairs:
        logger.warning("No image/annotation pairs found under %s.", images_dir)
        return 0
    logger.info("Found %d image/annotation pair(s) under %s.", len(pairs), images_dir)

    if backup_jsons or restore:
        _prepare_json_backups(pairs, images_dir, backup_dir, restore=restore)

    read_flag = cv2.IMREAD_GRAYSCALE if c == 1 else cv2.IMREAD_COLOR

    for image_path, annotation_path in tqdm(pairs, desc=f"Labelling {images_dir.name}", unit="img"):
        loader = AnnotationDoc.load(annotation_path)
        if len(loader) == 0:
            continue

        img = cv2.imread(str(image_path), read_flag)
        if img is None:
            logger.warning("Could not read image %s; skipping.", image_path)
            continue

        crops = build_crops(img, loader.annotations, cfg, h, w, c)

        predicted = predict_labels(model, crops, class_names, batch_size)
        for ann, label in zip(loader.annotations, predicted):
            ann.label = label

        loader.save()

        # Per-run record of labelled JSONs, originals are modified in place
        if labeled_copy_dir is not None:
            rel = annotation_path.relative_to(images_dir)
            dest = labeled_copy_dir / images_dir.name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(annotation_path, dest)

    return len(pairs)

def run_inference(
    images_dir: Union[str, Path],
    model_dir: Union[str, Path] = "models",
    batch_size: int = 64,
    cfg: Optional[BeeCombConfig] = None,
    backup_jsons: bool = True,
    restore: bool = False,
    backup_dirname: str = "unlabeled_jsons",
    labeled_copy_dir: Optional[Path] = None,
    skip_labeled: bool = False,
) -> None:
    """Label every annotation under *images_dir* (a directory or glob) with a trained model, in place; ``labeled_copy_dir`` additionally collects a copy of every labelled JSON (grouped by images-dir name)."""
    dirs = resolve_image_dirs(images_dir)
    model_dir = Path(model_dir)
    cfg = cfg or BeeCombConfig()

    model, class_names, h, w, c = load_model_and_classes(model_dir)
    sync_cfg_to_model(cfg, model_dir, h, w, c)

    if len(dirs) > 1:
        logger.info("Matched %d directories: %s", len(dirs), ", ".join(d.name for d in dirs))

    total = 0
    for directory in dirs:
        total += _classify_dir(directory, model, class_names, cfg, h, w, c, batch_size, backup_jsons, restore, backup_dirname, labeled_copy_dir=labeled_copy_dir, skip_labeled=skip_labeled)

    if labeled_copy_dir is not None and labeled_copy_dir.is_dir():
        logger.info("Copied the labelled JSONs to %s.", labeled_copy_dir)
    logger.info("Done. Wrote labels for %d annotation file(s).", total)
