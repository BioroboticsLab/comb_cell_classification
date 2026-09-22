"""``dataset.label_merge`` renames labels before the remap_to_other collapse, and the default config keeps the thesis mapping unchanged."""

from __future__ import annotations

import json

import cv2
import numpy as np

from src.core.annotations import Annotation, AnnotationDoc
from src.core.config.config import BeeCombConfig
from src.inference.inference import sync_cfg_to_model
from src.training.dataset.producer import BeeCombDatasetProducer
from src.training.dataset.utils import _dataset_generation_worker

RAW_LABELS = ["empty_cell", "open_honey", "capped_honey", "open_brood", "capped_brood", "pollen", "unclear_cell_class", "other_cell"]


def _merged_cfg() -> BeeCombConfig:
    cfg = BeeCombConfig()
    cfg.dataset.label_merge = {"open_honey": "empty_cell"}
    cfg.dataset.remap_classes_set = {"pollen", "unclear_cell_class", "other_cell"}
    return cfg


def test_default_mapping_is_unchanged():
    ds = BeeCombConfig().dataset
    assert ds.label_merge == {}
    assert [ds.target_label(label) for label in RAW_LABELS] == [
        "empty_cell", "other", "capped_honey", "open_brood", "capped_brood", "other", "other", "other",
    ]


def test_open_honey_merges_into_empty_cell():
    ds = _merged_cfg().dataset
    assert [ds.target_label(label) for label in RAW_LABELS] == [
        "empty_cell", "empty_cell", "capped_honey", "open_brood", "capped_brood", "other", "other", "other",
    ]


def test_merge_wins_even_if_label_is_in_remap_set():
    ds = BeeCombConfig().dataset  # open_honey is still in the default remap set
    ds.label_merge = {"open_honey": "empty_cell"}
    assert ds.target_label("open_honey") == "empty_cell"


def test_producer_stores_merged_labels(tmp_path):
    image_path = tmp_path / "comb.png"
    cv2.imwrite(str(image_path), np.full((200, 200), 128, dtype=np.uint8))
    doc = AnnotationDoc(path=tmp_path / "comb.json", annotations=[
        Annotation(id="a", center_x=100, center_y=100, radius=24, label="open_honey"),
        Annotation(id="b", center_x=60, center_y=60, radius=24, label="pollen"),
    ])
    labels = [label for _, label in _dataset_generation_worker((image_path, doc), _merged_cfg())]
    # Only the merge happens at crop time; the "other" collapse stays a load-time step as before.
    assert labels == ["empty_cell", "pollen"]


def test_merge_gets_its_own_dataset_cache_name(tmp_path):
    assert BeeCombDatasetProducer(tmp_path, BeeCombConfig()).readable_custom_name == "outerlayers-0_clahe-True_fixed-rNone"
    assert BeeCombDatasetProducer(tmp_path, _merged_cfg()).readable_custom_name == "outerlayers-0_clahe-True_fixed-rNone_merge-open_honey-to-empty_cell"


def test_merge_round_trips_through_model_info(tmp_path):
    hyperparameters = _merged_cfg().training_cfg_as_dict()
    assert hyperparameters["label_merge"] == {"open_honey": "empty_cell"}
    (tmp_path / "model_info.json").write_text(json.dumps({"hyperparameters": hyperparameters}))

    cfg = BeeCombConfig()
    sync_cfg_to_model(cfg, tmp_path, 224, 224, 1)
    assert cfg.dataset.label_merge == {"open_honey": "empty_cell"}
    assert cfg.dataset.target_label("open_honey") == "empty_cell"
    assert cfg.dataset.target_label("pollen") == "other"
