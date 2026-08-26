import logging
from datetime import datetime
from pathlib import Path
from src.core.config.config import BeeCombConfig
import yaml

logger = logging.getLogger(__name__)


def build_properties_dict(
    total_examples: int,
    samples_per_class: dict[str, int],
    label_to_index: dict[str, int],
    output_name: str,
    cfg: BeeCombConfig,
    layout: str = "mixed",
) -> dict:
    """Build the dataset properties dictionary that ``toTFRecord`` saves as YAML."""
    # layout: "mixed" (legacy) packs all classes into shared shards; "per_class"
    # writes shards per class so the reader can build per-class streams without
    # a filter pass — required for an efficient oversample path.
    return {
        "name": output_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "layout": layout,
        "num_samples": total_examples,
        "num_classes": len(label_to_index),
        "labels": list(label_to_index.keys()),
        "label_to_index": dict(label_to_index),
        "samples_per_class": dict(samples_per_class),
        "image_shape": [
            cfg.dataset.roi_size_tuple[0],
            cfg.dataset.roi_size_tuple[1],
            cfg.dataset.num_channels,
        ],
        "config": {
            "roi_size": list(cfg.dataset.roi_size_tuple),
            "num_channels": cfg.dataset.num_channels,
            "cell_outer_layer": cfg.dataset.cell_outer_layer,
            "apply_clahe": cfg.dataset.apply_clahe,
        },
    }


def save_properties_yaml(
    save_path: Path,
    total_examples: int,
    samples_per_class: dict[str, int],
    label_to_index: dict[str, int],
    output_name: str,
    cfg: BeeCombConfig,
    layout: str = "mixed",
) -> None:
    props = build_properties_dict(
        total_examples, samples_per_class, label_to_index, output_name, cfg,
        layout=layout,
    )
    yaml_path = save_path / f"properties.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(props, f, sort_keys=False)
    logger.info("Properties saved to %s", yaml_path)
