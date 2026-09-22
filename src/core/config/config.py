from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class RunFilesConfig:
    """Names of the files/dirs inside a per-run output dir (output/<kind>/<ts>/)."""
    log_dirname: str = "log"          # subdir holding the run's logs
    log_filename: str = "output.log"  # current log file inside log_dirname
    gpu_filename: str = "gpu.claim"   # per-run GPU claim marker, run-dir top level
    args_filename: str = "args.json"  # child argv, re-launched verbatim by --restart
    flags_filename: str = "launch_flags.json"  # training launch flags (debug/offline), reused by the OOM retrain
    config_ref_filename: str = "config_path.txt"  # path of the sweep YAML the run was started with
RUN_FILES = RunFilesConfig()


@dataclass
class WandbConfig:
    project: str = "ccc_models"
    entity: str = "joev98-freie-universit-t-berlin"
    tags: List[str] = field(default_factory=list)

@dataclass
class PathsConfig:
    # Root for all per-run outputs: output/<command>/<timestamp>/ holding the
    # run's log/, wandb/, model/ (training) or evaluation_results/ + jsons/
    # (classify). Relative paths resolve against the project root.
    output_path: Path = Path("output")
    raw_data_path: Path = Path("data/annotated")
    debug_data_path: Path = Path("data/annotated_DEBUG")
    raw_annotations_path: Path = Path("data/annotated")
    produced_dataset_save_path: Path = Path("output/dataset")
    model_save_path: Path = Path("models/")
    models_yaml_path: Path = Path("models.yaml")
    tensorboard_logs: str = "data/logs"
    # Where `ccc classify --evaluate` writes its performance report JSONs
    # (F1 scores + confusion matrix against ground-truth labels). Created on demand.
    evaluation_results_path: Path = Path("evaluation_results")

@dataclass
class ImportConfig:
    image_suffixes: str = ".jpeg, .png"
    annotation_suffix: str = ".json"
    sort_paths: bool = True


@dataclass
class DatasetConfig:
    name: str = "bee_comb_dataset"
    validation_amount: float = 0.1  # 10 % used for validation
    evaluation_amount: float = 0.1  # 10 % used for evaluation
    seed: int = 1                   # shuffling seed for training
    
    apply_clahe: bool = True
    cell_outer_layer: int = 0
    roi_size_tuple: tuple[int, int] = (299, 299)
    num_channels: int = 1
    batch_size: int = 32
    
    # When True, batch_size overwritten by models "recommended_batch_size" in models.yaml
    auto_batch_size: bool = True # Set false to force manual value

    fixed_radius: int = 0 # if > 0, the radius of the crop is fixed to this value instead of using the annotation radius

    remap_to_other: bool = True
    remap_class_name: str = "other"
    remap_classes_set: set = field(
        default_factory=lambda: {"open_honey", "pollen", "unclear_cell_class", "other_cell"}
    )

    # Explicit label renames, e.g. {"open_honey": "empty_cell"}. Applied when the dataset crops are produced, before the remap_to_other collapse, so a renamed label never reaches remap_classes_set.
    label_merge: dict = field(default_factory=dict)

    def target_label(self, label: str) -> str:
        """The class a raw annotation label is trained and evaluated as: label_merge first, then the remap_to_other collapse."""
        label = self.label_merge.get(label, label)
        if self.remap_to_other and label in self.remap_classes_set:
            return self.remap_class_name
        return label


@dataclass
class AugmentationConfig:
    """Parameters for the augmentations listed in ``training.augmentation``."""
    flip_mode: str = "horizontal_and_vertical"
    # 'rotate_60': random rotation by a multiple of step_degrees (n in [0, num_steps)).
    rotate_step_degrees: float = 60
    rotate_num_steps: int = 6
    # (lower, upper) zoom range; negative = zoom in, so (-0.1, 0.0) zooms in up to 10%.
    # Width follows height (aspect ratio preserved).
    zoom_factor: tuple[float, float] = (-0.1, 0.0)
    # Max shift as fraction of image size, per axis.
    wiggle_factor: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 50
    epoch_early_stop: bool = True
    epoch_early_stop_patience: int = 5
    learning_rate: float = 0.001
    optimizer_epsilon: float = 1e-4
    use_adaptive_lr: bool = True
    adaptive_lr_factor: float = 0.5 # LR is multiplied by this factor when a plateau in validation loss is detected
    adaptive_lr_patience: int = 3
    adaptive_min_lr_rate: float = 1e-7

    # Start with the pretrained backbone frozen and train only the fresh Dense
    # head — applies to every model (tf.keras.applications CNNs and KerasHub
    # DinoV3 alike). Head-only training works well at the default learning_rate
    # (~1e-3). Set False to train the whole network from epoch 1: use a much
    # smaller learning_rate (~1e-5..3e-5). Frozen BatchNorm layers run in
    # inference mode (Keras special-cases trainable=False on BN).
    freeze_backbone: bool = True

    # Two-phase fine-tune. When set, epochs 1..N warm up the freshly-initialised
    # head on the frozen backbone at `learning_rate`, then the backbone is
    # unfrozen (fresh optimizer at finetune_learning_rate) for the remaining
    # `epochs`. The warm-up keeps the random head's large early gradients from
    # wrecking the pretrained weights. Requires freeze_backbone=True (the
    # warm-up IS the frozen phase). None disables phase 2: freeze_backbone
    # alone then decides whether the whole run is frozen or trainable from epoch 1.
    # 0 and -1 are sweep sentinels so both single-phase modes fit on the sweep
    # axis: train() rewrites 0 (unfreeze after zero warm-up epochs) to None +
    # freeze_backbone=False and -1 (never unfreeze) to None + freeze_backbone=True.
    unfreeze_after_epochs: Optional[int] = 3

    # Learning rate for the unfrozen fine-tune phase above. Keep it ~1-2 orders
    # of magnitude below the head warm-up `learning_rate`: the pretrained
    # backbone only needs a nudge toward the new domain.
    finetune_learning_rate: float = 5e-5

    # How grayscale (1-channel) input is adapted to the 3-channel pretrained
    # backbone — applies to the CNN and the DinoV3 path alike, ignored for
    # 3-channel input:
    #   "replicate": fixed channel replication (default, 0 params)
    #   "conv1x1":   trainable 1x1 convolution (6 params), initialized to
    #                exact replication so it starts equal to "replicate"
    gray_to_rgb: str = "replicate"

    model: str = "InceptionResNetV2"
    # Default: no augmentation. The factorial ablation sweeps all 2^4 on/off
    # combinations of flip/rotate_60/zoom/wiggle_around_center via config YAMLs.
    augmentation: List[str] = field(default_factory=list)
    gradient_accumulation_steps: int = 1 #turns off gradient accumulation when set to 1; otherwise, gradients are accumulated for this many steps before applying an optimizer step. Useful when VRAM is too small for the desired batch size.
    
    # Global gradient-norm clip. None disables clipping. Absolutely necessary for pretrained backbones of CNNs
    clipnorm: Optional[float] = 5.0

    # Override BatchNorm momentum on the pretrained backbone. tf.keras.applications
    # backbones default to 0.99–0.999; the BN running mean/variance used at
    # inference/validation then adapt over ~1/(1-momentum) steps. Fine-tuning on a
    # domain far from ImageNet (grayscale CLAHE comb cells) leaves those running stats
    # anchored near ImageNet for many epochs → frozen/garbage validation accuracy
    # while training improves. 0.9 lets them track the new domain within ~10 steps.
    # None keeps each backbone's built-in default.
    batchnorm_momentum: Optional[float] = 0.9

    # Enable mixed_float16 global policy. Saves VRAM but interacts badly with large class_weight multipliers (overflow → clipped grads).
    mixed_precision: bool = False
    
    # Apply per-class loss weighting from dataset.class_weights during fit. With imbalanced data the multipliers can reach ~30×, dominating the gradient
    use_class_weight: bool = False
    
    # Resample the training stream so every class is sampled with equal probability per step (rare classes are repeated, common classes under-sampled)
    oversample_minority_classes: bool = True
    skip_imbalanced: bool = True
    skip_combined_cw_os: bool = True

@dataclass
class ModelCompileConfig:
    optimizer: str = "adam"
    loss: str = "sparse_categorical_crossentropy"
    metrics: list = field(default_factory=lambda: ["accuracy"])


@dataclass
class BeeCombConfig:
    """Typed representation of config.yaml, one section dataclass per top-level YAML section (``import_config`` stands in for the reserved keyword ``import``)."""
    wandb: WandbConfig = field(default_factory=WandbConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    import_config: ImportConfig = field(default_factory=ImportConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model_compile: ModelCompileConfig = field(default_factory=ModelCompileConfig)

    def apply_debug_overrides(self) -> None:
        """Switch wandb project and data/annotation paths to the debug variants."""
        self.wandb.project = "ccc_debug"
        self.paths.raw_data_path = self.paths.debug_data_path
        self.paths.raw_annotations_path = self.paths.debug_data_path
        if "debug" not in self.wandb.tags:
            self.wandb.tags.append("debug")


    def training_cfg_as_dict(self) -> dict[str, Any]:
        """Returns a dictionary of the training-related configuration fields."""
        return {
            "model": self.training.model,
            "epochs": self.training.epochs,
            "batch_size": self.dataset.batch_size,
            "auto_batch_size": self.dataset.auto_batch_size,
            "learning_rate": self.training.learning_rate,
            "use_adaptive_lr": self.training.use_adaptive_lr,
            "adaptive_lr_patience": self.training.adaptive_lr_patience,
            "unfreeze_after_epochs": self.training.unfreeze_after_epochs,
            "finetune_learning_rate": self.training.finetune_learning_rate,
            "epoch_early_stop_patience": self.training.epoch_early_stop_patience,
            "oversample_minority_classes": self.training.oversample_minority_classes,
            "use_class_weight": self.training.use_class_weight,
            "mixed_precision": self.training.mixed_precision,
            "wandb_project": self.wandb.project,
            "cell_outer_layer": self.dataset.cell_outer_layer,
            "remap_to_other": self.dataset.remap_to_other,
            # Sorted list (not set) so the dict stays json.dumps-able; inference
            # restores these so evaluation remaps ground truth exactly as trained.
            "remap_class_name": self.dataset.remap_class_name,
            "remap_classes_set": sorted(self.dataset.remap_classes_set),
            "label_merge": dict(self.dataset.label_merge),
            "augmentation": self.training.augmentation,
            "epoch_early_stop": self.training.epoch_early_stop,
            "freeze_backbone": self.training.freeze_backbone,
            "gray_to_rgb": self.training.gray_to_rgb,
        }