import json
import logging
import math
import re
from datetime import datetime
from typing import Literal, Optional, Union
from pathlib import Path

import tensorflow as tf
from keras.callbacks import EarlyStopping

import wandb
from wandb.integration.keras import WandbMetricsLogger

from src.core.config.config import BeeCombConfig
from src.training.augmentation import data_augmentation_transform
from src.training.dataset.producer import BeeCombDatasetProducer
from src.training.dataset.dataset import BeeCombDataset
from src.training.models import ModelFactory, sync_batch_size_to_model, sync_roi_to_model
from src.training.training.utils import (
    DiskSpaceGuard,
    GpuError,
    KeepBestCheckpointOnly,
    ProgressLogger,
    find_free_gpu,
    pin_process_to_gpu,
    gpu_device_string,
    assert_input_range,
    assert_label_range,
    _build_wandb_tags,
)
from src.core.evaluation import compute_f1_score
from src.core.wandb.wandb_logging import log_evaluation_result_to_wandb
from src.core.model_info import INFO_FILENAME, write_model_info, build_model_info, dump_model_info, collect_machine_info, build_timing

logger = logging.getLogger(__name__)


def train(dataset: BeeCombDataset, cfg: BeeCombConfig, repeat_train_dataset: bool = True, manual_name: Optional[str] = None, gpu_id: Optional[Union[int, GpuError]] = None) -> None:
    """Train one model on a ``BeeCombDataset`` on a single GPU (auto-detected when ``gpu_id`` is None), checkpointing to disk and logging metrics and evaluation F1 to Weights & Biases."""

    if cfg.training.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if cfg.training.use_class_weight and cfg.training.oversample_minority_classes:
        if cfg.training.skip_combined_cw_os:
            logger.warning("Skipping training: use_class_weight and oversample_minority_classes are both enabled and skip_combined_cw_os=True.")
            return
        logger.warning("use_class_weight and oversample_minority_classes are both enabled. This would double correct the dataset!")
    if not cfg.training.use_class_weight and not cfg.training.oversample_minority_classes and cfg.training.skip_imbalanced:
        logger.warning( "Skipping training for imbalanced dataset.")
        return

    # Single-phase sentinels (config/CLI): 0 unfreezes after zero warm-up epochs
    # (backbone trainable from epoch 1), -1 never unfreezes (backbone frozen for
    # the whole run). Normalized here so run naming and phase logic below only
    # ever see the canonical freeze_backbone/unfreeze_after_epochs pair.
    if cfg.training.unfreeze_after_epochs == 0:
        cfg.training.unfreeze_after_epochs = None
        cfg.training.freeze_backbone = False
    elif cfg.training.unfreeze_after_epochs == -1:
        cfg.training.unfreeze_after_epochs = None
        cfg.training.freeze_backbone = True

    filepath, run_name = ModelFactory.generate_wandb_checkpoint_path(cfg=cfg, mk_dir=True, include_date=True, added_manual_name=manual_name)

    checkpoint_dir = Path(filepath).parent

    folder_name = checkpoint_dir.name
    machine = collect_machine_info()
    start_time = datetime.now()
    write_model_info(checkpoint_dir, cfg, dataset.class_names, trained=build_timing(start_time), machine=machine, folder_name=folder_name)

    # gpu_id is None when train() is called standalone. batch_train passes respective GPU ID for use.
    if gpu_id is None:
        gpu_id = pin_process_to_gpu(find_free_gpu(max_used_mib=1000))
    if isinstance(gpu_id, GpuError):
        logger.info("No usable GPU (%s); training on CPU.", gpu_id.name)
    else:
        logger.info("Training on GPU %d.", gpu_id)
    device = gpu_device_string(gpu_id)
    logger.info("Training on single device %s (OneDeviceStrategy).", device)
    strategy = tf.distribute.OneDeviceStrategy(device)
    global_batch_size = cfg.dataset.batch_size * strategy.num_replicas_in_sync

    train_ds, val_ds, eval_ds = dataset.split(
        validation_split=cfg.dataset.validation_amount,
        eval_split=cfg.dataset.evaluation_amount,
        seed=cfg.dataset.seed,
        shuffle_training=True,
        batch_size=global_batch_size,
        oversample=cfg.training.oversample_minority_classes,
    )

    steps_per_epoch = None
    if repeat_train_dataset:
        # calculate epoch length by calculating number of training samples and dividing by global batch size
        train_n = (
            dataset.num_total_size
            - int(round(dataset.num_total_size * cfg.dataset.validation_amount))
            - int(round(dataset.num_total_size * cfg.dataset.evaluation_amount))
        )
        steps_per_epoch = math.ceil(train_n / global_batch_size)
        train_ds = train_ds.repeat()
    elif cfg.training.oversample_minority_classes:
        raise ValueError("oversample_minority_classes=True requires repeat_train_dataset=True so steps_per_epoch is defined for the infinite resample stream.")


    assert_input_range(train_ds)
    assert_label_range(train_ds, dataset.num_classes)

    # Two-phase fine-tune (see TrainingConfig.unfreeze_after_epochs)
    warmup_epochs = cfg.training.unfreeze_after_epochs or 0
    two_phase = warmup_epochs > 0
    if two_phase and not cfg.training.freeze_backbone:
        raise ValueError("unfreeze_after_epochs requires freeze_backbone=True — the warm-up phase is the frozen phase.")
    if two_phase and warmup_epochs >= cfg.training.epochs:
        raise ValueError(f"unfreeze_after_epochs={warmup_epochs} leaves no epochs for the fine-tune phase (training.epochs={cfg.training.epochs}).")

    def _make_optimizer(learning_rate: float) -> tf.keras.optimizers.Optimizer:
        optimizer_config = {
            "learning_rate": learning_rate,
            "epsilon": cfg.training.optimizer_epsilon,
        }
        if cfg.training.clipnorm is not None:
            optimizer_config["clipnorm"] = cfg.training.clipnorm
        if cfg.training.gradient_accumulation_steps > 1:
            optimizer_config["gradient_accumulation_steps"] = cfg.training.gradient_accumulation_steps
        return tf.keras.optimizers.get({
            "class_name": cfg.model_compile.optimizer,
            "config": optimizer_config,
        })

    # The model emits logits (see models.py head), hence from_logits=True.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Build the model inside the strategy scope so its variables live on the device.
    with strategy.scope():
        model = ModelFactory.create(name=cfg.training.model,
                          input_shape=(cfg.dataset.roi_size_tuple[0], cfg.dataset.roi_size_tuple[1], cfg.dataset.num_channels),
                          num_of_classes=dataset.num_classes,
                          batchnorm_momentum=cfg.training.batchnorm_momentum,
                          freeze_backbone=cfg.training.freeze_backbone,
                          gray_to_rgb=cfg.training.gray_to_rgb)
        model.compile(optimizer=_make_optimizer(cfg.training.learning_rate), loss=loss_fn, metrics=cfg.model_compile.metrics)

    logger.info("class weights: %s", dataset.class_weights)

    with wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=run_name, tags=_build_wandb_tags(cfg), config=cfg.__dict__) as run:
        run.summary["class_names"] = list(dataset.class_names)
        keep_best = KeepBestCheckpointOnly(filepath=filepath, monitor="val_loss", mode="min")

        def _make_callbacks(final_phase: bool) -> list[tf.keras.callbacks.Callback]:
            callbacks = [
                         tf.keras.callbacks.TerminateOnNaN(),
                         WandbMetricsLogger(),
                         DiskSpaceGuard(path=filepath),
                         tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor="val_loss", save_best_only=False),
                         keep_best,
                        ]
            if final_phase and cfg.training.epoch_early_stop:
                patience_value = cfg.training.epoch_early_stop_patience + (cfg.training.adaptive_lr_patience if cfg.training.use_adaptive_lr else 0)
                callbacks.append(EarlyStopping(patience=patience_value))
            if final_phase and cfg.training.use_adaptive_lr:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=cfg.training.adaptive_lr_factor,
                    patience=cfg.training.adaptive_lr_patience,
                    min_lr=cfg.training.adaptive_min_lr_rate,
                ))

            callbacks.append(ProgressLogger(min_interval_s=10.0))
            return callbacks

        initial_epoch = 0
        if two_phase:
            logger.info("Phase 1/2: warming up head on frozen backbone for %d epoch(s) at lr=%g.", warmup_epochs, cfg.training.learning_rate)
            warmup_history = model.fit(
                x=train_ds,
                validation_data=val_ds,
                initial_epoch=initial_epoch,
                epochs=warmup_epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=0,
                class_weight=dataset.class_weights if cfg.training.use_class_weight else None,
                callbacks=_make_callbacks(final_phase=False),
            )
            warmup_losses = warmup_history.history.get("loss", [])
            if warmup_losses and math.isnan(warmup_losses[-1]):
                raise RuntimeError("Warm-up phase ended with NaN loss; not fine-tuning on poisoned weights.")

            # uses the latest epoch which is trained on warm up or 0 if no epoch fully trained
            initial_epoch = warmup_history.epoch[-1] + 1 if warmup_history.epoch else 0

            with strategy.scope():
                ModelFactory.set_backbone_trainable(model, True)
                # Because compile caches the trainable values
                model.compile(optimizer=_make_optimizer(cfg.training.finetune_learning_rate), loss=loss_fn, metrics=cfg.model_compile.metrics)
            logger.info("Phase 2/2: unfroze backbone; fine-tuning at lr=%g (epoch %d..%d).", cfg.training.finetune_learning_rate, initial_epoch + 1, cfg.training.epochs)

        model.fit(
            x=train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=cfg.training.epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=0,
            class_weight=dataset.class_weights if cfg.training.use_class_weight else None,
            callbacks=_make_callbacks(final_phase=True),
        )

        if keep_best.best_path and Path(keep_best.best_path).is_file():
            model.load_weights(keep_best.best_path)
            logger.info("Reloaded best checkpoint %s (%s=%.6f) for evaluation.",
                        Path(keep_best.best_path).name, keep_best.monitor, keep_best.best_value)
        else:
            logger.warning("No best checkpoint recorded; evaluating the final in-memory weights.")
        eval_result = compute_f1_score(model, eval_ds, class_names=dataset.class_names)
        
        logger.info("Evaluation F1 Scores: %s", eval_result.f1_scores)
        log_evaluation_result_to_wandb(run, eval_result)

        #write model_info.json
        info = build_model_info(cfg, dataset.class_names, eval_result=eval_result,
                                trained=build_timing(start_time, datetime.now()),
                                machine=machine, folder_name=folder_name)
        dump_model_info(checkpoint_dir, info)
        run.summary["model_info_json"] = json.dumps(info)

        _log_best_model_artifact(run, checkpoint_dir, best_path=keep_best.best_path,
                                 monitor=keep_best.monitor, best_value=keep_best.best_value)


def _checkpoint_epoch(path: Path) -> Optional[int]:
    """1-indexed epoch encoded in a checkpoint filename by the filepath template, or None."""
    m = re.search(r"_epoch-(\d+)\.keras$", path.name)
    return int(m.group(1)) if m else None


def _log_best_model_artifact(run, checkpoint_dir: Path, best_path: Optional[str] = None,
                             monitor: Optional[str] = None, best_value: Optional[float] = None) -> None:
    """Upload the run's best checkpoint plus its model_info.json as one W&B model artifact."""
    best = Path(best_path) if best_path else None
    if best is None or not best.is_file():
        candidates = sorted(checkpoint_dir.glob("*.keras"), key=lambda p: (_checkpoint_epoch(p) or 0, p.stat().st_mtime))
        best = candidates[-1] if candidates else None
    if best is None:
        logger.warning("No checkpoint found in %s; skipping W&B model artifact upload.", checkpoint_dir)
        return

    metadata = {}
    if monitor is not None:
        metadata["monitor"] = monitor
    # Guard the ±inf sentinel (no monitor value ever seen), which wandb's metadata validation rejects (non-JSON float).
    if best_value is not None and math.isfinite(best_value):
        metadata["best_value"] = best_value
    # The download picker shows the checkpoint's epoch from artifact metadata.
    epoch = _checkpoint_epoch(best)
    if epoch is not None:
        metadata["epoch"] = epoch

    # Same collection name WandbModelCheckpoint used, so new runs list uniformly next to old ones in W&B.
    artifact = wandb.Artifact(f"run_{run.id}_model", type="model", metadata=metadata)
    artifact.add_file(str(best))
    info_path = checkpoint_dir / INFO_FILENAME
    if info_path.is_file():
        artifact.add_file(str(info_path))
    run.log_artifact(artifact, aliases=["best"])
    logger.info("Uploaded best checkpoint %s (+ %s) to W&B as %s:best.", best.name, INFO_FILENAME, artifact.name)


def batch_train(dataset_root: Path, cfgs: list[BeeCombConfig], gpu_id: Union[int, GpuError]) -> None:
    """Train multiple config variations serially on one shared GPU, collecting per-config failures and raising a summary ``RuntimeError`` at the end instead of aborting the sweep."""
    for cfg in cfgs:
        ModelFactory.ensure_model_dependencies(cfg.training.model)
        sync_roi_to_model(cfg)
        sync_batch_size_to_model(cfg)

    if isinstance(gpu_id, GpuError):
        logger.info("No usable GPU (%s); training %d config(s) on CPU.", gpu_id.name, len(cfgs))
    else:
        logger.info("Training on pre-claimed GPU %d over %d config(s).", gpu_id, len(cfgs))
    gpu_id = pin_process_to_gpu(gpu_id)

    failures: list[tuple] = []  # (idx, cfg, exception)

    for idx, cfg in enumerate(cfgs):
        logger.info("Starting config %d (%s).", idx, cfg.training.model)
        ModelFactory.load_registry(cfg.paths.models_yaml_path)
        transform = data_augmentation_transform(cfg)
        dataset_gen = BeeCombDatasetProducer(dataset_root, cfg)
        try:
            save_path = dataset_gen.toTFRecord(skip_when_exists=True)
            dataset = BeeCombDataset(root_path=save_path, cfg=cfg, transform=transform)

            train(dataset, cfg, gpu_id=gpu_id)
            logger.info("Config %d (%s) finished successfully.", idx, cfg.training.model)
        except Exception as e:  # noqa: BLE001 - one failed config must not abort the sweep
            logger.exception("Config %d (%s) failed: %s", idx, cfg.training.model, e)
            failures.append((idx, cfg, e))
        finally:
            # otherwise previous runs model variable still on GPU
            tf.keras.backend.clear_session()

    if failures:
        summary = ", ".join(f"#{idx} '{cfg.training.model}' ({type(e).__name__}: {e})" for idx, cfg, e in failures)
        raise RuntimeError(f"{len(failures)} of {len(cfgs)} training run(s) failed: {summary}")



