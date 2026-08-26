import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Union

import tensorflow as tf

from enum import Enum

from src.core.config.config import BeeCombConfig
from src.core.config.config_utils import augmentation_code

logger = logging.getLogger(__name__)
class ProgressLogger(tf.keras.callbacks.Callback):
    """Emit newline-terminated training progress via ``logging`` instead of the Keras ``\\r`` progress bar, so concurrent runs writing to one log interleave readably instead of clobbering each other's cursor line (pair with ``model.fit(verbose=0)``)."""

    _SHOWN_KEYS = ("loss", "accuracy", "val_loss", "val_accuracy", "learning_rate")

    def __init__(self, min_interval_s: float = 10.0) -> None:
        super().__init__()
        self.min_interval_s = min_interval_s
        self._epoch = 0
        self._last = 0.0

    def _fmt(self, logs: dict) -> str:
        parts = [f"{k}={logs[k]:.4g}" for k in self._SHOWN_KEYS if logs.get(k) is not None]
        return "  ".join(parts)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        self._epoch = epoch
        self._last = 0.0  # force a line early in each epoch

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        now = time.monotonic()
        if now - self._last < self.min_interval_s:
            return
        self._last = now
        epochs = self.params.get("epochs", "?")
        steps = self.params.get("steps")
        position = f"{batch + 1}/{steps}" if steps else f"{batch + 1}"
        logger.info("epoch %s/%s  %s  %s", self._epoch + 1, epochs, position, self._fmt(logs or {}))

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        epochs = self.params.get("epochs", "?")
        logger.info("epoch %s/%s done  %s", epoch + 1, epochs, self._fmt(logs or {}))

def check_for_diskspace(path: str, required_space_gb: float) -> bool:
    """Check if the specified path has at least the required free disk space in GB."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    logger.debug(f"Disk space at {path}: {free_gb:.2f} GB free")
    return free_gb >= required_space_gb


def _resolve_to_existing_parent(path: Union[str, Path]) -> Path:
    """Walk up ``path`` to the first existing parent, since ``shutil.disk_usage`` needs a real path but checkpoint filepaths are templates that don't exist yet."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p

class DiskSpaceGuard(tf.keras.callbacks.Callback):
    """Stop training when free disk space at the checkpoint path drops below a threshold."""
    # Preventing crashing computer when training
    def __init__(self, path: Union[str, Path], min_free_gb: float = 5.0) -> None:
        super().__init__()
        self.path = str(_resolve_to_existing_parent(path))
        self.min_free_gb = min_free_gb

    def _free_gb(self) -> float:
        return shutil.disk_usage(self.path).free / (1024 ** 3)

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        if not check_for_diskspace(self.path, self.min_free_gb):
            raise RuntimeError(f"Refusing to start training: only {self._free_gb():.2f} GB free at {self.path}, below the {self.min_free_gb:.2f} GB threshold.")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if not check_for_diskspace(self.path, self.min_free_gb):
            logger.warning("Stopping training after epoch %d: %.2f GB free at %s, below %.2f GB threshold — further checkpoints would risk filling the disk.", epoch, self._free_gb(), self.path, self.min_free_gb)
            self.model.stop_training = True

class KeepBestCheckpointOnly(tf.keras.callbacks.Callback):
    """Delete every checkpoint in the save directory except the best one when training has finished."""
    # Deletion is deferred to on_train_end (rather than save_best_only=True, which deletes during training) so EarlyStopping.restore_best_weights can still find older artifacts

    def __init__(self, filepath: Union[str, Path], monitor: str = "val_loss", mode: str = "min") -> None:
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.filepath = str(filepath)
        self.monitor = monitor
        self.mode = mode
        self.best_value: float = float("inf") if mode == "min" else float("-inf")
        self.best_path: Optional[str] = None

    def _is_better(self, current: float) -> bool:
        return current < self.best_value if self.mode == "min" else current > self.best_value

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            logger.warning("KeepBestCheckpointOnly: monitor %r not in logs (keys=%s); skipping epoch %d.", self.monitor, list(logs.keys()), epoch)
            return
        if not self._is_better(float(current)):
            return
        self.best_value = float(current)
        # ModelCheckpoint format the filepath with the 1-indexed epoch number, so mirror that here.
        self.best_path = self.filepath.format(epoch=epoch + 1, **logs)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        if self.best_path is None:
            logger.warning("KeepBestCheckpointOnly: no best checkpoint recorded; leaving directory untouched.")
            return

        best = Path(self.best_path)
        save_dir = best.parent
        if not save_dir.is_dir():
            logger.warning("KeepBestCheckpointOnly: directory %s does not exist; nothing to clean.", save_dir)
            return

        for entry in save_dir.glob("*.keras"):
            if entry.resolve() == best.resolve():
                continue
            try:
                entry.unlink()
                logger.info("Removed non-best checkpoint: %s", entry)
            except OSError as e:
                logger.warning("Failed to remove %s: %s", entry, e)
        logger.info("KeepBestCheckpointOnly: kept %s (best %s=%.6f)",best.name, self.monitor, self.best_value)


def _build_wandb_tags(cfg: BeeCombConfig) -> list[str]:
    """Build the list of wandb tags for a training run from the config, centralised so the tag vocabulary stays consistent."""
    imbalance_tag = {
        (True, False): "cw",
        (False, True): "os",
        (False, False): "none",
        (True, True): "both",
    }[(cfg.training.use_class_weight, cfg.training.oversample_minority_classes)]

    lr_tag = "adaptive-lr" if cfg.training.use_adaptive_lr else "static-lr"

    # 4-char augmentation-combination code ("rf00") for wandb
    aug_tag = augmentation_code(cfg.training.augmentation)

    return [*cfg.wandb.tags, cfg.training.model, imbalance_tag, lr_tag, aug_tag]

def assert_input_range(ds: tf.data.Dataset) -> None:
    """Fail fast if the train pipeline produces input the model can't handle."""
    # Check if Data arriving outside [0, 1] or containing NaN/Inf is the, to prevent NaN loss
    for x_batch, _ in ds.take(1):
        x_min = float(tf.reduce_min(x_batch))
        x_max = float(tf.reduce_max(x_batch))
        has_nan = bool(tf.reduce_any(tf.math.is_nan(x_batch)))
        has_inf = bool(tf.reduce_any(tf.math.is_inf(x_batch)))
        logger.info("input range: min=%.3f max=%.3f nan=%s inf=%s", x_min, x_max, has_nan, has_inf)

        if has_nan or has_inf:
            raise ValueError(f"Training data contains NaN/Inf (nan={has_nan}, inf={has_inf}). Fix the data pipeline before training.")
        if x_max > 1.5:
            raise ValueError(f"Training data looks unnormalized (max={x_max:.3f}), but the model's preprocessing layer assumes [0, 1]. Divide by 255 in the pipeline (see ModelFactory._apply_preprocess in models.py for what happens downstream).")
        if x_min < 0.0 - 1e-3 or x_max < 0.2:
            raise ValueError(f"Training data range looks broken: min={x_min:.3f}, max={x_max:.3f}. Expected float pixel values roughly spanning [0, 1].")
        return


def assert_label_range(ds: tf.data.Dataset, num_classes: int, num_batches: int = 8) -> None:
    """Fail fast if labels fall outside ``[0, num_classes)`` before they reach the loss."""
    observed_min: Optional[int] = None
    observed_max: Optional[int] = None
    for _, y_batch in ds.take(num_batches):
        y_int = tf.cast(y_batch, tf.int64)
        batch_min = int(tf.reduce_min(y_int))
        batch_max = int(tf.reduce_max(y_int))
        observed_min = batch_min if observed_min is None else min(observed_min, batch_min)
        observed_max = batch_max if observed_max is None else max(observed_max, batch_max)

    if observed_min is None:
        logger.warning("assert_label_range: dataset yielded no batches; skipping label check.")
        return

    logger.info("label range over %d batch(es): min=%d max=%d (num_classes=%d)", num_batches, observed_min, observed_max, num_classes)

    if observed_min < 0 or observed_max >= num_classes:
        raise ValueError(
            f"Labels out of range for SparseCategoricalCrossentropy: observed "
            f"[{observed_min}, {observed_max}] but num_classes={num_classes} "
            f"(valid labels are 0..{num_classes - 1}). A label >= num_classes is an "
            "out-of-bounds gather in the loss → CUDA_ERROR_MISALIGNED_ADDRESS and a "
            "fatal abort. Check num_classes inference against the remap/oversample "
            "label mapping."
        )
class GpuError(Enum):
    """Results for ``find_free_gpu`` when no GPU can be selected, kept as enum members so ``isinstance`` distinguishes them from real GPU indices."""
    NO_GPU_FOUND = -2
    ALL_GPUS_BUSY = -1


def gpu_device_string(gpu: Union[int, GpuError]) -> str:
    """Map a ``_find_free_gpu`` result to a TF device string, falling back to CPU on a ``GpuError``."""
    if isinstance(gpu, GpuError):
        logger.warning("No usable GPU (%s); falling back to CPU-only training.", gpu.name)
        return "/cpu:0"
    return f"/gpu:{gpu}"


def find_free_gpu(max_used_mib: int = 1000) -> Union[int, GpuError]:
    """Return the index of the least-used GPU per nvidia-smi, or a ``GpuError`` member when no device exists or every GPU exceeds ``max_used_mib`` of used memory."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,    # grabs stdout and stderr
            text=True,              # decode bytes to str
            check=True,             # raise CalledProcessError on non-zero exit
            timeout=5               # kills and raises TimeoutExpired if nvidia-smi hangs
        ).stdout
    except FileNotFoundError:
        logger.warning("nvidia-smi not found; no GPU available, falling back to CPU.")
        return GpuError.NO_GPU_FOUND
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Failed to query GPU memory usage with nvidia-smi: {e}") from e
    
    usage: list[tuple[int, int]] = []
    for line in out.splitlines():
        idx, mem_str = line.strip().split(", ")
        usage.append((int(idx), int(mem_str)))
        
    if not usage:
        logger.warning("No GPUs found by nvidia-smi; defaulting to CPU-only mode.")
        return GpuError.NO_GPU_FOUND
    
    idx, mem = min(usage, key=lambda x: x[1])
    if mem > max_used_mib:
        logger.warning(f"All GPUs are busy: GPU {idx} has {mem} MiB used, which exceeds the threshold of {max_used_mib} MiB.")
        return GpuError.ALL_GPUS_BUSY
    
    logger.info(f"Selected GPU {idx} with {mem} MiB used threshold of {max_used_mib} MiB.")
    return idx


def pin_process_to_gpu(gpu_id: Union[int, GpuError]) -> Union[int, GpuError]:
    """Restrict this process to a single physical GPU (returning its remapped ordinal 0, or the ``GpuError`` unchanged), before any TF op initialises CUDA."""
    if isinstance(gpu_id, GpuError):
        return gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info("Pinned process to physical GPU %d (CUDA_VISIBLE_DEVICES=%d → TF /gpu:0).", gpu_id, gpu_id)
    return 0
