from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

@dataclass
class EvaluationResult:
    f1_scores: dict[str, float]
    per_class_f1: np.ndarray
    true_labels: np.ndarray
    pred_labels: np.ndarray
    class_names: list[str]

def build_evaluation_result(true_labels: np.ndarray, pred_labels: np.ndarray, class_names: Optional[list[str]] = None) -> EvaluationResult:
    """Build an :class:`EvaluationResult` from index-coded true/predicted label arrays (position ``i`` in *class_names*)."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    num_classes = len(class_names) if class_names is not None else len(np.unique(true_labels))
    class_indices = list(range(num_classes))

    if class_names is None:
        class_names = [f"class_{i}" for i in class_indices]

    # Pass labels to ensure all classes are scored, even if absent in eval_ds, zero_division sets to score 0 instead of warning, when no samples in there
    per_class_f1 = f1_score( true_labels, pred_labels, labels=class_indices, average=None, zero_division=0)
    results: dict[str, float] = {f"f1_{name}": float(score) for name, score in zip(class_names, per_class_f1)}

    # macro = plain mean over classes (rare classes weigh same as common classes)
    # micro = computed over all samples pooled
    # weighted = mean weighted by class size
    results.update({
        "f1_all_macro":    float(f1_score(true_labels, pred_labels, average="macro",    labels=class_indices, zero_division=0)),
        "f1_all_micro":    float(f1_score(true_labels, pred_labels, average="micro",    labels=class_indices, zero_division=0)),
        "f1_all_weighted": float(f1_score(true_labels, pred_labels, average="weighted", labels=class_indices, zero_division=0)),
    })

    return EvaluationResult(
        f1_scores=results,
        per_class_f1=per_class_f1,
        true_labels=true_labels,
        pred_labels=pred_labels,
        class_names=class_names,
    )

def compute_f1_score(model: tf.keras.Model, ds: tf.data.Dataset, class_names: Optional[list[str]] = None) -> EvaluationResult:
    """Iterate through dataset once, gathering labels and predictions in the same pass.
    Then inference on the batches, to get the predicted index for via argmax."""
    true_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    # need to be splitted 
    for x, y in ds:
        preds = model(x, training=False)
        pred_batches.append(np.argmax(np.asarray(preds), axis=1))
        y_numpy = y.numpy()
        true_batches.append(np.argmax(y_numpy, axis=1) if y_numpy.ndim > 1 else y_numpy)

    all_true_labels = np.concatenate(true_batches) # concatenated to 1d array
    all_predicted_labels = np.concatenate(pred_batches)

    return build_evaluation_result(all_true_labels, all_predicted_labels, class_names)
