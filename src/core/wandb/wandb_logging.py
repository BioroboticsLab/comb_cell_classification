"""Builds evaluation plots (F1 bars, confusion matrices) and logs them to the active W&B run."""
import os

import matplotlib
matplotlib.use("Agg")  # headless: with DISPLAY set, PyQt5 (napari) + cv2's hijacked Qt plugin path make QtAgg abort the process
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import wandb
from wandb.sdk.data_types._private import MEDIA_TMP
from src.core.evaluation import EvaluationResult

def _ensure_media_tmp() -> None:
    """Re-create wandb's import-time media temp dir if /tmp cleanup deleted it during a long-running job (FileNotFoundError on wandb.Image save)."""
    os.makedirs(MEDIA_TMP.name, exist_ok=True)

def log_evaluation_result_to_wandb(run, eval_result: EvaluationResult) -> None:
    """Log an already-computed :class:`EvaluationResult` (aggregate F1 scalars, per-class F1 plot, both confusion matrices) without re-running the model."""
    _ensure_media_tmp()
    run.log({
        "eval/f1_macro": eval_result.f1_scores.get("f1_all_macro"),
        "eval/f1_micro": eval_result.f1_scores.get("f1_all_micro"),
        "eval/f1_weighted": eval_result.f1_scores.get("f1_all_weighted"),
        "eval/f1_per_class": _build_f1_bar_plot(eval_result),
        "eval/confusion_matrix": _build_confusion_matrix(eval_result),
        "eval/confusion_matrix_col_normalized": _build_col_normalized_confusion_matrix(eval_result),
    })

def _build_f1_bar_plot(eval_result: EvaluationResult):
    f1_table = wandb.Table(
        data=[[name, score] for name, score in eval_result.f1_scores.items()],
        columns=["class", "f1"],
    )
    return wandb.plot.bar(f1_table, "class", "f1", title="Evaluation F1 Scores")

def _build_confusion_matrix(eval_result: EvaluationResult):
    return wandb.plot.confusion_matrix(
        y_true=eval_result.true_labels.tolist(),
        preds=eval_result.pred_labels.tolist(),
        class_names=eval_result.class_names,
        title="Evaluation Confusion Matrix",
    )

def _build_col_normalized_confusion_matrix(eval_result: EvaluationResult):
    """Confusion matrix with predicted classes as rows and true classes as columns, normalized so that the largest count in each column is 1.0."""
    class_names = eval_result.class_names
    cm = confusion_matrix(
        eval_result.true_labels,
        eval_result.pred_labels,
        labels=list(range(len(class_names))),
    ).T  # sklearn returns rows=true, cols=predicted; transpose to swap

    col_max = cm.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1  # avoid divide-by-zero for empty true classes
    cm_normalized = cm / col_max

    n = len(class_names)
    size = max(8.0, n * 0.5)
    figure, axes = plt.subplots(figsize=(size, size))
    im = axes.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    figure.colorbar(im, ax=axes)

    axes.set_xticks(range(n))
    axes.set_yticks(range(n))
    axes.set_xticklabels(class_names, rotation=45, ha="right")
    axes.set_yticklabels(class_names)
    axes.set_xlabel("True")
    axes.set_ylabel("Predicted")
    axes.set_title("Confusion Matrix (column-max normalized)")

    for i in range(n):
        for j in range(n):
            axes.text(
                j, i, f"{cm_normalized[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > 0.5 else "black",
                fontsize=8,
            )

    figure.tight_layout()
    image = wandb.Image(figure)
    plt.close(figure)
    return image
