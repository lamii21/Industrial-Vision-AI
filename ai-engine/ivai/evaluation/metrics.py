"""Classification metrics (computed with scikit-learn from real predictions)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict:
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(f1m),
        "per_class": {
            name: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f1[k]), "support": int(support[k])}
            for k, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
        "n_errors": int(np.sum(np.asarray(y_true) != np.asarray(y_pred))),
    }
