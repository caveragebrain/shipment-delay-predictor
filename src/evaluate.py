"""Evaluation utilities: metrics, threshold analysis, confusion matrix.

Used by `src/train.py` and importable from notebooks. Pure functions, no I/O
side effects unless explicitly requested (e.g., plot saving).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ThresholdMetrics:
    threshold: float
    f1: float
    precision: float
    recall: float
    tn: int
    fp: int
    fn: int
    tp: int


def metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> ThresholdMetrics:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ThresholdMetrics(
        threshold=float(threshold),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> ThresholdMetrics:
    """Return the threshold with max F1 over a grid (default 0.10..0.90 step 0.02)."""
    if grid is None:
        grid = np.arange(0.10, 0.91, 0.02)
    best: ThresholdMetrics | None = None
    for t in grid:
        m = metrics_at_threshold(y_true, y_proba, float(t))
        if best is None or m.f1 > best.f1:
            best = m
    assert best is not None
    return best


def full_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: List[float] | None = None,
) -> Dict:
    """Comprehensive evaluation: ROC-AUC, PR-AUC, metrics at each requested threshold,
    plus the F1-optimal threshold."""
    if thresholds is None:
        thresholds = [0.4, 0.5]

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "by_threshold": {
            f"{t:.2f}": asdict(metrics_at_threshold(y_true, y_proba, t)) for t in thresholds
        },
        "best_threshold": asdict(find_best_threshold(y_true, y_proba)),
    }
    # Add the textual classification report at threshold 0.5 for human readability
    y_pred_05 = (y_proba >= 0.5).astype(int)
    out["classification_report_0.5"] = classification_report(
        y_true, y_pred_05, target_names=["on_time", "delayed"], zero_division=0
    )
    return out
