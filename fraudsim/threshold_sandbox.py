from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    medium_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional arrays with equal length")
    if not 0.0 <= medium_threshold < high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= medium < high <= 1")

    high_pred = scores >= high_threshold
    review_pred = (scores >= medium_threshold) & ~high_pred
    pass_pred = scores < medium_threshold
    positives = labels == 1
    negatives = labels == 0
    true_positive = int((high_pred & positives).sum())
    false_positive = int((high_pred & negatives).sum())

    return {
        "rows": int(len(labels)),
        "fraud_rows": int(positives.sum()),
        "normal_rows": int(negatives.sum()),
        "thresholds": {
            "medium": float(medium_threshold),
            "high": float(high_threshold),
        },
        "decisions": {
            "pass": int(pass_pred.sum()),
            "review": int(review_pred.sum()),
            "reject": int(high_pred.sum()),
        },
        "high_risk_metrics": {
            "precision": float(precision_score(labels, high_pred, zero_division=0)),
            "recall": float(recall_score(labels, high_pred, zero_division=0)),
            "f1": float(f1_score(labels, high_pred, zero_division=0)),
            "false_positive_rate": _rate(false_positive, int(negatives.sum())),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "missed_fraud": int((~high_pred & positives).sum()),
        },
        "workload": {
            "review_rate": _rate(int(review_pred.sum()), len(labels)),
            "reject_rate": _rate(int(high_pred.sum()), len(labels)),
            "intervention_rate": _rate(int((scores >= medium_threshold).sum()), len(labels)),
        },
    }
