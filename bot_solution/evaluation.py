"""Считаем метрики по кукам, используя metric.py из условия задания."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from metric import precision_at_recall, recall_at_fpr


def operating_point(y: np.ndarray, score: np.ndarray) -> dict:
    """Находим достижимый порог для описания результата, но не для submission.

    Куки с одинаковым скором идут одной группой. Среди точек с Recall >= 0.7
    выбираем максимальный Precision, а при равенстве — большую полноту.
    """
    order = np.argsort(-score, kind="mergesort")
    sorted_score, sorted_y = score[order], y[order]
    ends = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
    tp = np.cumsum(sorted_y)[ends]
    precision = tp / (ends + 1)
    recall = tp / y.sum()
    eligible = np.flatnonzero(recall >= 0.7)
    best_precision = precision[eligible].max()
    best = eligible[precision[eligible] == best_precision][-1]
    threshold = float(sorted_score[ends[best]])
    predicted = score >= threshold
    tp_n = int((predicted & (y == 1)).sum())
    fp_n = int((predicted & (y == 0)).sum())
    if not np.isclose(tp_n / predicted.sum(), precision_at_recall(y, score), atol=1e-12):
        raise AssertionError("Operating point disagrees with the official metric")
    return {
        "threshold": threshold, "precision": float(best_precision),
        "recall": float(recall[best]), "true_positives": tp_n,
        "false_positives": fp_n, "false_negatives": int(y.sum() - tp_n),
        "true_negatives": int((y == 0).sum() - fp_n),
    }


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """Основная метрика и диагностика. Здесь нужны оба класса."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if y.ndim != 1 or y.shape != score.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation requires equally sized 1D arrays and both binary classes")
    if not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all():
        raise ValueError("Model returned invalid probabilities")
    return {
        "precision_at_recall_0.7": precision_at_recall(y, score),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_fpr_0.01": recall_at_fpr(y, score),
        "n_cookies": int(len(y)), "n_positive": int(y.sum()),
        "positive_fraction": float(y.mean()), "operating_point": operating_point(y, score),
    }
