"""Residual-based anomaly scoring + lightweight metrics.

Subset of the SKAB sequel's lib_anomaly / lib_metrics, focused on the 4-channel
PLC pipeline. Kept self-contained per the article's independent stance.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

AggregationStrategy = Literal["mean", "max", "pca"]


def sliding_windows(
    arr: np.ndarray, context_len: int, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carve (T, V) into per-window context/forecast slices.

    Returns:
        X:           (N, context_len, V)
        Y:           (N, horizon, V)
        end_indices: (N,) inclusive end-of-horizon row index per window
    """
    T, V = arr.shape
    last_start = T - context_len - horizon
    if last_start < 0:
        raise ValueError(
            f"series too short: T={T} but need context+horizon={context_len + horizon}"
        )
    starts = np.arange(0, last_start + 1, stride, dtype=np.int64)
    n = len(starts)
    X = np.empty((n, context_len, V), dtype=np.float32)
    Y = np.empty((n, horizon, V), dtype=np.float32)
    end_indices = np.empty(n, dtype=np.int64)
    for i, s in enumerate(starts):
        X[i] = arr[s : s + context_len]
        Y[i] = arr[s + context_len : s + context_len + horizon]
        end_indices[i] = s + context_len + horizon
    return X, Y, end_indices


def fit_zscore(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """arr: (T, V). Returns (mean, std) of shape (V,) each."""
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def residual_score(
    pred: np.ndarray,
    truth: np.ndarray,
    std: np.ndarray,
    metric: Literal["mae", "mse"] = "mae",
) -> np.ndarray:
    """Per-sensor residual aggregated over the horizon, in z-score space.

    pred, truth: (N, h, V) in the original sensor space.
    std:         (V,) z-score parameter from fit_zscore on anomaly-free data.
    Returns      (N, V)
    """
    diff = (pred - truth) / std
    if metric == "mse":
        return np.mean(diff * diff, axis=1).astype(np.float32)
    return np.mean(np.abs(diff), axis=1).astype(np.float32)


def aggregate(
    per_sensor: np.ndarray, strategy: AggregationStrategy = "mean"
) -> np.ndarray:
    """Collapse (N, V) per-sensor residual to (N,) single anomaly score."""
    if strategy == "mean":
        return per_sensor.mean(axis=1).astype(np.float32)
    if strategy == "max":
        return per_sensor.max(axis=1).astype(np.float32)
    if strategy == "pca":
        centered = per_sensor - per_sensor.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        except np.linalg.LinAlgError:
            direction = np.ones(per_sensor.shape[1], dtype=np.float32) / np.sqrt(
                per_sensor.shape[1]
            )
        return np.abs(centered @ direction).astype(np.float32)
    raise ValueError(f"unknown strategy: {strategy}")


def align_labels_to_windows(
    row_labels: np.ndarray, end_indices: np.ndarray, horizon: int
) -> np.ndarray:
    """A window is labelled 1 if ANY row in its forecast horizon is anomalous."""
    out = np.zeros(len(end_indices), dtype=np.int8)
    for i, e in enumerate(end_indices):
        e = int(e)
        out[i] = int(row_labels[e - horizon : e].sum() > 0)
    return out


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal AUC. No sklearn dependency."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s_lab = labels[order]
    tps = np.cumsum(s_lab)
    fps = np.cumsum(1 - s_lab)
    tpr = np.concatenate([[0.0], tps / pos])
    fpr = np.concatenate([[0.0], fps / neg])
    return float(np.trapezoid(tpr, fpr))


def _f1(pred_mask: np.ndarray, labels: np.ndarray) -> float:
    tp = int(np.sum(pred_mask & (labels == 1)))
    fp = int(np.sum(pred_mask & (labels == 0)))
    fn = int(np.sum(~pred_mask & (labels == 1)))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    n_candidates: int = 50,
) -> dict[str, float]:
    """If threshold is None, search for the F1-maximising decile."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if threshold is None:
        qs = np.linspace(0.50, 0.99, n_candidates)
        candidates = np.quantile(scores, qs)
        best_t, best_f1 = float(candidates[0]), -1.0
        for t in candidates:
            f1 = _f1(scores >= t, labels)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        threshold = best_t
    pred = scores >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    far = fp / (fp + tn) if fp + tn > 0 else 0.0
    mar = fn / (fn + tp) if fn + tp > 0 else 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "far": float(far),
        "mar": float(mar),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
