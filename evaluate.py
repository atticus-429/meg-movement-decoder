"""Evaluation utilities.

The error-rate + 95% bootstrap-CI technique is taken directly from this repo's
SummarizeRNNPerformance.ipynb, which resamples held-out trials with replacement
(10,000 times) and reports the 2.5/97.5 percentiles of the resampled rate.
Here we apply the same idea to classification error (1 - accuracy) instead of
character error rate.
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def error_rate(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def bootstrap_error_ci(y_true, y_pred, n_resamples=10000, seed=0):
    """Return (error_rate, ci_low, ci_high) as fractions, via trial bootstrap."""
    rng = np.random.default_rng(seed)
    err = (np.asarray(y_true) != np.asarray(y_pred)).astype(float)
    n = len(err)
    rates = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        rates[i] = err[idx].mean()
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(err.mean()), float(lo), float(hi)


def format_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    width = max(len(c) for c in class_names) + 2
    header = " " * width + "".join(f"{c:>{width}}" for c in class_names) + "   (pred)"
    lines = [header]
    for i, c in enumerate(class_names):
        row = f"{c:>{width}}" + "".join(f"{cm[i, j]:>{width}d}" for j in range(len(class_names)))
        lines.append(row)
    lines.append("(true) rows / (pred) cols")
    return "\n".join(lines), cm


def per_class_accuracy(cm):
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1)
    return acc
