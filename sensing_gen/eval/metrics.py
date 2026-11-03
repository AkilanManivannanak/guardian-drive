"""Torch-free metrics."""
from __future__ import annotations

import numpy as np


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
