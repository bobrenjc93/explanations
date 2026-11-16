"""Solution-length horizon: how long a worked solution can be before a model's exact-match accuracy falls to q.

Every test problem has a reference worked solution; its length in tokens is the problem's size. Accuracy is binned on a
half-octave grid of that length, and a logistic curve in log2(length) gives the horizon as a single number."""
import json, math
import numpy as np

EDGES = [8] + [round(16 * 2 ** (i / 2)) for i in range(15)] + [2305]  # 8, 16, 23, 32, 45, ... 2048, 2305


def bins(lens, ok):
    """-> [(lo, hi, n, accuracy)] over the half-octave grid (empty bins dropped)."""
    lens, ok = np.asarray(lens), np.asarray(ok, float)
    out = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        m = (lens >= lo) & (lens < hi)
        if m.sum():
            out.append((lo, hi, int(m.sum()), float(ok[m].mean())))
    return out


def fit(lens, ok, iters=60):
    """Logistic regression of correctness on log2(length). -> (a, b) with logit p = a + b * log2(len)."""
    x = np.log2(np.asarray(lens, float))
    y = np.asarray(ok, float)
    X = np.stack([np.ones_like(x), x], 1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p) - 1e-6 * w)
        w = w + step
        if np.abs(step).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def horizon(lens, ok, q=0.5):
    """Length at which the fitted accuracy equals q (None when accuracy does not fall with length)."""
    a, b = fit(lens, ok)
    if b >= -1e-6:
        return None
    return float(2 ** ((math.log(q / (1 - q)) - a) / b))


def curve(lens, ok, xs):
    a, b = fit(lens, ok)
    return [1 / (1 + math.exp(-(a + b * math.log2(x)))) for x in xs]


def subset_index(levels, n=200):
    """Test indices of the first n problems of every level (the subset train3.py scores during training), in test order."""
    levels = np.asarray(levels)
    rank = np.zeros(len(levels), int)
    for l in np.unique(levels):
        idx = np.where(levels == l)[0]
        rank[idx] = np.arange(len(idx))
    return np.where(rank < n)[0]
