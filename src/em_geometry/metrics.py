"""Legacy numeric operations with explicit sorting and binning protocols.

Do not cast distances to float64 before binning: the saved repeatability
protocol averages float32 distances in float32. These helpers do not assert
that a legacy association is a causal or prospectively validated metric.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Correlations:
    spearman: float
    pearson: float

    @property
    def mean_rp(self):
        return (self.spearman + self.pearson) / 2


def paired_vectors(distances, scores):
    d, y = np.asarray(distances), np.asarray(scores)
    if d.ndim != 1 or y.ndim != 1 or d.shape != y.shape:
        raise ValueError("distances and scores must be matching one-dimensional arrays")
    if not np.isfinite(d).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite distances or scores")
    return d, y


def correlations(distances, scores):
    d, y = paired_vectors(distances, scores)
    if len(d) < 2 or np.ptp(d) == 0 or np.ptp(y) == 0:
        return Correlations(float("nan"), float("nan"))
    return Correlations(float(stats.spearmanr(d, y).statistic),
                        float(stats.pearsonr(d, y).statistic))


def equal_count(distances, scores, size=20, *, sort="default"):
    """Full equal-count bins; drop highest-distance leftovers.

    ``default`` calls np.argsort without a kind, matching the fresh GPT passes;
    ``stable`` matches the separate historical sensitivity scripts.
    """
    d, y = paired_vectors(distances, scores)
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("bin size must be a positive integer")
    if sort not in ("default", "stable"):
        raise ValueError("sort must be default or stable")
    order = np.argsort(d) if sort == "default" else np.argsort(d, kind="stable")
    nb = len(d) // size
    order = order[:nb * size]
    if not nb:
        return np.empty(0, dtype=d.dtype), np.empty(0, dtype=y.dtype)
    return (d[order].reshape(nb, size).mean(axis=1),
            y[order].reshape(nb, size).mean(axis=1))


def equal_width(distances, scores, n_bins=20):
    """Historical equal-width view: nonempty bins and distance midpoints."""
    d, y = paired_vectors(distances, scores)
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if not len(d):
        return np.array([]), np.array([])
    edges = np.linspace(d.min(), d.max(), n_bins + 1)
    assign = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    keep = [b for b in range(n_bins) if (assign == b).any()]
    return (np.array([(edges[b] + edges[b + 1]) / 2 for b in keep]),
            np.array([y[assign == b].mean() for b in keep]))


def residual_layers(distances):
    """HF hidden-state indices: omit embedding 0, final post-norm, constants."""
    d = np.asarray(distances)
    if d.ndim != 2 or d.shape[1] < 3 or not len(d) or not np.isfinite(d).all():
        raise ValueError("expected finite nonempty [prompts, hidden states] distances")
    return [layer for layer in range(1, d.shape[1] - 1)
            if np.ptp(d[:, layer]) != 0]


def equal_count_sweep(distances, scores, size=20, *, sort="default"):
    d = np.asarray(distances)
    layers = residual_layers(d)
    results = []
    for layer in layers:
        c = correlations(*equal_count(d[:, layer], scores, size, sort=sort))
        results.append({"layer": layer, "spearman": c.spearman,
                        "pearson": c.pearson, "mean_rp": c.mean_rp})
    finite = [r for r in results if np.isfinite(r["mean_rp"])]
    if not finite:
        raise ValueError("no eligible layers with finite binned correlations")
    return results, min(finite, key=lambda r: r["mean_rp"])
