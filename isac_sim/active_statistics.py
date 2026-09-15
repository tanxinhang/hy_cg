"""Paired, cluster-aware statistics for active-sensing comparisons."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def paired_cluster_summary(
    active: Sequence[float],
    reference: Sequence[float],
    clusters: Sequence[object],
    *,
    epsilon: float = 1e-6,
    weak_threshold: float = 1.0,
    bootstrap_samples: int = 5000,
    seed: int = 151515,
) -> dict[str, float | int]:
    """Summarize paired gains and bootstrap complete geometry clusters."""
    active_array = np.asarray(active, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    cluster_array = np.asarray(clusters, dtype=object)
    if active_array.shape != reference_array.shape or active_array.ndim != 1:
        raise ValueError("active and reference arrays must be paired one-dimensional data")
    if cluster_array.shape != active_array.shape or active_array.size == 0:
        raise ValueError("one non-empty cluster label is required per paired observation")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not np.all(np.isfinite(active_array)) or not np.all(np.isfinite(reference_array)):
        raise ValueError("paired observations must be finite")
    delta = active_array - reference_array
    log_delta = np.log1p(active_array) - np.log1p(reference_array)
    normalized = delta / (1.0 + reference_array)
    weak = reference_array <= float(weak_threshold)

    unique = list(dict.fromkeys(cluster_array.tolist()))
    cluster_delta = np.asarray([
        float(np.mean(delta[cluster_array == label])) for label in unique
    ])
    rng = np.random.default_rng(seed)
    boot_mean = np.empty(bootstrap_samples)
    boot_median = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        sampled = rng.integers(0, len(unique), len(unique))
        expanded = np.concatenate([
            delta[cluster_array == unique[item]] for item in sampled
        ])
        boot_mean[index] = float(np.mean(expanded))
        boot_median[index] = float(np.median(expanded))
    mean_ci = np.percentile(boot_mean, [2.5, 97.5])
    median_ci = np.percentile(boot_median, [2.5, 97.5])
    p10, p50, p90 = np.percentile(delta, [10, 50, 90])
    return {
        "pairs": int(delta.size),
        "clusters": int(len(unique)),
        "delta_mean": float(np.mean(delta)),
        "delta_mean_cluster_ci95_low": float(mean_ci[0]),
        "delta_mean_cluster_ci95_high": float(mean_ci[1]),
        "delta_median": float(np.median(delta)),
        "delta_median_cluster_ci95_low": float(median_ci[0]),
        "delta_median_cluster_ci95_high": float(median_ci[1]),
        "delta_p10": float(p10),
        "delta_p50": float(p50),
        "delta_p90": float(p90),
        "fraction_delta_gt_epsilon": float(np.mean(delta > epsilon)),
        "fraction_non_worse": float(np.mean(delta >= -epsilon)),
        "delta_log1p_mean": float(np.mean(log_delta)),
        "delta_log1p_median": float(np.median(log_delta)),
        "normalized_gain_mean": float(np.mean(normalized)),
        "weak_threshold": float(weak_threshold),
        "weak_pairs": int(np.sum(weak)),
        "weak_delta_mean": float(np.mean(delta[weak])) if np.any(weak) else float("nan"),
        "weak_delta_median": float(np.median(delta[weak])) if np.any(weak) else float("nan"),
        "cluster_delta_std": float(np.std(cluster_delta, ddof=1))
        if cluster_delta.size > 1 else 0.0,
    }
