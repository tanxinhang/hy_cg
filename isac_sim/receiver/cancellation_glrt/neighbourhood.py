"""Delay--Doppler target-neighbourhood GLRT with family-wise calibration."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_sim.receiver.cancellation import target_dictionary
from isac_sim.receiver.cancellation_glrt.glrt import target_conditioned_glrt


def target_neighbourhood_glrt(
    cfg, obs, result, model, *, target=None, p_fa=0.05,
    radius_bins=None, grid_points=None, dictionary="belief", aggregation="max",
):
    """Aggregate fixed-rank GLRT evidence over a local DD grid.

    Every location uses ``p_fa / L`` (Bonferroni), so the returned analytic
    threshold controls family-wise false alarm without assuming independence.
    Experiments should still calibrate the *maximum statistic itself* on an
    independent H0 split; the receiver benchmark does exactly that.
    """
    tgt = int(obs.weak_index if target is None else target)
    radius = float(cfg.cancellation.target_glrt_radius_bins
                   if radius_bins is None else radius_bins)
    points = int(cfg.cancellation.target_glrt_grid_points
                 if grid_points is None else grid_points)
    offsets = np.linspace(-radius, radius, points) if radius > 0 and points > 1 else np.array([0.0])
    sources = obs.targets_belief if dictionary == "belief" else obs.targets
    target_sources = [src for src in (sources or ()) if int(src.target) == tgt]
    if not target_sources:
        raise ValueError(f"target {tgt} has no source in {dictionary} dictionary")
    count = int(offsets.size ** 2)
    trials = []
    for dl in offsets:
        for dk in offsets:
            shifted = [replace(
                src,
                delay_bin=float(src.delay_bin) + float(dl),
                doppler_bin=float(src.doppler_bin) + float(dk),
            ) for src in target_sources]
            template = target_dictionary(
                cfg, shifted, tangent_order=0, covariance_expanded=False
            )
            out = target_conditioned_glrt(
                cfg, obs, result, model, target=tgt,
                p_fa=float(p_fa) / count, dictionary=dictionary,
                template_override=template, centre_only=True,
            )
            trials.append((float(out.statistic), float(dl), float(dk), out))
    _, dl, dk, best = max(trials, key=lambda item: item[0])
    if aggregation == "max":
        statistic = float(best.statistic)
    elif aggregation == "logmeanexp":
        values = np.asarray([item[0] for item in trials], dtype=float)
        peak = float(np.max(values))
        statistic = peak + float(np.log(np.mean(np.exp(values - peak))))
    else:
        raise ValueError("aggregation must be 'max' or 'logmeanexp'")
    threshold = max(float(item[3].threshold) for item in trials)
    if aggregation != "max":
        threshold = float("inf")
    return replace(
        best,
        statistic=statistic,
        raw_statistic=statistic,
        threshold=threshold,
        p_fa=float(p_fa),
        detected=bool(statistic > threshold),
        selected_offset_delay=dl,
        selected_offset_doppler=dk,
        neighbourhood_size=count,
        statistic_normalization=(
            best.statistic_normalization if aggregation == "max"
            else "neighbourhood_logmeanexp"
        ),
    )
