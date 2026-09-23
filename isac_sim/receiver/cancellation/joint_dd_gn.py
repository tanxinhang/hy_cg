"""Bounded Gauss--Newton solver for the shared direct-path DD MAP state."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from isac_sim.receiver.cancellation.dictionaries import direct_dictionary
from isac_sim.receiver.cancellation.joint_dd_map import _projected_looks


def refine_direct_dd_joint_gn(
    cfg, observations, *, radius_sigma: float = 2.0,
    prior_weight: float = 1.0, max_nfev: int = 4,
):
    """Fit common DD offsets with profiled look-specific complex gains.

    A trust-region reflective Gauss--Newton step enforces the same +/-2 sigma
    box used by the 5x5 correctness grid.  Complex gains are profiled out at
    every residual evaluation; Gaussian DD prior residuals complete the MAP
    objective.  Only receiver-visible fields are used.
    """
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one reference look is required")
    if radius_sigma < 0.0 or prior_weight < 0.0 or max_nfev < 1:
        raise ValueError("invalid Gauss--Newton controls")
    initial = list(observations[0].direct_est or [])
    if not initial:
        return initial
    if any(len(obs.direct_est or []) != len(initial) for obs in observations):
        raise ValueError("direct-source count differs across reference looks")
    sigma_l = max(float(cfg.cancellation.direct_estimation_sigma_delay_bins), 0.0)
    sigma_k = max(float(cfg.cancellation.direct_estimation_sigma_doppler_bins), 0.0)
    if sigma_l == 0.0 and sigma_k == 0.0:
        return initial
    projected = _projected_looks(observations)
    scale = np.tile([max(sigma_k, np.finfo(float).tiny),
                     max(sigma_l, np.finfo(float).tiny)], len(initial))
    lower = -radius_sigma * scale
    upper = radius_sigma * scale

    def sources_at(offsets):
        return [replace(src,
                        doppler_bin=float(src.doppler_bin) + float(offsets[2 * i]),
                        delay_bin=float(src.delay_bin) + float(offsets[2 * i + 1]))
                for i, src in enumerate(initial)]

    def residual(offsets):
        sources = sources_at(offsets)
        blocks = []
        for outside, y, sigma2 in projected:
            design = outside(direct_dictionary(cfg, sources, tangent_order=0))
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            error = (y - design @ coef) / np.sqrt(max(sigma2, np.finfo(float).tiny))
            blocks.extend((error.real, error.imag))
        if prior_weight > 0.0:
            blocks.append(np.sqrt(prior_weight) * offsets / scale)
        return np.concatenate(blocks)

    result = least_squares(
        residual, np.zeros(2 * len(initial)), jac="2-point", method="trf",
        bounds=(lower, upper), max_nfev=max_nfev,
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )
    return sources_at(result.x)
