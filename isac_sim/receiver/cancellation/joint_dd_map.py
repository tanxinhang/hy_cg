"""Hierarchical MAP for DD state shared by multiple receiver looks."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_sim.receiver.cancellation.dictionaries import direct_dictionary
from isac_sim.receiver.cancellation.nonlinear_refinement import _orth


def refine_direct_dd_joint(
    cfg, observations, *, radius_sigma: float = 2.0, grid_points: int = 5,
    rounds: int = 1, prior_weight: float = 1.0,
):
    """Estimate common DD sources while profiling one complex gain per look."""
    observations = tuple(observations)
    if not observations:
        raise ValueError("at least one reference look is required")
    if grid_points < 3 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 3")
    if radius_sigma < 0.0 or rounds < 1 or prior_weight < 0.0:
        raise ValueError("invalid joint-refinement controls")
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

    sources = list(initial)
    offsets_l = np.linspace(-radius_sigma * sigma_l, radius_sigma * sigma_l, grid_points)
    offsets_k = np.linspace(-radius_sigma * sigma_k, radius_sigma * sigma_k, grid_points)
    for _ in range(rounds):
        for index, source in enumerate(tuple(sources)):
            best = None
            for dk in offsets_k:
                for dl in offsets_l:
                    trial = replace(source,
                                    doppler_bin=float(source.doppler_bin) + float(dk),
                                    delay_bin=float(source.delay_bin) + float(dl))
                    proposed = list(sources)
                    proposed[index] = trial
                    loss = _profiled_loss(cfg, proposed, projected)
                    if sigma_k > 0.0:
                        loss += prior_weight * (float(dk) / sigma_k) ** 2
                    if sigma_l > 0.0:
                        loss += prior_weight * (float(dl) / sigma_l) ** 2
                    key = (loss, abs(float(dk)) + abs(float(dl)))
                    if best is None or key < best[0]:
                        best = (key, trial)
            sources[index] = best[1]
    return sources


def _projected_looks(observations):
    """Build target-protected receiver views without consulting truth fields."""
    projected = []
    for obs in observations:
        protect = _orth(
            np.asarray(obs.basis_belief, dtype=complex)
            if obs.basis_belief is not None
            else np.zeros((obs.y.size, 0), dtype=complex)
        )

        def outside(v, basis=protect):
            return v - basis @ (basis.conj().T @ v) if basis.shape[1] else v

        projected.append((outside, outside(np.asarray(obs.y, complex)), float(obs.sigma2)))
    return projected


def _profiled_loss(cfg, sources, projected) -> float:
    """Sum protected residual likelihoods with look-specific gains profiled out."""
    loss = 0.0
    for outside, y, sigma2 in projected:
        design = outside(direct_dictionary(cfg, sources, tangent_order=0))
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ coef
        loss += float(np.vdot(residual, residual).real) / max(
            sigma2, np.finfo(float).tiny
        )
    return loss
