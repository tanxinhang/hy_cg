"""Low-complexity delay/Doppler tangent WLS-MAP target update."""
from __future__ import annotations

import numpy as np

from isac_sim.receiver.cancellation.dictionaries import target_dictionary
from isac_sim.receiver.cancellation_glrt.linalg import _project_out

__all__ = ["fit_jacobian_wls"]


def _dd_innovations(view, ridge: float, max_shift_bins: float):
    """Estimate per-source (dk, dl) from centre/tangent coefficient ratios."""
    target = int(view.scorer.target)
    sources = [s for s in view.sources if int(s.target) == target]
    if not sources:
        return []
    template = target_dictionary(
        view.scorer.cfg, sources, tangent_order=1, covariance_expanded=False)
    model = view.scorer.model
    design = model.cov.whiten_matrix(model.signal_transfer(template))
    design = _project_out(view.scorer.q_neg, design)
    gram = design.conj().T @ design
    scale = max(float(np.trace(gram).real) / max(gram.shape[0], 1), 1e-12)
    beta = np.linalg.solve(
        gram + max(float(ridge), 0.0) * scale * np.eye(gram.shape[0]),
        design.conj().T @ view.scorer.z,
    )
    centre_power = np.abs(beta[0::3]) ** 2
    reference = max(float(np.median(centre_power)), 1e-12)
    out = []
    for index, source in enumerate(sources):
        centre, d_delay, d_doppler = beta[3 * index:3 * index + 3]
        denom = max(float(abs(centre) ** 2), 1e-12)
        dl = float(np.real(d_delay * np.conj(centre)) / denom)
        dk = float(np.real(d_doppler * np.conj(centre)) / denom)
        dl = float(np.clip(dl, -max_shift_bins, max_shift_bins))
        dk = float(np.clip(dk, -max_shift_bins, max_shift_bins))
        confidence = float(np.clip(denom / reference, 0.05, 20.0))
        out.append((source, dk, dl, confidence))
    return out


def fit_jacobian_wls(evaluator, views, sigma: float, options: dict | None = None):
    """Fuse tangent-derived DD innovations with an isotropic position prior."""
    opts = {} if options is None else dict(options)
    ridge = float(opts.get("wls_ridge", 1e-3))
    dd_sigma = max(float(opts.get("wls_dd_sigma_bins", 0.35)), 1e-6)
    max_shift = max(float(opts.get("wls_max_shift_bins", 3.0)), 0.0)
    rows, values, weights = [], [], []
    for view in views:
        for source, dk, dl, confidence in _dd_innovations(view, ridge, max_shift):
            gk = np.asarray(source.jacobian_doppler_state, dtype=float)[:2]
            gl = np.asarray(source.jacobian_delay_state, dtype=float)[:2]
            for gradient, innovation in ((gk, dk), (gl, dl)):
                if np.linalg.norm(gradient) > 0.0:
                    rows.append(gradient)
                    values.append(innovation)
                    weights.append(confidence / dd_sigma ** 2)
    prior_precision = np.eye(2) / max(float(sigma), 1e-12) ** 2
    if rows:
        H = np.asarray(rows, dtype=float)
        b = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        normal = prior_precision + H.T @ (w[:, None] * H)
        rhs = H.T @ (w * b)
        best = np.linalg.solve(normal, rhs)
    else:
        best = np.zeros(2, dtype=float)
    value, gains = evaluator.objective(best, views)
    zero_value, _ = evaluator.objective(np.zeros(2), views)
    return {
        "best": np.asarray(best, dtype=float),
        "value": float(value),
        "gains": np.asarray(gains, dtype=float),
        "converged": True,
        "objective_at_zero": float(zero_value),
        "gain_over_zero": float(value - zero_value),
        "coarse_candidates": 0,
        "full_candidates": 1,
        "screen_receiver_count": len(views),
    }
