"""Truth-free nonlinear refinement of the direct-path DD dictionary."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.dictionaries import direct_dictionary


def _orth(a: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    if a.ndim != 2 or a.shape[1] == 0:
        return np.zeros((a.shape[0], 0), dtype=complex)
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    keep = s > rtol * max(float(s[0]), np.finfo(float).tiny)
    return u[:, keep]


def refine_direct_dd(
    cfg,
    obs: Observation,
    *,
    radius_sigma: float = 2.0,
    grid_points: int = 5,
    rounds: int = 1,
) -> Observation:
    """Return an observation with a locally refined direct-path dictionary.

    Candidate DD locations are selected by least-squares fit after projecting
    both the observation and direct dictionary outside the receiver's believed
    target-protection subspace.  The routine never reads ``direct``,
    ``x_direct``, ``h_true`` or any other simulator-truth field.

    This is deliberately a front end to TP-UIC rather than another cancellation
    weight: it attacks nonlinear manifold placement error before the existing
    protected estimator and uncertainty-consistent residual model are applied.
    """
    if grid_points < 3 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 3")
    if radius_sigma < 0.0 or rounds < 1:
        raise ValueError("radius_sigma must be nonnegative and rounds positive")
    sources = list(obs.direct_est or [])
    if not sources:
        return obs
    sigma_l = max(float(cfg.cancellation.direct_estimation_sigma_delay_bins), 0.0)
    sigma_k = max(float(cfg.cancellation.direct_estimation_sigma_doppler_bins), 0.0)
    if sigma_l == 0.0 and sigma_k == 0.0:
        return obs

    protect = _orth(
        np.asarray(obs.basis_belief, dtype=complex)
        if obs.basis_belief is not None
        else np.zeros((obs.y.size, 0), dtype=complex)
    )

    def outside(v):
        return v - protect @ (protect.conj().T @ v) if protect.shape[1] else v

    y = outside(np.asarray(obs.y, dtype=complex))
    offsets_l = np.linspace(-radius_sigma * sigma_l, radius_sigma * sigma_l, grid_points)
    offsets_k = np.linspace(-radius_sigma * sigma_k, radius_sigma * sigma_k, grid_points)
    for _ in range(rounds):
        for index, source in enumerate(tuple(sources)):
            best = None
            for dk in offsets_k:
                for dl in offsets_l:
                    trial = replace(
                        source,
                        doppler_bin=float(source.doppler_bin) + float(dk),
                        delay_bin=float(source.delay_bin) + float(dl),
                    )
                    proposed = list(sources)
                    proposed[index] = trial
                    design = outside(direct_dictionary(cfg, proposed, tangent_order=0))
                    coef = np.linalg.lstsq(design, y, rcond=None)[0]
                    residual = y - design @ coef
                    loss = float(np.vdot(residual, residual).real)
                    key = (loss, abs(float(dk)) + abs(float(dl)))
                    if best is None or key < best[0]:
                        best = (key, trial)
            sources[index] = best[1]
    return replace(obs, direct_est=sources, X=direct_dictionary(cfg, sources))


def apply_direct_dd(cfg, obs: Observation, sources) -> Observation:
    """Apply receiver-estimated direct sources to another compatible look."""
    sources = list(sources)
    if len(sources) != len(obs.direct_est or []):
        raise ValueError("direct-source count differs across receiver looks")
    return replace(obs, direct_est=sources, X=direct_dictionary(cfg, sources))
