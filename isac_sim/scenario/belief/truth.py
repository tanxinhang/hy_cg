"""truth（自 ``isac_sim/scenario/belief.py`` 拆出）。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.model import BaseGains, Geometry


def truth_captured_links(
    cfg: Config,
    base_truth: BaseGains,
    base_belief: BaseGains,
    selected: Dict[int, List[Link]],
    dd_std_bins: tuple[np.ndarray, np.ndarray] | None = None,
) -> Dict[int, List[Link]]:
    """Restrict ``selected`` to links whose belief-guided search captures the truth.

    A selected sensing link yields target evidence when the true continuous DD
    coordinate falls inside the covariance-derived search gate around the
    predicted coordinate.  Half a bin accounts for quantisation; the remaining
    width is ``prior.search_gate_sigma`` times the propagated standard deviation.
    """
    out: Dict[int, List[Link]] = {}
    for q, links in selected.items():
        kept: List[Link] = []
        for (i, j) in links:
            if cfg.dd.use_otfs_bin_validity and not base_truth.valid_dd[i, j, q]:
                continue
            if dd_std_bins is None:
                std_l = std_k = 0.0
            else:
                std_l = float(dd_std_bins[0][i, j, q])
                std_k = float(dd_std_bins[1][i, j, q])
            dl = abs(base_truth.tau[i, j, q] - base_belief.tau[i, j, q]) \
                * cfg.waveform.L * cfg.waveform.delta_f
            dk = abs(base_truth.doppler[i, j, q] - base_belief.doppler[i, j, q]) \
                * cfg.waveform.N * cfg.waveform.T
            if dl > 0.5 + cfg.prior.search_gate_sigma * std_l:
                continue
            if dk > 0.5 + cfg.prior.search_gate_sigma * std_k:
                continue
            kept.append((i, j))
        out[q] = kept
    return out


def geometry_robust_base(cfg: Config, base: BaseGains) -> BaseGains:
    r"""Return a conservative path-gain view over a position confidence ball.

    The belief error is horizontal isotropic Gaussian with standard deviation
    ``sigma``.  Its radius has Rayleigh CDF, hence the configured confidence
    mass corresponds to

    ``r = sigma * sqrt(-2 log(1-confidence))``.

    For every target position within that ball, the triangle inequality gives
    ``d_true <= d_hat + r``.  Since the bistatic target gain is proportional to
    ``d_i^-2 d_j^-2``, multiplying the predicted gain by

    ``[d_i/(d_i+r)]^2 [d_j/(d_j+r)]^2``

    is a valid lower bound on the distance-dependent component.  RCS, DD
    leakage, correlation and communication uncertainty are intentionally not
    claimed by this bound.
    """
    sigma = max(float(cfg.prior.belief_sigma_pos_m), 0.0)
    if not cfg.prior.belief_mode or sigma <= 0.0:
        return base
    confidence = float(cfg.prior.robust_position_confidence)
    radius = sigma * np.sqrt(-2.0 * np.log1p(-confidence))
    distances = np.maximum(np.asarray(base.d_uav_tgt, dtype=float), 1.0)
    one_way = (distances / (distances + radius)) ** 2
    bistatic_factor = one_way[:, None, :] * one_way[None, :, :]
    return replace(base, target_gain=base.target_gain * bistatic_factor)
