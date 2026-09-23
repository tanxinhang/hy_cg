"""cancellation 积木：直达字典与目标字典。"""

from __future__ import annotations

from isac_sim.core.config import Config
import math
import numpy as np
from typing import List, Sequence

from isac_sim.receiver.cancellation.manifold import kernel_vector, tangent_columns
from isac_sim.receiver.cancellation.sources import DirectSource, TargetSource
from isac_sim.receiver.cancellation.steering import _n_obs, lift_dictionary

def _source_bearings(sources: Sequence, width: int) -> List[float]:
    """One bearing per *column*: each source's block is ``width`` columns wide."""
    out: List[float] = []
    for src in sources:
        out.extend([float(getattr(src, "u", 0.0))] * int(width))
    return out


def direct_dictionary(
    cfg: Config,
    sources: Sequence[DirectSource],
    tangent_order: int | None = None,
    tangent_step: float | None = None,
) -> np.ndarray:
    """``X_j`` of eq. (1b), with an optional fractional-DD tangent extension.

    ``tangent_order = 0`` (the default) gives one column per active illuminator:
    the direct path's delay and Doppler are *computed* from the shared UAV
    positions, so only a complex gain is unknown.  ``tangent_order = n`` adds
    ``2n`` columns per illuminator to absorb synchronisation/oscillator
    mismatch; each added column costs estimation variance (eq. 3) and is
    therefore a robustness axis rather than an improvement.
    """
    c = cfg.cancellation
    order = int(c.interference_tangent_order if tangent_order is None else tangent_order)
    step = float(c.tangent_step_bins if tangent_step is None else tangent_step)
    blocks: List[np.ndarray] = []
    for src in sources:
        amp = math.sqrt(max(src.power_at_receiver, 0.0))
        block = amp * tangent_columns(cfg, src.doppler_bin, src.delay_bin, order, step)
        if order == 1 and bool(c.interference_uncertainty_weighted):
            block = block.copy()
            block[:, 1] *= max(float(c.direct_estimation_sigma_delay_bins), 0.0)
            block[:, 2] *= max(float(c.direct_estimation_sigma_doppler_bins), 0.0)
        blocks.append(block)
    if not blocks:
        return np.zeros((_n_obs(cfg), 0), dtype=complex)
    return lift_dictionary(cfg, np.concatenate(blocks, axis=1),
                           _source_bearings(sources, 1 + 2 * order))


def target_dictionary(
    cfg: Config,
    sources: Sequence[TargetSource],
    tangent_order: int | None = None,
    covariance_expanded: bool | None = None,
) -> np.ndarray:
    """``A_j`` of eq. (1c): one protected tangent set per believed target."""
    c = cfg.cancellation
    order = int(c.tangent_order if tangent_order is None else tangent_order)
    step = float(c.tangent_step_bins)
    expanded = (
        bool(c.covariance_protection)
        if covariance_expanded is None else bool(covariance_expanded)
    )
    blocks: List[np.ndarray] = []
    z95 = 1.6448536269514722
    for src in sources:
        amp = math.sqrt(max(src.power, 0.0))
        block = amp * tangent_columns(
            cfg, src.doppler_bin, src.delay_bin, order, step
        )
        bearings = [float(src.u)] * int(block.shape[1])
        if expanded:
            extra: List[np.ndarray] = []
            extra_bearings: List[float] = []
            sigma_k = max(float(src.sigma_doppler_bin or 0.0), 0.0)
            sigma_l = max(float(src.sigma_delay_bin or 0.0), 0.0)
            sigma_u = max(float(src.sigma_bearing_u or 0.0), 0.0)
            for sign in (1.0, -1.0):
                if sigma_k > 0.0:
                    extra.append(amp * kernel_vector(
                        cfg, float(src.doppler_bin) + sign * z95 * sigma_k,
                        float(src.delay_bin),
                    ))
                    extra_bearings.append(float(src.u))
                if sigma_l > 0.0:
                    extra.append(amp * kernel_vector(
                        cfg, float(src.doppler_bin),
                        float(src.delay_bin) + sign * z95 * sigma_l,
                    ))
                    extra_bearings.append(float(src.u))
                if sigma_u > 0.0:
                    extra.append(amp * kernel_vector(
                        cfg, float(src.doppler_bin), float(src.delay_bin)
                    ))
                    extra_bearings.append(float(np.clip(
                        float(src.u) + sign * z95 * sigma_u, -1.0, 1.0
                    )))
            if extra:
                block = np.column_stack([block, *extra])
                bearings.extend(extra_bearings)
        blocks.append(lift_dictionary(cfg, block, bearings))
    if not blocks:
        return np.zeros((_n_obs(cfg), 0), dtype=complex)
    return np.concatenate(blocks, axis=1)
