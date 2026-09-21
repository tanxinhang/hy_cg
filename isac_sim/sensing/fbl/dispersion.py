"""dispersion（自 ``isac_sim/sensing/fbl.py`` 拆出）。"""

from __future__ import annotations

import math
import numpy as np
from isac_sim.core.config import Config


LN2 = math.log(2.0)


LOG2E = 1.0 / LN2


def channel_dispersion(gamma: np.ndarray | float) -> np.ndarray | float:
    """Channel dispersion ``V(gamma) = (1 - (1+gamma)^-2) (log2 e)^2``."""
    g = np.asarray(gamma, dtype=float)
    v = (1.0 - 1.0 / (1.0 + np.maximum(g, 0.0)) ** 2) * LOG2E ** 2
    return float(v) if np.ndim(gamma) == 0 else v


def blocklength_latency_s(cfg: Config) -> float:
    """Latency of one FBL report block: ``n`` channel uses at bandwidth ``B``."""
    B = cfg.waveform.N * cfg.waveform.delta_f
    return float(cfg.comm.n_block) / B


def packet_bits(cfg: Config) -> float:
    """Payload of one soft-information report, in bits."""
    return float(max(cfg.comm.K_candidates, 0) * cfg.comm.b_d)


def report_latency_s(cfg: Config, rate_bps: float) -> float:
    """Latency of one report under the configured latency bookkeeping."""
    if cfg.comm.latency_model.lower() == "blocklength":
        return blocklength_latency_s(cfg)
    return packet_bits(cfg) / max(float(rate_bps), 1e-12)
