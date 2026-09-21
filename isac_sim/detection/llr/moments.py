"""moments（自 ``isac_sim/detection/llr.py`` 拆出）。"""

from __future__ import annotations

import numpy as np
from isac_sim.core.config import Config

from isac_sim.detection.llr.statistics import llr_delta, llr_var0, llr_var1


def soft_mean(cfg: Config, gamma: np.ndarray | float) -> np.ndarray | float:
    """H1 mean of the soft statistic under the configured model."""
    if cfg.detect.soft_stat_model.lower() == "llr":
        return llr_delta(gamma, cfg.detect.n_looks)
    # Legacy: kappa_mu * log(1 + gamma).  Arithmetic untouched for parity.
    g = np.asarray(gamma, dtype=float)
    out = cfg.detect.soft_mu_scale * np.log1p(np.maximum(g, 0.0))
    return float(out) if np.ndim(gamma) == 0 else out


def soft_var0(cfg: Config, gamma: np.ndarray | float, pair_var0: np.ndarray | float) -> np.ndarray | float:
    """H0 variance of the soft statistic under the configured model.

    In ``"llr"`` mode the variance is *derived* from the sensing SINR, so the
    legacy pair-level ``sigma0^2`` (which only carried the residual-interference
    inflation) is no longer used -- the residual interference already sits in
    the denominator of ``gamma`` itself, and carrying it twice would be double
    counting.
    """
    if cfg.detect.soft_stat_model.lower() == "llr":
        return llr_var0(gamma, cfg.detect.n_looks)
    return pair_var0


def soft_var1(cfg: Config, gamma: np.ndarray | float, pair_var0: np.ndarray | float) -> np.ndarray | float:
    """H1 variance of the soft statistic (used by the Monte-Carlo detector)."""
    if cfg.detect.soft_stat_model.lower() == "llr":
        return llr_var1(gamma, cfg.detect.n_looks)
    # Legacy: sigma0^2 / (1 + gamma).
    g = np.asarray(np.maximum(gamma, 0.0), dtype=float)
    out = np.asarray(pair_var0, dtype=float) / (1.0 + g)
    return float(out) if np.ndim(gamma) == 0 else out
