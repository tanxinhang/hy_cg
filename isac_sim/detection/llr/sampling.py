"""sampling（自 ``isac_sim/detection/llr.py`` 拆出）。"""

from __future__ import annotations

import numpy as np


def draw_llr(
    gamma: float,
    n_looks: int,
    rng: np.random.Generator,
    h1: bool,
) -> float:
    """Draw one centred local LLR sample.

    Under ``H1`` the normalised bin energy is Gamma(L, scale = 1 + gamma);
    under ``H0`` it is Gamma(L, scale = 1).  ``gamma <= 0`` carries no target
    information, so the two hypotheses coincide and the draw is the H0 one.
    """
    g = max(float(gamma), 0.0)
    scale = (1.0 + g) if h1 else 1.0
    x = float(rng.gamma(shape=float(max(n_looks, 1)), scale=scale))
    return g / (1.0 + g) * (x - float(max(n_looks, 1)))


def draw_llr_erased(
    gamma: float,
    n_looks: int,
    rng: np.random.Generator,
    h1: bool,
) -> float:
    """Draw the statistic the fusion node substitutes after a *failed* packet.

    An erased short packet carries no target information, so under both
    hypotheses the substituted value is an H0-valued LLR, slightly inflated by
    ``soft_error_sigma_scale`` to model the fusion node's uncertainty about
    what it actually received.
    """
    g = max(float(gamma), 0.0)
    x = float(rng.gamma(shape=float(max(n_looks, 1)), scale=1.0))
    base = g / (1.0 + g) * (x - float(max(n_looks, 1)))
    if not h1:
        return base
    # Under H1 a lost report still removes the target evidence; the legacy
    # model expressed this as "mean pulled towards zero", which here is
    # automatic because the H0 draw has zero mean.
    return base
