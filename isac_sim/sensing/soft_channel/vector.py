"""vector（自 ``isac_sim/sensing/soft_channel/sampling.py`` 拆出）。"""

from __future__ import annotations

import math
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.sensing.soft_channel.moments import local_moments, received_moments

from isac_sim.sensing.soft_channel.sampling import draw_received_soft_stat


def draw_received_soft_vector(
    cfg: Config,
    tables,
    links: list[Link],
    q: int,
    rng: np.random.Generator,
    h1: bool,
    plan: "object | None" = None,
    base: "object | None" = None,
    rng_by_link: dict[Link, np.random.Generator] | None = None,
) -> np.ndarray:
    """Draw one internally consistent vector of received soft statistics.

    The canonical independence path deliberately delegates to the exact
    per-link LLR/channel sampler, preserving its established distribution and
    random stream.  When observation correlation is enabled, selection and
    thresholding use a covariance model; Monte Carlo must sample that same
    joint model.  We therefore draw the declared moment-matched multivariate
    Gaussian with the exact post-report marginal means/variances and the same
    correlation matrix used by fusion.

    This is a correlation *sensitivity model*, not a claim that correlated
    finite-look LLRs are jointly Gaussian.  Its value is internal calibration:
    under H0 the covariance used to set the threshold is exactly the one being
    sampled, so a configured false-alarm probability is testable.
    """
    if not links:
        return np.zeros(0, dtype=float)
    if (not cfg.corr.enable) or len(links) == 1:
        return np.asarray([
            draw_received_soft_stat(
                cfg, tables, link, q,
                rng_by_link[link] if rng_by_link is not None else rng,
                h1, plan,
            )
            for link in links
        ], dtype=float)

    from isac_sim.detection.corr import covariance_matrix

    moments = [received_moments(cfg, tables, link, q, plan) for link in links]
    mean = np.asarray([
        moment.m1 if h1 else moment.m0 for moment in moments
    ], dtype=float)
    sigma = np.asarray([
        math.sqrt(max(moment.v1 if h1 else moment.v0, 0.0))
        for moment in moments
    ], dtype=float)
    covariance = covariance_matrix(cfg, links, sigma, base=base, q=q)
    if rng_by_link is None:
        return np.asarray(
            rng.multivariate_normal(mean, covariance, check_valid="raise"),
            dtype=float,
        )

    # Use one keyed primitive innovation per physical observation.  The
    # symmetric covariance square root is permutation-equivariant, so merely
    # reordering a selected set cannot change its draw.  Adding/removing a
    # correlated observation may change the covariance transform, as it must,
    # but the underlying per-link innovations remain paired across methods.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_sqrt = (
        eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    ) @ eigenvectors.T
    innovations = np.asarray([
        rng_by_link[link].standard_normal() for link in links
    ], dtype=float)
    return np.asarray(mean + covariance_sqrt @ innovations, dtype=float)
