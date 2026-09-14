"""One source of truth for local and post-report soft-statistic distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config, Link
from .llr import draw_llr, llr_var1
from .reporting import report_dest


@dataclass(frozen=True)
class BinaryMoments:
    """First two moments under H0 and H1."""

    m0: float
    v0: float
    m1: float
    v1: float

    @property
    def gap(self) -> float:
        return self.m1 - self.m0


def local_moments(cfg: Config, tables, link: Link, q: int) -> BinaryMoments:
    """Moments before the reporting channel."""
    i, j = link
    gamma = max(float(tables.gamma_sense[i, j, q]), 0.0)
    m1 = float(tables.mu_soft[i, j, q])
    v0 = max(float(tables.var0_q[i, j, q]), 0.0)
    if cfg.detect.soft_stat_model.lower() == "llr":
        v1 = float(llr_var1(gamma, cfg.detect.n_looks))
    else:
        sigma0 = float(tables.sigma0[i, j])
        sigma1 = max(cfg.detect.soft_sigma_floor, sigma0 / math.sqrt(1.0 + gamma + 1e-12))
        v1 = sigma1 * sigma1
    return BinaryMoments(0.0, v0, m1, max(v1, 0.0))


def _mix(chi: float, success_m: float, success_v: float,
         failure_m: float, failure_v: float) -> tuple[float, float]:
    mean = chi * success_m + (1.0 - chi) * failure_m
    var = (
        chi * (success_v + (success_m - mean) ** 2)
        + (1.0 - chi) * (failure_v + (failure_m - mean) ** 2)
    )
    return float(mean), float(max(var, 0.0))


def received_moments(
    cfg: Config, tables, link: Link, q: int, plan: "object | None" = None
) -> BinaryMoments:
    """Exact first two moments after packet success/failure mixing."""
    local = local_moments(cfg, tables, link, q)
    if not cfg.detect.enable_comm_error_pollution:
        return local

    _, j = link
    chi = float(np.clip(tables.chi_comm[j, report_dest(plan, link, q)], 0.0, 1.0))
    d = cfg.detect
    model = d.comm_error_model

    if model == "erasure":
        # A lost report is replaced by zero-mean uncertainty carrying no target
        # information under either hypothesis.
        fv = d.soft_error_sigma_scale ** 2 * local.v0
        f0_m, f0_v, f1_m, f1_v = 0.0, fv, 0.0, fv
    elif model == "flip":
        a = float(d.soft_error_flip_scale)
        f0_m, f0_v = -a * local.m0, a * a * local.v0
        f1_m, f1_v = -a * local.m1, a * a * local.v1
    elif model == "biased":
        f0_m = float(d.h0_error_bias_scale) * math.sqrt(local.v0)
        f1_m = float(d.soft_error_bias_scale) * local.m1
        f0_v = f1_v = d.soft_error_sigma_scale ** 2 * local.v0
    else:
        raise ValueError(model)

    m0, v0 = _mix(chi, local.m0, local.v0, f0_m, f0_v)
    m1, v1 = _mix(chi, local.m1, local.v1, f1_m, f1_v)
    return BinaryMoments(m0, v0, m1, v1)


def received_h0_third_central(
    cfg: Config, tables, link: Link, q: int, plan: "object | None" = None
) -> float:
    """Third H0 central moment of the received statistic.

    The canonical erasure channel has zero conditional means, so the mixture
    third moment is simply the packet-success probability times the centred-Gamma
    LLR moment.  Other error models fall back to zero because their third-order
    calibration is not used by the paper release.
    """
    if cfg.detect.soft_stat_model.lower() != "llr":
        return 0.0
    if cfg.detect.comm_error_model != "erasure":
        return 0.0
    i, j = link
    gamma = max(float(tables.gamma_sense[i, j, q]), 0.0)
    a = gamma / (1.0 + gamma)
    local_mu3 = 2.0 * float(max(cfg.detect.n_looks, 1)) * a ** 3
    if not cfg.detect.enable_comm_error_pollution:
        return local_mu3
    chi = float(np.clip(tables.chi_comm[j, report_dest(plan, link, q)], 0.0, 1.0))
    return chi * local_mu3


def _draw_local(
    cfg: Config, tables, link: Link, q: int, rng: np.random.Generator, h1: bool
) -> float:
    i, j = link
    gamma = max(float(tables.gamma_sense[i, j, q]), 0.0)
    if cfg.detect.soft_stat_model.lower() == "llr":
        return draw_llr(gamma, cfg.detect.n_looks, rng, h1=h1)
    moments = local_moments(cfg, tables, link, q)
    mean, var = (moments.m1, moments.v1) if h1 else (moments.m0, moments.v0)
    return float(rng.normal(mean, math.sqrt(max(var, 0.0))))


def draw_received_soft_stat(
    cfg: Config,
    tables,
    link: Link,
    q: int,
    rng: np.random.Generator,
    h1: bool,
    plan: "object | None" = None,
) -> float:
    """Draw from the same post-report channel used by ``received_moments``."""
    local = local_moments(cfg, tables, link, q)
    if not cfg.detect.enable_comm_error_pollution:
        return _draw_local(cfg, tables, link, q, rng, h1)

    _, j = link
    chi = float(np.clip(tables.chi_comm[j, report_dest(plan, link, q)], 0.0, 1.0))
    if rng.random() < chi:
        return _draw_local(cfg, tables, link, q, rng, h1)

    d = cfg.detect
    if d.comm_error_model == "erasure":
        return float(rng.normal(0.0, d.soft_error_sigma_scale * math.sqrt(local.v0)))
    if d.comm_error_model == "flip":
        return -float(d.soft_error_flip_scale) * _draw_local(cfg, tables, link, q, rng, h1)
    if d.comm_error_model == "biased":
        mean = (
            float(d.soft_error_bias_scale) * local.m1
            if h1 else float(d.h0_error_bias_scale) * math.sqrt(local.v0)
        )
        return float(rng.normal(mean, d.soft_error_sigma_scale * math.sqrt(local.v0)))
    raise ValueError(d.comm_error_model)
