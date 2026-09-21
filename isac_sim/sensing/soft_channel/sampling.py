"""sampling（自 ``isac_sim/sensing/soft_channel/sampling.py`` 拆出）。"""

from __future__ import annotations

import math
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.detection.llr import draw_llr, llr_h0_offset, llr_var1
from isac_sim.cooperation.reporting import report_chi
from isac_sim.sensing.soft_channel.moments import local_moments, received_moments


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

    chi = float(np.clip(report_chi(tables, plan, link, q), 0.0, 1.0))
    if rng.random() < chi:
        return _draw_local(cfg, tables, link, q, rng, h1)

    d = cfg.detect
    if d.comm_error_model == "erasure":
        return 0.0
    if d.comm_error_model == "gaussian_replacement":
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


def draw_received_full_llr(
    cfg: Config,
    tables,
    link: Link,
    q: int,
    rng: np.random.Generator,
    h1: bool,
    plan: "object | None" = None,
) -> float:
    """Draw one exact local LLR, or zero when its packet is erased."""
    if cfg.detect.soft_stat_model.lower() != "llr":
        raise ValueError("exact LLR fusion requires detect.soft_stat_model='llr'")
    if cfg.detect.comm_error_model != "erasure":
        raise ValueError("exact LLR fusion requires a true-erasure report channel")
    chi = (
        float(np.clip(report_chi(tables, plan, link, q), 0.0, 1.0))
        if cfg.detect.enable_comm_error_pollution else 1.0
    )
    if cfg.detect.enable_comm_error_pollution and rng.random() >= chi:
        return 0.0
    i, j = link
    gamma = max(float(tables.gamma_sense[i, j, q]), 0.0)
    return float(
        _draw_local(cfg, tables, link, q, rng, h1)
        + llr_h0_offset(gamma, cfg.detect.n_looks)
    )
