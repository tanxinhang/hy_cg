"""Detection diagnostics for the disjoint pilot-MAP screen."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.receiver.cancellation import target_dictionary
from isac_sim.receiver.cancellation_glrt.covariance import LowRankCovariance
from isac_sim.receiver.cancellation_glrt.residual_form import ResidualModel


def _covariance(sigma2, factor, size):
    if factor.shape[1]:
        q, r = np.linalg.qr(factor, mode="reduced")
        lam, vectors = np.linalg.eigh(r @ r.conj().T)
        keep = lam > np.finfo(float).eps * max(float(lam.max()), 1.0)
        return LowRankCovariance(sigma2, q @ vectors[:, keep], lam[keep], 1.0)
    return LowRankCovariance(
        sigma2, np.zeros((size, 0), complex), np.zeros(0), 1.0)


def pilot_statistic(cfg, obs, pilot, sensing, target, oracle_direct=False):
    estimate = cx.fit_pilot_map(obs, pilot, cfg.cancellation.prior_variance)
    _residual, factor = cx.apply_pilot_map(obs, estimate, sensing)
    if oracle_direct:
        left = obs.x_direct[sensing] - obs.X[sensing] @ estimate.h_hat
        factor = np.column_stack((factor, left))
    covariance = _covariance(obs.sigma2, factor, sensing.size)
    sliced = replace(
        obs, y=obs.y[sensing], X=obs.X[sensing], A=obs.A[sensing],
        x_direct=obs.x_direct[sensing], s_target=obs.s_target[sensing],
        basis_belief=None, basis_truth=None)
    model = ResidualModel(
        "pilot_map", np.zeros((sensing.size, 0), complex), np.zeros((0, 0)),
        obs.X[sensing] @ estimate.h_hat, 1.0, obs.sigma2, covariance)
    dummy = cx.cancellation_arms(cfg, obs, weak_target=target)["tp_uic_stage1"]
    offsets = np.linspace(
        -float(cfg.cancellation.target_glrt_radius_bins),
        float(cfg.cancellation.target_glrt_radius_bins),
        int(cfg.cancellation.target_glrt_grid_points))
    sources = [src for src in (obs.targets_belief or ())
               if int(src.target) == int(target)]
    statistics = []
    for delay in offsets:
        for doppler in offsets:
            shifted = [replace(
                src, delay_bin=float(src.delay_bin) + float(delay),
                doppler_bin=float(src.doppler_bin) + float(doppler))
                for src in sources]
            template = target_dictionary(
                cfg, shifted, tangent_order=0, covariance_expanded=False)[sensing]
            statistics.append(gl.target_conditioned_glrt(
                cfg, sliced, dummy, model, target=target,
                template_override=template).statistic)
    return float(max(statistics))


__all__ = ["pilot_statistic"]
