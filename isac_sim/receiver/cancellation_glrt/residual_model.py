"""装配一条臂的仿射形式与它留下的残余协方差 ``C_res``。"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from isac_sim.receiver.cancellation import EPS, CancellationResult, Observation
from isac_sim.receiver.cancellation_glrt.arm_table import arm_plans
from isac_sim.receiver.cancellation_glrt.belief_error import _belief_error_factor
from isac_sim.receiver.cancellation_glrt.covariance import (
    LowRankCovariance,
    _check_covariance_trace,
)
from isac_sim.receiver.cancellation_glrt.linalg import _psd_sqrt
from isac_sim.receiver.cancellation_glrt.priors import (
    _direct_prior,
    _estimator_covariance,
    _oracle_gain,
)
from isac_sim.receiver.cancellation_glrt.probe import _low_rank_form
from isac_sim.receiver.cancellation_glrt.residual_form import ResidualModel


def _direct_mismatch_factor(cfg, obs: Observation, basis, small, gain: float) -> np.ndarray:
    """First-order DD-placement uncertainty left after the cancellation map.

    ``X`` is built at the receiver's estimated direct-path delay/Doppler.  A
    nonzero estimation sigma therefore creates two random tangent directions
    per source.  The old covariance only propagated coefficient uncertainty
    inside ``span(X)`` and silently omitted these out-of-dictionary directions;
    after LS cancellation that reduced ``C_res`` almost to the noise floor even
    when the measured structural residual was large.
    """
    if float(gain) == 0.0:
        return np.zeros((obs.y.size, 0), dtype=complex)
    sources = obs.direct_est
    if not sources:
        return np.zeros((obs.y.size, 0), dtype=complex)
    c = cfg.cancellation
    sigma_l = max(float(c.direct_estimation_sigma_delay_bins or 0.0), 0.0)
    sigma_k = max(float(c.direct_estimation_sigma_doppler_bins or 0.0), 0.0)
    if sigma_l == 0.0 and sigma_k == 0.0:
        return np.zeros((obs.y.size, 0), dtype=complex)

    from isac_sim.receiver.cancellation.dictionaries import direct_dictionary

    step = 1.0e-3
    columns = []
    for src in sources:
        if sigma_k > 0.0:
            plus = direct_dictionary(
                cfg, [replace(src, doppler_bin=float(src.doppler_bin) + step)],
                tangent_order=0,
            )[:, 0]
            minus = direct_dictionary(
                cfg, [replace(src, doppler_bin=float(src.doppler_bin) - step)],
                tangent_order=0,
            )[:, 0]
            columns.append(sigma_k * (plus - minus) / (2.0 * step))
        if sigma_l > 0.0:
            plus = direct_dictionary(
                cfg, [replace(src, delay_bin=float(src.delay_bin) + step)],
                tangent_order=0,
            )[:, 0]
            minus = direct_dictionary(
                cfg, [replace(src, delay_bin=float(src.delay_bin) - step)],
                tangent_order=0,
            )[:, 0]
            columns.append(sigma_l * (plus - minus) / (2.0 * step))
    if not columns:
        return np.zeros((obs.y.size, 0), dtype=complex)
    factor = np.column_stack(columns)
    if basis.shape[1]:
        factor = factor - basis @ (small @ (basis.conj().T @ factor))
    return float(gain) * factor


def residual_model(
    cfg,
    obs: Observation,
    name: str,
    results: dict[str, CancellationResult] | None = None,
    *,
    plans: dict[str, object] | None = None,
    direct_variance: str = "prior",
    check: bool = True,
    dictionary: str = "belief",
) -> ResidualModel:
    """重建一条臂的仿射形式，并写出它留下的**参考式**协方差::

        C_res = sigma^2 I  +  T_d X R_h X^H T_d^H  +  (M X) C_h (M X)^H / n_cpi
                ^噪声地板     ^残留的直连场           ^系数误差
                            +  B B^H                 （可选，见下）
                            ^信念误差造成的模板失配

    ``dictionary="truth"`` 用真实目标状态造模板，那就不存在失配可计费；
    ``dictionary="belief"``（默认，也是所有发布数字用的口径）在
    ``cancellation.belief_error_in_cres`` 打开时才计费。**传一个口径却按另一个
    计费**，是让一次标定测量变得毫无意义的最容易的办法。

    地板取 ``sigma^2 I`` 而**不是** ``sigma^2 (I-F)(I-F)^H``，这是一个值得声明的
    建模决定：在"自己要清理的那份观测"上拟合系数的抵消器，也会把被拟合子空间里
    的噪声一并去掉，于是它的残余协方差在 ``span(M X)`` 上是**奇异**的（此处是
    4096 维中的 14 维）—— 一个检测器绝不能白化进去的零空间。
    ``cancellation.n_cpi`` 已经声明系数来自**参考**预算，按这个读法，系数误差
    是独立噪声，``C_res`` 才有一个真正的满秩地板，参考预算也才总算能出现在检测
    指标里（V1 测过它买不到抵消深度，因为保留项占主导；在这里它买到的是标定）。
    """
    if results is None or name not in results:
        raise KeyError("residual_model needs the recorded result of arm %r" % (name,))
    plan = (plans if plans is not None else arm_plans(cfg, obs, results))[name]
    sigma2 = float(obs.sigma2)
    n_bins = int(obs.y.size)
    label, r_diag = _direct_prior(cfg, obs, plan)
    if direct_variance == "zero":
        r_diag = np.zeros_like(r_diag)
        label = "zero"
    elif direct_variance != "prior":
        raise ValueError("direct_variance must be 'prior' or 'zero', got %r" % (direct_variance,))

    if plan.kind == "estimator":
        basis, small, error = _low_rank_form(cfg, obs, plan)
        const = np.zeros(n_bins, dtype=complex)
        gain = 1.0
        subtraction_dictionary = (
            obs.X if (plan.candidates or plan.soft_mu is not None)
            else plan.subspace.complement_matrix(obs.X)
        )
        est_factor = subtraction_dictionary @ _psd_sqrt(
            _estimator_covariance(cfg, obs, plan)
        )
        est_factor = est_factor / math.sqrt(float(cfg.cancellation.n_cpi))
    else:
        basis = np.zeros((n_bins, 0), dtype=complex)
        small = np.zeros((0, 0), dtype=complex)
        const = obs.y - results[name].residual
        gain = _oracle_gain(cfg, name)
        error = 0.0
        est_factor = np.zeros((n_bins, 0), dtype=complex)

    X = obs.X
    ix = X - (basis @ (small @ (basis.conj().T @ X))) if basis.shape[1] else X
    direct_factor = (gain * ix) * np.sqrt(r_diag)[None, :]

    if cfg.cancellation.belief_error_in_cres and dictionary == "belief":
        belief_factor = _belief_error_factor(cfg, obs)
    else:
        belief_factor = np.zeros((n_bins, 0), dtype=complex)

    # 只有**扰动**进低秩因子。``basis`` 张成 F 的值域，绝不能加进来：噪声地板
    # 是 sigma^2 I 而不是 sigma^2 (I-F)(I-F)^H，所以 F 的值域本身不带任何协方差
    # —— 只有 ``(I-F) X``（抵消器给直连场留下的那部分）、系数误差和信念误差
    # 失配才带。把它加进去曾把条件数顶到 1e13，让 ``eigh`` 报出低于地板的
    # 最小特征值。
    direct_mismatch_factor = _direct_mismatch_factor(
        cfg, obs, basis, small, gain
    )
    blocks = [b for b in (
        direct_factor, direct_mismatch_factor, est_factor, belief_factor
    ) if b.shape[1]]

    def c_apply(V: np.ndarray) -> np.ndarray:
        """``C_res @ V``，全程不构造 K x K 矩阵。"""
        out = sigma2 * V
        for b in blocks:
            out = out + b @ (b.conj().T @ V)
        return out

    if blocks:
        Q0, _ = np.linalg.qr(np.concatenate(blocks, axis=1))
        C_red = Q0.conj().T @ c_apply(Q0)
        C_red = 0.5 * (C_red + C_red.conj().T)
        mu, V = np.linalg.eigh(C_red)
    else:
        Q0 = np.zeros((n_bins, 0), dtype=complex)
        mu = np.zeros(0)
        V = np.zeros((0, 0), dtype=complex)
    W = Q0 @ V if Q0.shape[1] else np.zeros((n_bins, 0), dtype=complex)
    min_ratio = float(mu.min() / sigma2) if mu.size else 1.0
    # ``C_red`` is analytically sigma2*I plus PSD terms.  With the direct-DD
    # mismatch block its condition number can exceed 1e6, and ``eigh`` may put
    # a floor eigenvalue a few ulps below sigma2.  Clip only that numerical
    # neighbourhood; a materially sub-floor eigenvalue remains a hard error.
    if min_ratio >= 1.0 - 1e-6:
        mu = np.maximum(mu, sigma2)
        min_ratio = float(mu.min() / sigma2) if mu.size else 1.0
    if check and min_ratio < 1.0 - 1e-6:
        raise RuntimeError(
            "arm %r: the modelled residual covariance has an eigenvalue below "
            "the noise floor (min / sigma^2 = %.6f).  C_res is a sum of PSD "
            "terms plus sigma^2 I, so this means the stated belief is "
            "inconsistent with the arm rather than a typo in the experiment."
            % (name, min_ratio)
        )
    keep = np.abs(mu - sigma2) > 1e-12 * sigma2
    cov = LowRankCovariance(
        sigma2=sigma2, W=W[:, keep], lam=mu[keep] - sigma2, min_ratio=min_ratio
    )
    model = ResidualModel(
        name=name,
        basis=basis,
        small=small,
        const=const,
        direct_gain=gain,
        sigma2=sigma2,
        cov=cov,
        probe_error=error,
        direct_variance=label,
        r_diag=r_diag,
    )
    if check:
        _check_covariance_trace(cov, blocks, sigma2, n_bins, name)
    return model
