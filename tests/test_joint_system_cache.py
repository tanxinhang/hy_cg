"""联合系统的缓存必须**逐位恒等**，而不是"差不多"。

``_ArmSolveMixin.solve`` 在一臂的生命期里被调用二十几次（5 个分量、矩一次、
十几个 sigma 点）。正规方程矩阵 ``D^H D + diag(lam)`` 与后验协方差只依赖
``(X, A_cand, sigma2, pv)`` —— 与被拟合的右端**无关**。所以逐次重建是纯浪费，
缓存它是合法的提速。

但"合法"必须被证明，不能被假定：这里的提速如果偷偷改了数值，它就成了
一个静默改变发布口径的优化。所以本文件把承诺写成断言 ——
**给不给 ``system``，结果必须逐位相同**。
"""

from __future__ import annotations

import numpy as np
import pytest

from isac_sim.receiver.cancellation.joint import (
    joint_interference_covariance,
    joint_refine,
    joint_system,
    marginalized_interference_map,
)


def _case(n_bins: int = 96, d: int = 9, m: int = 4, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n_bins, d)) + 1j * rng.normal(size=(n_bins, d)))
    A = (rng.normal(size=(n_bins, m)) + 1j * rng.normal(size=(n_bins, m)))
    y = rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)
    return X, A, y, 3.83e-14


@pytest.mark.parametrize("prior_variance", [None, 1e-3])
def test_marginalized_map_matches_joint_gaussian_solution(prior_variance):
    X, A, y, sigma2 = _case()
    target_prior = 2e-3
    h_joint, _ = joint_refine(
        y, X, A, sigma2, prior_variance, target_prior
    )
    h_marginal, cov_marginal = marginalized_interference_map(
        y, X, A, sigma2, prior_variance, target_prior
    )
    cov_joint = joint_interference_covariance(
        X, A, sigma2, prior_variance, target_prior
    )

    np.testing.assert_allclose(h_marginal, h_joint, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(cov_marginal, cov_joint, rtol=2e-10, atol=1e-24)
    np.testing.assert_allclose(
        y - X @ h_marginal, y - X @ h_joint, rtol=2e-11, atol=2e-12
    )


def test_marginalized_empty_target_limit_is_ridge_map():
    X, _A, y, sigma2 = _case()
    empty = np.zeros((X.shape[0], 0), dtype=complex)
    prior_variance = 1e-3
    h_marginal, _ = marginalized_interference_map(
        y, X, empty, sigma2, prior_variance, 1e-3
    )
    h_joint, _ = joint_refine(
        y, X, empty, sigma2, prior_variance, 1e-3
    )
    np.testing.assert_allclose(h_marginal, h_joint, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("pv", [None, 0.0, 1e-3])
def test_joint_refine_is_bit_identical_with_and_without_the_system(pv):
    """``system=`` 只是跳过矩阵重建，不是另一条数值路径。"""
    X, A, y, sigma2 = _case()
    tp = pv or 1.0
    plain = joint_refine(y, X, A, sigma2, pv, tp)
    cached = joint_refine(y, X, A, sigma2, pv, tp,
                          system=joint_system(X, A, sigma2, pv, tp))
    assert np.array_equal(plain[0], cached[0])
    assert np.array_equal(plain[1], cached[1])


@pytest.mark.parametrize("pv", [None, 0.0, 1e-3])
def test_joint_covariance_is_bit_identical_with_and_without_the_system(pv):
    X, A, _y, sigma2 = _case()
    tp = pv or 1.0
    plain = joint_interference_covariance(X, A, sigma2, pv, tp)
    cached = joint_interference_covariance(X, A, sigma2, pv, tp,
                                           system=joint_system(X, A, sigma2, pv, tp))
    assert np.array_equal(plain, cached)


def test_the_system_really_is_independent_of_the_right_hand_side():
    """前提检查：缓存之所以合法，是因为它与被拟合的向量无关。

    这条一旦失败，缓存就变成了"用上一个 sigma 点的矩阵解这一个"，而上面的
    逐位断言**测不出来**（它们每次都传同一个 system）。所以必须单独钉住。
    """
    X, A, _y, sigma2 = _case()
    rng = np.random.default_rng(11)
    sys_a = joint_system(X, A, sigma2, 1e-3, 1e-3)
    for _ in range(3):
        v = rng.normal(size=X.shape[0]) + 1j * rng.normal(size=X.shape[0])
        joint_refine(v, X, A, sigma2, 1e-3, 1e-3, system=sys_a)
    sys_b = joint_system(X, A, sigma2, 1e-3, 1e-3)
    assert np.array_equal(sys_a[0], sys_b[0])
    assert np.array_equal(sys_a[1], sys_b[1])
    assert np.array_equal(sys_a[2], sys_b[2])
    assert np.array_equal(sys_a[3], sys_b[3])
    # 同一 system 连续解不同的右端，必须给出不同的解 —— 否则说明缓存在漏用。
    v1 = rng.normal(size=X.shape[0]) + 1j * rng.normal(size=X.shape[0])
    v2 = v1 + 1.0
    h1 = joint_refine(v1, X, A, sigma2, 1e-3, 1e-3, system=sys_a)[0]
    h2 = joint_refine(v2, X, A, sigma2, 1e-3, 1e-3, system=sys_a)[0]
    assert not np.array_equal(h1, h2)


def test_empty_candidate_block_is_not_a_special_case_that_silently_changes():
    """候选集为空时 ``D == X``；走同一条代码路径，不留第二份实现。"""
    X, A, y, sigma2 = _case()
    empty = A[:, :0]
    plain = joint_refine(y, X, empty, sigma2, 1e-3, 1e-3)
    cached = joint_refine(y, X, empty, sigma2, 1e-3, 1e-3,
                          system=joint_system(X, empty, sigma2, 1e-3, 1e-3))
    assert np.array_equal(plain[0], cached[0])
    assert plain[1].size == 0 and cached[1].size == 0
