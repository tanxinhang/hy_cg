import numpy as np
import pytest
from scipy.stats import gamma, norm
from isac_sim.mixture_probability import mixture_tail


@pytest.mark.parametrize("h", [0, 1])
@pytest.mark.parametrize("chi", [.3, 1.0])
def test_singleton_matches_direct_mixture(h, chi):
    g, k, w, t = .7, 16, .4, .8
    a = g/(1+g)
    expected = chi*gamma.sf(t/(w*a)+k, k, scale=1+h*g)
    expected += (1-chi)*norm.sf(t/(w*np.sqrt(9*k)*a))
    result = mixture_tail([g], [chi], [w], t, hypothesis=h)
    assert result.converged
    assert abs(result.midpoint-expected) < 1e-14


@pytest.mark.parametrize("h", [0, 1])
@pytest.mark.parametrize("n", [2, 6])
def test_convolution_brackets_closed_form_equal_scale_gamma(h, n):
    g, k, w, t = .5, 16, .7, 2.
    expected = gamma.sf(t/(w*g/(1+g))+n*k, n*k, scale=1+h*g)
    result = mixture_tail([g]*n, [1]*n, [w]*n, t, hypothesis=h)
    assert result.converged
    assert result.lower <= expected <= result.upper
    assert result.upper-result.lower <= 1e-3
    assert result.omitted_mass < 1e-8


def test_budget_exhaustion_is_explicit_and_atoms_use_strict_tail():
    result = mixture_tail([.4, .9], [.4, .5], [1, 1], 1, max_bins=32)
    assert not result.converged
    assert result.lower == 0 and result.upper == 1
    for threshold, expected in [(0, 0), (-1, 1), (1, 0)]:
        result = mixture_tail([.4, .9], [0, 0], [1, 1], threshold, failure_factor=0)
        assert result.midpoint == expected


def test_scaling_weights_and_threshold_preserves_probability():
    args = ([.2, .7, 1.1], [.3, .7, 1], [.3, .8, .2])
    r1 = mixture_tail(*args, 1.5)
    r2 = mixture_tail(args[0], args[1], np.array(args[2])*1e4, 1.5e4)
    assert abs(r1.midpoint-r2.midpoint) < 1e-9


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        mixture_tail([.4], [1.2], [1], 0)
    with pytest.raises(ValueError):
        mixture_tail([.4], [.7], [-1], 0)


def test_all_failed_gaussian_sum_contains_exact_normal_tail():
    g = np.array([.2, .7, 1.1])
    w = np.array([.3, .8, .2])
    sd = np.sqrt(np.sum(w*w*9*16*(g/(1+g))**2))
    exact = norm.sf(.8/sd)
    result = mixture_tail(g, [0, 0, 0], w, .8)
    assert result.converged
    assert result.lower <= exact <= result.upper


def test_erasure_singleton_atom_at_threshold_is_not_detected():
    result = mixture_tail([.5], [.3], [1], 0, failure_factor=0)
    assert abs(result.midpoint-.3*gamma.sf(16, 16, scale=1.5)) < 1e-14


def test_erasure_convolution_brackets_exact_binomial_gamma_mixture():
    from math import comb
    n, g, chi, k, threshold = 3, .4, .6, 16, .8
    a = g/(1+g)
    exact = sum(comb(n, r)*chi**r*(1-chi)**(n-r)*
                gamma.sf(threshold/a+r*k, r*k, scale=1+g)
                for r in range(1, n+1))
    result = mixture_tail([g]*n, [chi]*n, [1]*n, threshold, failure_factor=0)
    assert result.converged
    assert result.lower <= exact <= result.upper
