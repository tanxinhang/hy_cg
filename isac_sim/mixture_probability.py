"""Independent Gamma/report-mixture tail probabilities with grid brackets.

An audit evaluator, not a replacement for the released selector. Brackets
follow from rounding each retained input down/up. SciPy CDF/FFT floating-point
errors are monitored, but are NOT formally interval-arithmetic certified.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import gamma as gamma_law, norm


@dataclass(frozen=True)
class TailInterval:
    lower: float
    upper: float
    step: float
    bins: int
    omitted_mass: float
    converged: bool
    floating_point_certified: bool = False

    @property
    def midpoint(self):
        return (self.lower + self.upper) / 2


def _cdf(x, g, chi, weight, looks, h, failure_factor, *, left=False):
    a = g / (1 + g)
    success = gamma_law.cdf(np.asarray(x) / (weight*a) + looks,
                            looks, scale=1+h*g)
    sd = weight * np.sqrt(failure_factor * looks) * a
    failure = norm.cdf(np.asarray(x)/sd) if sd > 0 else (
        np.asarray(x) > 0 if left else np.asarray(x) >= 0)
    return chi*success + (1-chi)*failure


def mixture_tail(gammas, successes, weights, threshold, *, looks=16,
                 hypothesis=1, failure_factor=9.0, tolerance=1e-3,
                 tail_budget=1e-9, max_bins=262144):
    """Bracket P(sum(w_e Y_e) > threshold) for fixed independent inputs.

    Gamma success and hypothesis-independent Gaussian replacement (or zero
    erasure) only. No covariance, belief averaging, or decoder model is added.
    If resolution budget is exhausted, converged=False: never hide that fact.
    Empty/all-zero statistic uses a strict threshold comparison.
    """
    g, chi, w = [np.asarray(x, dtype=float) for x in
                 (gammas, successes, weights)]
    if not (g.ndim == chi.ndim == w.ndim == 1 and g.shape == chi.shape == w.shape):
        raise ValueError("inputs must be equal-length vectors")
    if (not all(np.all(np.isfinite(x)) for x in (g, chi, w)) or
        np.any(g < 0) or np.any(w < 0) or np.any((chi < 0) | (chi > 1)) or
        not np.isfinite(threshold) or looks < 1 or int(looks) != looks or
        hypothesis not in (0, 1) or not np.isfinite(failure_factor) or failure_factor < 0 or
        not 0 < tail_budget < tolerance < 1 or max_bins < 32):
        raise ValueError("invalid distribution or accuracy parameters")
    keep = (g > 0) & (w > 0) & ((chi > 0) | (failure_factor > 0))
    g, chi, w = g[keep], chi[keep], w[keep]
    n = len(g)
    if not n:
        value = float(0 > threshold)
        return TailInterval(value, value, 0, 0, 0, True)
    if n == 1:
        value = float(1-_cdf(threshold, g[0], chi[0], w[0], looks,
                             hypothesis, failure_factor))
        return TailInterval(value, value, 0, 0, 0, True)
    a = g/(1+g)
    means = chi*looks*a*g*hypothesis
    variances = (chi*looks*a*a*(1+hypothesis*g)**2 +
                 (1-chi)*failure_factor*looks*a*a +
                 chi*(1-chi)*(looks*a*g*hypothesis)**2)
    scale = np.sqrt(np.sum(w*w*variances))
    w, threshold = w/scale, threshold/scale
    eps = tail_budget/(2*n)
    lows, highs = [], []
    for gi, ci, wi in zip(g, chi, w):
        ai = gi/(1+gi)
        ls, hs = [], []
        if ci > 0:
            ls.append(wi*ai*(gamma_law.ppf(eps, looks, scale=1+hypothesis*gi)-looks))
            hs.append(wi*ai*(gamma_law.isf(eps, looks, scale=1+hypothesis*gi)-looks))
        if ci < 1:
            sd = wi*np.sqrt(failure_factor*looks)*ai
            ls.append(sd*norm.ppf(eps))
            hs.append(sd*norm.isf(eps))
        lows.append(min(ls)); highs.append(max(hs))
    step = 1/32
    result = TailInterval(0, 1, step*scale, 0, 1, False)
    while True:
        lower_indices = np.floor(np.array(lows)/step).astype(int)
        upper_indices = np.ceil(np.array(highs)/step).astype(int) + 1
        lengths = upper_indices-lower_indices
        bins = int(lengths.sum()-n+1)
        if bins > max_bins:
            return result
        masses = []
        for gi, ci, wi, lo, hi in zip(g, chi, w, lower_indices, upper_indices):
            edges = np.arange(lo, hi+1)*step
            cdf = _cdf(edges, gi, ci, wi, looks, hypothesis, failure_factor, left=True)
            masses.append(np.maximum(np.diff(cdf), 0))
        retained = float(np.prod([m.sum() for m in masses]))
        pmf = masses[0]
        numerical_residual = 0.0
        for mass in masses[1:]:
            pmf = fftconvolve(pmf, mass)
            numerical_residual += float(np.maximum(-pmf, 0).sum())
            pmf = np.maximum(pmf, 0)
        numerical_residual += abs(float(pmf.sum())-retained)
        # Empirical roundoff allowance, explicitly not a formal machine proof.
        guard = numerical_residual + 1e-10
        origin = int(lower_indices.sum())
        def mass_above_shift(shift):
            start = int(np.floor(threshold/step-origin-shift))+1
            return float(pmf[max(start, 0):].sum()) if start < len(pmf) else 0.0
        low = max(0.0, mass_above_shift(0)-guard)
        omitted = max(0.0, 1-retained)
        high = min(1.0, mass_above_shift(n)+omitted+guard)
        result = TailInterval(low, high, step*scale, bins, omitted,
                              high-low <= tolerance)
        if result.converged:
            return result
        step /= 2


def cf_detector_intervals(cfg, tables, q, links, *, plan=None, base=None, tolerance=1e-3):
    """Audit the released CF-threshold linear statistic on supplied tables.

    Caller must supply belief-side tables for planning. This is not the current
    separately calibrated true-erasure detector. It never modifies selection.
    """
    from .fusion import compute_weights, fused_h0_variance, fused_h0_skewness
    from .reporting import report_chi
    if cfg.corr.enable or cfg.detect.soft_stat_model != "llr":
        raise ValueError("independent centered energy LLRs required")
    if cfg.detect.comm_error_model not in ("erasure", "gaussian_replacement"):
        raise ValueError("hypothesis-independent replacement required")
    if not links:
        zero = TailInterval(0, 0, 0, 0, 0, True)
        return {"pd": zero, "pfa": zero, "threshold": 0.0}
    weights = compute_weights(cfg, tables, q, links, mode="deflection", plan=plan, base=base)
    z = float(norm.isf(cfg.detect.Pfa_target))
    variance = fused_h0_variance(cfg, tables, q, links, weights, plan=plan, base=base)
    skew = fused_h0_skewness(cfg, tables, q, links, weights, plan=plan, base=base)
    threshold = (z+skew*(z*z-1)/6)*np.sqrt(max(variance, 1e-12))
    gs = [tables.gamma_sense[i, j, q] for i, j in links]
    cs = [report_chi(tables, plan, link, q) if cfg.detect.enable_comm_error_pollution else 1.
          for link in links]
    ws = [weights[link] for link in links]
    factor = cfg.detect.soft_error_sigma_scale**2 if cfg.detect.comm_error_model == "gaussian_replacement" else 0.
    result = {"threshold": float(threshold)}
    for h, name in ((0, "pfa"), (1, "pd")):
        result[name] = mixture_tail(gs, cs, ws, threshold, looks=cfg.detect.n_looks,
                                   hypothesis=h, failure_factor=factor, tolerance=tolerance)
    return result

