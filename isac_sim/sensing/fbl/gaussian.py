"""gaussian（自 ``isac_sim/sensing/fbl.py`` 拆出）。"""

from __future__ import annotations

import math
import numpy as np


def _qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian Q-function, accurate and fast enough for O(M^2) tables."""
    arr = np.asarray(x, dtype=float)
    # erfc is not a numpy ufunc; a small cached vectorisation keeps the table
    # build at ~microseconds for the M=15 regime used throughout.
    flat = arr.ravel()
    out = np.array([0.5 * math.erfc(v / math.sqrt(2.0)) for v in flat], dtype=float)
    out = out.reshape(arr.shape)
    return float(out) if np.ndim(x) == 0 else out


def _qinv(p: float) -> float:
    """Inverse Q-function (Acklam-style rational approximation + refinement)."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    # Initial guess via the Beasley-Springer-Moro approximation on the
    # standard normal quantile, then two Newton steps on erfc.
    x = _norm_ppf(1.0 - p)
    for _ in range(3):
        f = 0.5 * math.erfc(x / math.sqrt(2.0)) - p
        fp = -math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        step = f / fp
        x -= step
        if abs(step) < 1e-14:
            break
    return x


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
