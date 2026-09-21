"""Scratch: check where the between-group term lives.

Two claims to verify:

C1  ``_mix`` reproduces the paper's eq:received_moments exactly, i.e. with the
    report-failure branch at (m, v) = (0, a*v0),

        v0_mix = chi*v0 + (1-chi)*a*v0                    (no between term)
        v1_mix = chi*v1 + (1-chi)*a*v0 + chi*(1-chi)*mu^2 (between term present)

C2  ``deflection_variance_for_link`` therefore returns the H0 variance v0_mix,
    which is exactly the denominator of the paper's single-observation
    deflection d(chi) = chi^2*delta^2 / [v0*(a+(1-a)*chi)].

Both are pure-arithmetic statements, so they are checked here without building
link tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isac_sim.sensing.soft_channel import _mix  # noqa: E402

rng = np.random.default_rng(7)

a = 9.0  # cfg.detect.soft_error_sigma_scale, the archived V1 surrogate
worst_v0 = worst_v1 = 0.0
worst_paper_d = 0.0

for _ in range(20000):
    chi = float(rng.uniform(0.05, 1.0))
    v0 = float(rng.uniform(1e-3, 10.0))
    v1 = float(rng.uniform(1e-3, 10.0))
    mu = float(rng.uniform(0.0, 50.0))

    m0, got_v0 = _mix(chi, 0.0, v0, 0.0, a * v0)
    m1, got_v1 = _mix(chi, mu, v1, 0.0, a * v0)

    pred_v0 = chi * v0 + (1.0 - chi) * a * v0
    pred_v1 = chi * v1 + (1.0 - chi) * a * v0 + chi * (1.0 - chi) * mu ** 2

    worst_v0 = max(worst_v0, abs(got_v0 - pred_v0) / max(abs(got_v0), 1e-300))
    worst_v1 = max(worst_v1, abs(got_v1 - pred_v1) / max(abs(got_v1), 1e-300))

    # paper d(chi) = chi^2 delta^2 / (v0 (a + (1-a) chi))
    paper_d = chi ** 2 * mu ** 2 / (v0 * (a + (1.0 - a) * chi))
    # implemented d = (delta_eff)^2 / v0_mix  with delta_eff = chi*mu
    impl_d = (chi * mu) ** 2 / got_v0
    worst_paper_d = max(worst_paper_d, abs(impl_d - paper_d) / max(abs(paper_d), 1e-300))

print(f"C1 max rel err  v0_mix vs chi*v0+(1-chi)*a*v0                  : {worst_v0:.3e}")
print(f"C1 max rel err  v1_mix vs chi*v1+(1-chi)*a*v0+chi(1-chi)mu^2   : {worst_v1:.3e}")
print(f"C2 max rel err  implemented d(chi) vs paper d(chi)            : {worst_paper_d:.3e}")
print()
print("read: <1e-12 => the between term sits in v1 and the deflection")
print("      denominator is the H0 variance, exactly as the manuscript states.")
