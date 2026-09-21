# RETIRED PREMISE (2026-09-20): this script swept / read the config field
# `interference.direct_cancellation_db` (kappa_dc).
# That field was DELETED: it asserted a fixed 40 dB direct-path cancellation with no
# receiver implementation behind it while propping up the whole SINR denominator.
# Direct-path cancellation is now only ever a MEASURED TP-UIC residual.  Running this
# script as-is will fail on the missing attribute -- kept as historical evidence only.
"""Which lever actually buys sensing performance, and how much?

Two parts, both at the paper operating point:
  (a) analytic/table: median per-link sensing SINR under each lever (no MC);
  (b) Monte-Carlo P_D for the most promising levers.

The point is to separate levers that attack the NOISE floor from levers that only
help once the receiver is INTERFERENCE-limited -- the ranking changes completely
between the two regimes.
"""
from __future__ import annotations

import io
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from isac_sim.core.config import Config, apply_overrides  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    denominator_guard,
    generate_geometry,
    noise_power,
)
from experiments.flow.simulate import run_simulation  # noqa: E402

PAPER = {
    "geometry.uav_speed_min": 30,
    "geometry.target_speed_min": 50,
    "geometry.target_speed_max": 90,
    "comm.interference_model": "active_set",
}
BASE = apply_overrides(Config(), PAPER)

LEVERS = [
    ("baseline (kappa_dc=40, N=L=64, NF=7dB, rho=0.8)",
     {}),
    ("1. direct-path cancellation +10 dB (kappa_dc=50)",
     {"interference.direct_cancellation_db": 50.0}),
    ("2. Tx power x10",
     {"radio.P_default": 10.0}),
    ("3. Tx power x100",
     {"radio.P_default": 100.0}),
    ("4. receiver NF 7 -> 3 dB",
     {"radio.noise_figure_db": 3.0}),
    ("5. processing gain x2 (L=128)",
     {"waveform.L": 128}),
    ("6. processing gain x4 (L=256)",
     {"waveform.L": 256}),
    ("7. sensing power fraction rho 0.8 -> 0.95",
     {"radio.rho": 0.95}),
    ("8. CPI n_looks 16 -> 64 (incoherent)",
     {"detect.n_looks": 64}),
    ("9. noise-limited -> interference-limited probe (kappa_dc=20)",
     {"interference.direct_cancellation_db": 20.0}),
]


def median_sensing_sinr(cfg: Config, n_trials: int = 20) -> float:
    vals = []
    for t in range(n_trials):
        rng = np.random.default_rng([cfg.run.seed, t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        tb = compute_link_tables(cfg, base)
        valid = base.valid_dd & base.edge_mask[..., None]
        i, j, q = np.argwhere(valid).T
        g = np.asarray(tb.gamma_sense[i, j, q])
        g = g[np.isfinite(g) & (g > 0)]
        if len(g):
            vals.append(float(np.median(g)))
    return 10 * np.log10(float(np.mean(vals)))


print("=" * 92)
print("PART A  median per-link sensing SINR (geometry/arithmetic, no MC)")
print("=" * 92)
base_sinr = None
rows = []
for name, ov in LEVERS:
    cfg = apply_overrides(BASE, ov)
    s = median_sensing_sinr(cfg)
    if base_sinr is None:
        base_sinr = s
    n0 = noise_power(cfg)
    rinr = 10 ** (-cfg.interference.direct_cancellation_db / 10)
    print(f"  {name:<52} gamma^s = {s:+7.2f} dB   (delta {s-base_sinr:+6.2f} dB)")
    rows.append((name, s - base_sinr))

print()
print("=" * 92)
print("PART B  effect on P_D (MC=150, proposed selector)")
print("=" * 92)
SUBSET = ["baseline (kappa_dc=40, N=L=64, NF=7dB, rho=0.8)",
          "1. direct-path cancellation +10 dB (kappa_dc=50)",
          "2. Tx power x10",
          "4. receiver NF 7 -> 3 dB",
          "5. processing gain x2 (L=128)",
          "8. CPI n_looks 16 -> 64 (incoherent)"]
print(f"  {'lever':<52} {'P_D':>8} {'worst':>8}")
for name, ov in LEVERS:
    if name not in SUBSET:
        continue
    cfg = apply_overrides(BASE, {**ov, "run.num_mc": 150, "run.verbose": False})
    with contextlib.redirect_stdout(io.StringIO()):
        s = run_simulation(cfg, methods=["proposed_lagrangian"])["proposed_lagrangian"]
    print(f"  {name:<52} {s['P_D']:8.4f} {s['actual_worst_target_P_D']:8.4f}", flush=True)

print()
print("=" * 92)
print("PART C  noise- vs interference-limited: what does more power buy?")
print("=" * 92)
n0 = noise_power(BASE)
print(f"  {'kappa_dc':>9} {'residual I/N0':>14} "
      f"{'gamma^s @P':>10} {'gamma^s @10P':>13} {'gain':>7}")
for db in (20.0, 30.0, 40.0, 50.0):
    g1 = median_sensing_sinr(apply_overrides(BASE, {"interference.direct_cancellation_db": db}))
    g10 = median_sensing_sinr(apply_overrides(BASE, {
        "interference.direct_cancellation_db": db, "radio.P_default": 10.0}))
    cfg = apply_overrides(BASE, {"interference.direct_cancellation_db": db})
    tb_rinr = None
    rng = np.random.default_rng([cfg.run.seed, 0])
    geom = generate_geometry(cfg, rng)
    bg = build_base_gains(cfg, geom, rng)
    tb = compute_link_tables(cfg, bg)
    valid = bg.valid_dd & bg.edge_mask[..., None]
    i, j, q = np.argwhere(valid).T
    resid = float(np.median(tb.rinr[i, j]))
    print(f"  {db:9.0f} {10*np.log10(resid):+13.2f} dB {g1:+9.2f} {g10:+12.2f} "
          f"{g10-g1:+6.2f} dB")
print()
print("  Reading: when the residual interference is well below the noise floor")
print("  (kappa_dc=40-50) the receiver is NOISE-limited and 10 dB more power buys")
print("  ~10 dB of SINR.  Once the residual interference dominates (kappa_dc<=30)")
print("  the receiver is INTERFERENCE-limited and the same 10 dB of power buys")
print("  almost nothing -- the lever that pays is then more CANCELLATION or fewer")
print("  concurrent illuminators, not more power.")
