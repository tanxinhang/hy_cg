"""Audit of the unified-interference / SINR-guard changes.

Every check is a *numerical* assertion, not a code reading.  The point is to
find anything that would silently poison a large re-run before it is launched.

Run:  python tools/audit_coupling_changes.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from isac_sim.config import PRESETS, Config, apply_overrides, apply_preset, iter_leaf_paths  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    denominator_guard,
    generate_geometry,
    noise_power,
)
from isac_sim.simulate import run_simulation  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def finite_max(a: np.ndarray, default: float = 0.0) -> float:
    """NaN-safe maximum.

    ``max(0.0, nan)`` silently returns 0.0 in Python, which is exactly how a
    negative-SINR bug can masquerade as "no change".  Any non-finite entry is
    treated as a failure of its own, so we must not reduce with plain max/min.
    """
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return default
    if not np.all(np.isfinite(a)):
        FAILURES.append("non-finite value reached a max/min reduction")
        return float("nan")
    return float(np.max(a))


def finite_min(a: np.ndarray, default: float = 0.0) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return default
    if not np.all(np.isfinite(a)):
        FAILURES.append("non-finite value reached a max/min reduction")
        return float("nan")
    return float(np.min(a))


def health(cfg: Config, base, name: str) -> bool:
    """Assert every SINR-like table entry is finite and non-negative."""
    tb = compute_link_tables(cfg, base)
    for attr in ("gamma_comm", "rate", "chi_comm", "raw_gamma_sense",
                 "gamma_sense", "rinr", "mu_soft"):
        v = np.asarray(getattr(tb, attr), dtype=float)
        if not np.all(np.isfinite(v)):
            print(f"[FAIL] {name}: {attr} contains non-finite entries")
            FAILURES.append(f"{name}:{attr}:non-finite")
            return False
        if attr in ("gamma_comm", "rate", "chi_comm", "raw_gamma_sense",
                    "gamma_sense", "rinr") and np.any(v < 0):
            print(f"[FAIL] {name}: {attr} contains negative entries "
                  f"(min = {float(np.min(v)):.3e})")
            FAILURES.append(f"{name}:{attr}:negative")
            return False
    return True


def trial_objects(cfg: Config, t: int = 0):
    rng = np.random.default_rng([cfg.run.seed, t])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    return base


print("=" * 78)
print("AUDIT: unified interference field + SINR guard")
print("=" * 78)

cfg0 = Config()
n0 = noise_power(cfg0)
print(f"n0 = {n0:.6e} W   legacy guard = {denominator_guard(cfg0, n0):.3e}   "
      f"noise_relative guard = {denominator_guard(apply_overrides(cfg0, {'radio.eps_mode': 'noise_relative'}), n0):.3e}")

# --------------------------------------------------------------------------
# T0. Pre-flight health: every SINR-like table must be finite and non-negative
#     under BOTH models.  A negative interference term (the failure mode of a
#     mis-subtracted field) shows up here immediately.
# --------------------------------------------------------------------------
cfg_leg_pre = apply_overrides(cfg0, {"interference.coupling": "legacy",
                                     "radio.eps_mode": "legacy"})
cfg_cpl_pre = apply_overrides(cfg0, {"interference.coupling": "shared_spectrum",
                                     "radio.eps_mode": "legacy"})
cfg_leg = apply_overrides(cfg0, {"interference.coupling": "legacy",
                                 "radio.eps_mode": "noise_relative"})
cfg_cpl = apply_overrides(cfg0, {"interference.coupling": "shared_spectrum",
                                 "radio.eps_mode": "noise_relative"})
cfg_cpl_guard = cfg_cpl
cfg_strict = apply_preset(cfg0, "isac-consistent-strict")

print("\nT0  table health under both models")
health_ok = True
for t in range(3):
    base = trial_objects(cfg0, t)
    M = cfg0.scale.M
    mask = np.zeros(M, dtype=bool)
    mask[[0, 3]] = True
    for tag, cfg_v, m in (("legacy", cfg_leg_pre, None),
                          ("legacy[mask]", cfg_leg_pre, mask),
                          ("coupled", cfg_cpl_pre, None),
                          ("coupled[mask]", cfg_cpl_pre, mask),
                          ("coupled+guard", cfg_cpl_guard, mask),
                          ("strict", cfg_strict, None)):
        health_ok &= health(cfg_v, base, f"t{t}:{tag}")
check("T0 all SINR tables finite and non-negative", health_ok)

# --------------------------------------------------------------------------
# T1. The coupled communication interference.
# --------------------------------------------------------------------------
print("\nT1  coupled vs legacy communication interference")
r = cfg0.radio
P_sense_v = r.rho * r.P_default
eps_s = cfg0.comm.comm_leakage_from_sensing

eq_nomask = 0.0          # max relative diff with no active-set gating
worst_pred = 0.0         # max relative error of the predicted masked difference

for t in range(5):
    base = trial_objects(cfg0, t)
    M = cfg0.scale.M
    mask = np.zeros(M, dtype=bool)
    mask[[0, 2, 3, 5]] = True
    G = base.direct_gain

    # --- (a) with no gating the coupled comm side must be bit-identical ------
    ga = compute_link_tables(cfg_leg, base, active_tx_mask=None).gamma_comm
    gb = compute_link_tables(cfg_cpl, base, active_tx_mask=None).gamma_comm
    nz = ga > 0
    if nz.any():
        eq_nomask = max(eq_nomask, finite_max(np.abs(ga[nz] - gb[nz]) / ga[nz]))

    # --- (b) with gating the only difference must be the *ungated* sensing
    #         leakage of the silent transmitters, exactly -------------------
    ta = compute_link_tables(cfg_leg, base, active_tx_mask=mask)
    tb = compute_link_tables(cfg_cpl, base, active_tx_mask=mask)
    ga, gb = ta.gamma_comm, tb.gamma_comm
    check_here = np.isfinite(gb).all() and (gb >= 0).all()
    if not check_here:
        check("T1 coupled comm gamma finite & non-negative [masked]", False,
              f"min = {float(np.min(gb)):.3e}")
        continue
    P_comm_v = (1.0 - r.rho) * r.P_default
    for i in range(M):
        for jj in range(M):
            if i == jj or not base.edge_mask[i, jj] or ga[i, jj] <= 0:
                continue
            signal = P_comm_v * G[i, jj]
            delta_den = signal * (1.0 / gb[i, jj] - 1.0 / ga[i, jj])
            pred = eps_s * sum(
                P_sense_v * G[k, jj]
                for k in range(M)
                if k != i and k != jj and not mask[k]
            )
            scale = max(abs(pred), 1e-30)
            worst_pred = max(worst_pred, abs(delta_den - pred) / scale)

check("T1a without gating, coupled == legacy communication SINR",
      eq_nomask < 1e-12, f"max relative diff = {eq_nomask:.3e}")
check("T1b with gating, the difference is exactly the ungated leakage of the "
      "silent UAVs", worst_pred < 1e-9, f"max relative error = {worst_pred:.3e}")
# and that difference must be an *increase* in interference (never a decrease)
never_lower = True
for t in range(5):
    base = trial_objects(cfg0, t)
    M = cfg0.scale.M
    mask = np.zeros(M, dtype=bool)
    mask[[0, 2, 3, 5]] = True
    ga = compute_link_tables(cfg_leg, base, active_tx_mask=mask).gamma_comm
    gb = compute_link_tables(cfg_cpl, base, active_tx_mask=mask).gamma_comm
    nz = (ga > 0) & (gb > 0)
    if nz.any() and not bool(np.all(gb[nz] <= ga[nz] + 1e-15)):
        never_lower = False
check("T1c coupling never *reduces* communication interference", never_lower)

# --------------------------------------------------------------------------
# T2. The fast path (reuse_from) must agree with a full rebuild under the
#     coupled model whenever the sensing block is schedule-independent.
# --------------------------------------------------------------------------
print("\nT2  fast-path reuse vs full rebuild under shared_spectrum")
worst = 0.0
for t in range(5):
    base = trial_objects(cfg0, t)
    M = cfg0.scale.M
    mask = np.zeros(M, dtype=bool)
    mask[[1, 4, 6]] = True
    full = compute_link_tables(cfg_cpl, base)
    reused = compute_link_tables(cfg_cpl, base, active_tx_mask=mask, reuse_from=full)
    rebuilt = compute_link_tables(cfg_cpl, base, active_tx_mask=mask)
    for name in ("gamma_sense", "rinr", "mu_soft", "var0_q", "raw_gamma_sense"):
        x = getattr(reused, name)
        y = getattr(rebuilt, name)
        nz = np.abs(y) > 0
        if nz.any():
            worst = max(worst, float(np.max(np.abs(x[nz] - y[nz]) / np.abs(y[nz]))))
check("T2 reuse path matches rebuild", worst == 0.0, f"max relative diff = {worst:.3e}")

# T2b. With sense_gate_by_active_tx the fast path must be DISABLED.
cfg_gate = apply_overrides(cfg_cpl, {"interference.sense_gate_by_active_tx": True})
base = trial_objects(cfg0, 0)
M = cfg0.scale.M
mask = np.zeros(M, dtype=bool)
mask[[1, 4, 6]] = True
full = compute_link_tables(cfg_gate, base)
reused = compute_link_tables(cfg_gate, base, active_tx_mask=mask, reuse_from=full)
rebuilt = compute_link_tables(cfg_gate, base, active_tx_mask=mask)
same_as_full = np.allclose(reused.rinr, full.rinr)
same_as_rebuilt = np.allclose(reused.rinr, rebuilt.rinr)
check("T2b gated sensing forces a rebuild", same_as_rebuilt and not same_as_full,
      f"matches rebuild={same_as_rebuilt}, still equals selection stage={same_as_full}")

# --------------------------------------------------------------------------
# T3. Guard fix: the guard is *global*, so quantify (do not assume) how far it
#     moves the communication side.  Expected behaviour:
#       * SINR can only go UP  (the old guard inflated every denominator);
#       * the change is bounded by the guard ratio (1e-12 / n0 = 26.1x = 14.2 dB);
#       * strong-interference links barely move (interference >> guard).
# --------------------------------------------------------------------------
print("\nT3  guard fix impact on the communication side")
guard_ratio = denominator_guard(cfg0, n0) / n0
max_up_db = 0.0
max_down_db = 0.0
worst_chi = 0.0
flip_pairs = 0
total_pairs = 0
for t in range(5):
    base = trial_objects(cfg0, t)
    M = cfg0.scale.M
    mask = np.zeros(M, dtype=bool)
    mask[[0, 2, 5]] = True
    a = compute_link_tables(
        apply_overrides(cfg0, {"interference.coupling": "shared_spectrum"}),
        base, active_tx_mask=mask)
    b = compute_link_tables(cfg_cpl, base, active_tx_mask=mask)
    nz = (a.gamma_comm > 0)
    d_db = 10 * np.log10(b.gamma_comm[nz] / a.gamma_comm[nz])
    max_up_db = max(max_up_db, finite_max(d_db))
    max_down_db = min(max_down_db, finite_min(d_db))
    worst_chi = max(worst_chi, finite_max(
        np.abs(a.chi_comm[nz] - b.chi_comm[nz]) / a.chi_comm[nz]))
    flip_pairs += int(np.sum(a.feasible_comm != b.feasible_comm))
    total_pairs += int(a.feasible_comm.size)
bound_db = 10 * np.log10(1.0 + guard_ratio)
print(f"    guard ratio = {guard_ratio:.2f}x  -> hard upper bound {bound_db:.2f} dB")
print(f"    comm SINR change:  max {max_up_db:+.2f} dB / {max_down_db:+.2f} dB")
print(f"    comm chi max relative change  = {worst_chi:.2%}")
print(f"    feasibility flips: {flip_pairs} of {total_pairs} edge slots")
check("T3 comm SINR never decreases under the guard fix", max_down_db > -1e-9,
      f"min change = {max_down_db:+.3e} dB")
check("T3b comm SINR gain bounded by the guard ratio", max_up_db <= bound_db + 1e-6,
      f"{max_up_db:.2f} dB <= {bound_db:.2f} dB")
check("T3c feasibility flips are rare (<2% of edge slots)",
      flip_pairs <= 0.02 * total_pairs, f"{flip_pairs}/{total_pairs}")

# --------------------------------------------------------------------------
# T4. Detection-side numerics across the whole cancellation sweep.
# --------------------------------------------------------------------------
print("\nT4  numerical health across the cancellation sweep (MC=30)")
cfg_base = apply_overrides(cfg0, {"interference.coupling": "shared_spectrum",
                                  "radio.eps_mode": "noise_relative",
                                  "comm.interference_model": "active_set"})
bad: list[str] = []
prev = None
monotone_ok = True
for db in (0.0, 20.0, 40.0, 80.0):
    cfg = apply_overrides(cfg_base, {"interference.direct_cancellation_db": db})
    cfg.run.num_mc = 30
    cfg.run.verbose = False
    summary = run_simulation(cfg)
    p = summary["proposed_lagrangian"]
    for key in ("P_D", "P_FA", "P_FA_overall", "D_mean", "T_mean_ms", "B_mean_bits",
                "worst_target_satisfied_prob", "actual_worst_target_P_D"):
        v = p.get(key)
        if v is None or not np.isfinite(v):
            bad.append(f"kappa={db}dB {key}={v}")
    if not (0.0 <= p["P_D"] <= 1.0):
        bad.append(f"kappa={db}dB P_D out of range: {p['P_D']}")
    if p["P_D_ci95"][0] > p["P_D"] or p["P_D_ci95"][1] < p["P_D"]:
        bad.append(f"kappa={db}dB CI does not bracket P_D")
    if prev is not None and p["P_D"] < prev - 0.05:
        monotone_ok = False
    prev = p["P_D"]
    print(f"    kappa={db:5.1f} dB  P_D={p['P_D']:.4f}  D={p['D_mean']:.3f}  "
          f"T={p['T_mean_ms']:.2f} ms  links={p['selected_links_mean']:.1f}")
check("T4 all summary metrics finite and in range", not bad, "; ".join(bad[:4]))
check("T4b P_D non-decreasing in cancellation (tol 0.05)", monotone_ok)

# --------------------------------------------------------------------------
# T5. Config machinery: presets must not mutate the default config, and every
#     new field must be reachable through --set / iter_leaf_paths.
# --------------------------------------------------------------------------
print("\nT5  configuration machinery")
before = dict(iter_leaf_paths(Config()))
_ = apply_preset(Config(), "isac-consistent")
after = dict(iter_leaf_paths(Config()))
check("T5 apply_preset does not mutate the default", before == after)
paths = set(after)
for want in ("interference.coupling", "interference.direct_cancellation_db",
             "interference.sense_gate_by_active_tx", "radio.eps_mode", "radio.eps_rel_db"):
    if want not in paths:
        FAILURES.append(f"missing config path {want}")
        print(f"[FAIL] T5 missing config path {want}")
else:
    print(f"[PASS] T5 all new config paths enumerable ({len(paths)} leaves)")
for key in ("isac-consistent", "isac-consistent-strict"):
    if key not in PRESETS:
        FAILURES.append(f"missing preset {key}")
# invalid values must raise, not silently fall back
try:
    apply_preset(Config(), "nope")
    check("T5c unknown preset raises", False)
except KeyError:
    check("T5c unknown preset raises", True)

# --------------------------------------------------------------------------
# T6. Coupled model must not change the *evaluation-side* physics: the DD gains
#     and the target gains are untouched by the interference coupling.
# --------------------------------------------------------------------------
print("\nT6  physical quantities unchanged by the coupling switch")
base = trial_objects(cfg0, 0)
a = compute_link_tables(cfg_leg, base)
b = compute_link_tables(cfg_cpl, base)
dd_ok = np.allclose(a.raw_gamma_sense, b.raw_gamma_sense, rtol=1e-12)
check("T6 raw sensing SINR identical (DD/target/processing gain untouched)", dd_ok,
      f"max abs diff = {float(np.max(np.abs(a.raw_gamma_sense - b.raw_gamma_sense))):.3e}")

# --------------------------------------------------------------------------
# T7. Ordering / decision stability: does the corrected model change which
#     method wins?  This is what the paper claims, so verify it.
# --------------------------------------------------------------------------
print("\nT7  method ordering, legacy vs corrected (MC=60)")
order = ["proposed_lagrangian", "topk_deflection", "sense_sinr", "single_best",
         "nearest", "shortest_bistatic", "random", "all_neighbor"]
for tag, ov in [("legacy(active_set)", {"comm.interference_model": "active_set"}),
                ("corrected(active_set)", PRESETS["isac-consistent"])]:
    cfg = apply_overrides(cfg0, ov)
    cfg.run.num_mc = 60
    cfg.run.verbose = False
    s = run_simulation(cfg)
    ranked = sorted(order, key=lambda m: -s[m]["P_D"])
    print(f"    {tag:22s} " + "  ".join(f"{m}={s[m]['P_D']:.3f}" for m in order[:5]))
    print(f"      ranking: {' > '.join(ranked[:5])}")
    if tag.startswith("corrected"):
        check("T7 proposed beats sense_sinr under the corrected model",
              s["proposed_lagrangian"]["P_D"] > s["sense_sinr"]["P_D"],
              f"{s['proposed_lagrangian']['P_D']:.4f} vs {s['sense_sinr']['P_D']:.4f}")

# --------------------------------------------------------------------------
# T8. Operating point.  rerun_paper.py pins comm.interference_model=active_set,
#     so an experiment that inherits the package default (full_concurrent) is
#     not comparable.  Measure the gap, and verify the harness now *records*
#     the operating point on every row so this cannot go unnoticed again.
# --------------------------------------------------------------------------
print("\nT8  operating-point sensitivity (MC=30) and row provenance")
vals: Dict[str, float] = {}
for tag, model in (("full_concurrent", "full_concurrent"), ("active_set", "active_set")):
    cfg = apply_overrides(cfg_base, {"comm.interference_model": model})
    cfg.run.num_mc = 30
    cfg.run.verbose = False
    s = run_simulation(cfg)["proposed_lagrangian"]
    vals[tag] = s["P_D"]
    print(f"    {tag:18s} P_D={s['P_D']:.4f}  T={s['T_mean_ms']:.2f} ms  "
          f"links={s['selected_links_mean']:.1f}")
print(f"    => the two operating points differ by {abs(vals['active_set'] - vals['full_concurrent']):.4f} "
      f"in P_D, so every experiment MUST be run at the paper's active_set point")

from isac_sim.experiments import interference_consistency  # noqa: E402

probe_cfg = apply_overrides(cfg_base, {"run.num_mc": 2})
probe_cfg.run.verbose = False
probe_rows = interference_consistency(probe_cfg, [40.0])
need = {"interference_model", "coupling", "eps_mode", "direct_cancellation_db"}
have = set(probe_rows[0].keys())
missing = need - have
check("T8b every result row records its operating point", not missing,
      f"missing keys: {sorted(missing)}" if missing else
      f"recorded: {sorted(need)}")

# --------------------------------------------------------------------------
# T9. A mistyped switch value must raise, never silently reproduce legacy.
# --------------------------------------------------------------------------
print("\nT9  mistyped switches must raise")
base = trial_objects(cfg0, 0)
for path, bad in (("interference.coupling", "shared_spectrUm"),
                  ("radio.eps_mode", "noise_rel")):
    try:
        compute_link_tables(apply_overrides(Config(), {path: bad}), base)
        check(f"T9 {path}={bad!r} raises", False, "silently accepted")
    except ValueError:
        check(f"T9 {path}={bad!r} raises", True)

# --------------------------------------------------------------------------
# T10. Variant labels must be truthful even when the caller's config already
#      carries the switches.  Run the bookkeeping part on a deliberately
#      "contaminated" base and verify the labels still describe reality.
# --------------------------------------------------------------------------
print("\nT10 variant labels are truthful on a contaminated base (MC=8)")

cfg_base_overrides = {"interference.coupling": "shared_spectrum",
                      "radio.eps_mode": "noise_relative"}


def bookkeeping_gammas(cfg: Config) -> Dict[str, float]:
    contaminated = apply_overrides(apply_overrides(cfg, cfg_base_overrides), {"run.num_mc": 8})
    contaminated.run.verbose = False
    rows = interference_consistency(contaminated, [40.0])
    out: Dict[str, float] = {}
    for r in rows:
        if r.get("group") == "bookkeeping" and r.get("method") == "proposed_lagrangian":
            out[str(r["variant"])] = float(r["sense_sinr_db"])
    return out


probe_cfg = apply_overrides(cfg0, {"run.num_mc": 8})
probe_cfg.run.verbose = False
gammas = bookkeeping_gammas(probe_cfg)
for k in sorted(gammas):
    print(f"    {k:15s} gamma^s = {gammas[k]:7.2f} dB")
ok = (gammas.get("legacy", 0.0) < gammas.get("coupled+guard", 0.0) - 6.0
      and abs(gammas.get("legacy", 0.0) - gammas.get("coupled", 0.0)) < 0.5
      and abs(gammas.get("legacy+guard", 0.0) - gammas.get("coupled+guard", 0.0)) < 0.5)
check("T10 labels stay truthful regardless of the launching config", ok,
      "legacy must sit ~12 dB below coupled+guard")

print("\n" + "=" * 78)
if FAILURES:
    print(f"AUDIT FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
    raise SystemExit(1)
print("AUDIT PASSED -- no silent numerical or methodological problem found")
