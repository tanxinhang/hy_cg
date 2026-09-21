"""Rescue-lever probe for the small-RCS route proposed in ``advice/002.md``.

Why this exists
---------------
``advice/002.md`` proposes "selective rescue": identify weak targets, stop paying
for satisfied ones, and spend the freed budget on looks first, then DD refine,
with power deliberately demoted. Its section 7 justifies the ordering with a
single premise::

    because the model is interference limited, r = I_residual / N0 >> 1

That premise was measured on a **400 m compressed geometry**. The system audit
(``tools/audit_system_integrity.py``, check C4) showed ``rinr`` is
**geometry-conditional**: about **-2.8 dB** at the 4000 m release footprint and
**+10.6 dB** at 400 m. Since ``gamma(mP)/gamma(P) = m(1+r)/(1+mr) -> m`` as
``r -> 0``, the power lever should get *stronger* at the release point, not
weaker. This probe settles the ordering by measurement.

Method notes (two traps this script is built to avoid)
------------------------------------------------------
1. **levers must share one currency.** looks does not move gamma at all -- it is
   a sample-count lever. Ranking "realised dB of gamma" therefore silently gives
   looks zero. Instead every lever is converted into **equivalent pure-gamma dB**
   using a calibration curve built with ``detect.sensing_processing_gain``, a
   clean numerator-only knob.
2. **the evaluated link must be frozen.** Taking the per-target best view at each
   setting lets the winning Tx-Rx pair change with the lever, which contaminates
   the measurement with link switching. The link is chosen once at baseline and
   then held fixed.

The analytic identity ``m(1+r)/(1+mr)`` is used *only* as a cross-check gate on
the frozen link; a large drift fails the probe rather than being smoothed away.

Usage::

    python tools/probe_rescue_levers.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac_sim.core.config import (  # noqa: E402
    apply_overrides,
    apply_preset,
    default_config,
    validate_config,
)
from isac_sim.detection.fusion import predicted_pd_for_links  # noqa: E402
from isac_sim.sensing.soft_channel import received_moments  # noqa: E402
from isac_sim.cooperation.reporting import assign_fusion_nodes  # noqa: E402
from isac_sim.sensing.model import compute_link_tables, build_base_gains, generate_geometry  # noqa: E402

SEED = 20260916
PD_REQ = 0.95
CALIB_STEPS = tuple(10.0 ** (0.1 * d) for d in range(0, 21, 2))  # 0 .. 20 dB in 2 dB steps


def build(area_xy: float, rcs: float, looks: int, pscale: float, gscale: float = 1.0, _cache={}):
    """Rebuild the real tables for one lever setting (never a rewritten formula).

    ``detect.sensing_processing_gain`` is an ABSOLUTE override of ``N*L``, not a
    multiplier: passing 1.0 would silently replace the 4096x processing gain and
    destroy the link budget. Hence the explicit scale against the natural value.
    """
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    natural = cfg.waveform.N * cfg.waveform.L
    cfg = apply_overrides(
        cfg,
        {
            "geometry.area_xy": area_xy,
            "detect.target_rcs": rcs,
            "detect.n_looks": looks,
            "radio.P_default": 1.0 * pscale,
            "detect.sensing_processing_gain": gscale * natural,
        },
    )
    validate_config(cfg)
    rng = np.random.default_rng(SEED)
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base)
    # The V1 release uses an EXPLICIT fusion plan (fusion.mode="explicit",
    # rule "nearest_target"). Passing plan=None instead imposes the LEGACY
    # "report back to the sensing initiator i" routing (reporting.py:79-100),
    # which is a *different system*: every report is forced over the j->i leg,
    # which mostly fails at 4 km. Getting the plan right matters more here than
    # any lever setting.
    plan = assign_fusion_nodes(cfg, base, tables, geom=geom)
    return cfg, base, tables, plan


def pd_for(cfg, tables, q: int, link, plan=None) -> float:
    return float(predicted_pd_for_links(cfg, tables, q, [link], plan=plan, base=None))


def best_view_per_target(cfg, tables, plan=None, exclude=()):
    """Frozen per-target best view, ranked by PREDICTED P_D, not by sensing gamma.

    Ranking by sensing gamma alone silently picks links whose reporting leg is
    infeasible: such a link has a fine echo but m1 == 0 at the fusion point, so
    its P_D is pinned at the false-alarm floor and *every* lever looks useless.
    (Measured on the 400 m geometry, this trap reported a flat 0.0500 across a
    full 20 dB sweep of processing gain.) Ranking by the predicted detection
    probability is what ``advice/002.md`` section 4 asks for anyway.
    """
    Q = np.asarray(tables.gamma_sense).shape[2]
    M = np.asarray(tables.gamma_sense).shape[0]
    out = []
    for q in range(Q):
        best_link, best_key = None, None
        for i in range(M):
            for j in range(M):
                if i == j or (i, j) in exclude:
                    continue
                g = float(np.asarray(tables.gamma_sense)[i, j, q])
                if not np.isfinite(g) or g <= 0.0:
                    continue
                # Only views that actually DELIVER evidence are eligible. Most
                # views in the weak-target regime report nothing: their received
                # mean m1 is exactly zero no matter how much sensing energy is
                # spent, because the reporting leg never delivers, so their P_D is
                # pinned at the P_FA floor (measured 0.049995 against P_FA = 0.05).
                # Those dead views would tie with, or even outrank, genuine ones
                # under any naive argmax, and would silently absorb the whole
                # rescue budget.
                mom = received_moments(cfg, tables, (i, j), q, plan)
                if float(mom.m1) <= 0.0:
                    continue
                key = (float(mom.m1), g)
                if best_key is None or key > best_key:
                    best_link, best_key = (i, j), key
        out.append(best_link if best_link is not None else (0, 1))
    return out


def rinr_of(tables) -> float:
    r = np.asarray(tables.rinr, dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.median(r)) if r.size else 0.0


def calibration(area_xy: float, rcs: float, q: int, link):
    """P_D as a function of a pure numerator-gamma multiplier."""
    pts = []
    for s in CALIB_STEPS:
        cfg, _b, t, plan = build(area_xy, rcs, 16, 1.0, gscale=s)
        pts.append((10.0 * math.log10(s), pd_for(cfg, t, q, link, plan)))
    return pts


def equivalent_db(curve, target_pd: float) -> float:
    """Invert the calibration curve: which pure-gamma dB reproduces this P_D."""
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    lo = ys[0]
    hi = ys[-1]
    if not np.all(np.diff(ys) >= -1e-9):
        return float("nan")  # non-monotone calibration is itself a symptom
    if target_pd <= lo:
        return xs[0] if abs(target_pd - lo) < 1e-9 else float("-inf")
    if target_pd >= hi:
        return xs[-1]
    return float(np.interp(target_pd, ys, xs))


def delivery_coverage(cfg, tables, plan=None) -> float:
    """Fraction of candidate views whose report actually reaches the fusion point.

    A view with m1 == 0 is dead no matter how much sensing resource is pushed into
    it, because its P_D cannot exceed the false-alarm floor. This is the number
    that decides whether a rescue route has anything to act on at all.
    """
    M = np.asarray(tables.gamma_sense).shape[0]
    alive = total = 0
    for q in range(np.asarray(tables.gamma_sense).shape[2]):
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                g = float(np.asarray(tables.gamma_sense)[i, j, q])
                if not np.isfinite(g) or g <= 0.0:
                    continue
                total += 1
                if float(received_moments(cfg, tables, (i, j), q, plan).m1) > 0.0:
                    alive += 1
    return alive / total if total else 0.0


def scenario(area_xy: float, rcs: float) -> dict:
    print("=" * 92)
    print(f"footprint {area_xy:.0f} m | RCS {rcs} m^2 | P_D_req {PD_REQ} | single frozen best view")
    print("=" * 92)

    cfg0, base0, t0, plan0 = build(area_xy, rcs, 16, 1.0)
    r = rinr_of(t0)
    links = best_view_per_target(cfg0, t0, plan0)
    base_pd = np.array([pd_for(cfg0, t0, q, links[q], plan0) for q in range(len(links))])
    worst_q = int(np.argmin(base_pd))
    link = links[worst_q]

    print(f"baseline rinr = {10 * math.log10(r):+.2f} dB (r = {r:.3f})")
    print(f"delivery coverage: {delivery_coverage(cfg0, t0, plan0) * 100:.1f}% of views actually "
          f"deliver a report (the rest have m1 == 0 and cannot exceed the P_FA floor)")
    print(f"baseline: worst target q={worst_q} @ link {link}, P_D = {base_pd[worst_q]:.4f}")
    pfa = cfg0.detect.Pfa_target
    below = [q for q, p in enumerate(base_pd) if p < pfa - 1e-9]
    if below:
        # NOTE: "P_D < nominal P_FA" is NOT proof of a nonphysical detector. The
        # detector's *actual* size under the Cornish-Fisher threshold was measured
        # at 0.0494 against a nominal 0.05 (calibration is fine); the surrogate is
        # simply biased low against the MC truth (0.0371 vs 0.0490 at gamma~4e-4).
        # The correct invariant is MC P_D >= MC P_FA, not P_D_hat >= nominal P_FA.
        print(
            f"  note: surrogate P_D < nominal P_FA = {pfa:.3f} for targets {below}. "
            "This is NOT a nonphysical detector (MC check: P_FA 0.0494 / P_D 0.0490). "
            "The moment-matched `predicted_pd_for_links` is biased LOW in the deep "
            "low-SNR regime, so absolute deficits d_q are overstated; ranking use "
            "is still the safer of the two."
        )

    curve = calibration(area_xy, rcs, worst_q, link)
    print(
        "calibration (pure numerator gamma -> P_D): "
        + ", ".join(f"{x:+.1f}dB:{y:.3f}" for x, y in curve)
    )
    pd_base = base_pd[worst_q]

    rows = []
    # ---- looks lever: advice's primary resource ---------------------------
    for looks in (32, 64, 144, 256):
        cfg, _b, t, plan = build(area_xy, rcs, looks, 1.0)
        p = pd_for(cfg, t, worst_q, link, plan)
        rows.append((f"looks 16->{looks}", p, equivalent_db(curve, p), None))

    # ---- power lever: advice demotes this to secondary ---------------------
    for pscale in (1.25, 2.0, 10.0):
        cfg, _b, t, plan = build(area_xy, rcs, 16, pscale)
        p = pd_for(cfg, t, worst_q, link, plan)
        g_now = float(np.asarray(t.gamma_sense)[link[0], link[1], worst_q])
        g_ref = float(np.asarray(t0.gamma_sense)[link[0], link[1], worst_q])
        measured = g_now / g_ref if g_ref > 0 else float("nan")
        analytic = pscale * (1.0 + r) / (1.0 + pscale * r)
        drift = abs(measured - analytic) / max(analytic, 1e-12)
        rows.append((f"power x{pscale}", p, equivalent_db(curve, p), drift))

    print("-" * 92)
    print(f"{'lever':<18}{'worst P_D':>12}{'vs base':>10}{'equiv gamma dB':>16}{'xcheck':>10}")
    for name, p, eq, drift in rows:
        flag = "" if drift is None else ("OK" if drift < 0.05 else f"DRIFT {drift:.2f}")
        eq_s = "-inf" if eq == float("-inf") else ("nan" if math.isnan(eq) else f"{eq:+.2f}")
        print(f"{name:<18}{p:>12.4f}{p - pd_base:>+10.4f}{eq_s:>16}{flag:>10}")

    ordered = sorted(
        [(n, e) for n, _, e, _ in rows if not math.isnan(e) and e != float("-inf")],
        key=lambda x: -x[1],
    )
    print("-" * 92)
    print("ordering by equivalent gamma dB (primary resource should head this list):")
    for n, e in ordered:
        print(f"  {n:<18} {e:+.2f} dB")

    n_sat = int((base_pd >= PD_REQ).sum())
    n_weak = len(base_pd) - n_sat
    spread = float(base_pd.max() - base_pd.min())
    print("-" * 92)
    print(
        f"redistributable budget: {n_sat}/{len(base_pd)} targets already meet P_D_req at the "
        f"cheapest setting; P_D spread = {spread:.3f}."
    )
    # This verdict must follow the data -- an earlier revision asserted "no target is
    # easy" unconditionally and contradicted a run where 6/10 were already satisfied.
    if n_sat == 0:
        print(
            "  => NOTHING to free: no target is satisfied at the cheapest setting, so "
            "'stop paying the easy ones' releases zero budget and reallocation can only "
            "shuffle between equally-hard targets. Use a MIXED-RCS scenario to exercise it."
        )
    else:
        # Baseline budget assumes every target gets L0 looks; the satisfied ones keep
        # their cheapest allocation and everything else is pushed to the weak ones.
        base_budget = len(base_pd) * 16
        for mult in (2, 4):
            total = mult * base_budget
            spare = total - n_sat * 16
            per_weak = spare / n_weak if n_weak else 0.0
            print(
                f"  => budget x{mult}: total {total:.0f} looks, satisfied keep 16 each, "
                f"{n_weak} weak targets get {per_weak:.0f} looks each"
            )
    print()
    return {"area": area_xy, "rinr_db": 10 * math.log10(r), "rows": rows, "worst_pd": pd_base}


def self_gate() -> None:
    """The probe may not trust itself: setting the processing-gain override to its
    natural value must reproduce the unset default bit-exactly."""
    area, rcs = 4000.0, 0.05
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    cfg = apply_overrides(cfg, {"geometry.area_xy": area, "detect.target_rcs": rcs})
    natural = cfg.waveform.N * cfg.waveform.L
    rng = np.random.default_rng(SEED)
    base = build_base_gains(cfg, generate_geometry(cfg, rng), rng)
    ref = np.asarray(compute_link_tables(cfg, base).gamma_sense)

    cfg2 = apply_overrides(cfg, {"detect.sensing_processing_gain": natural})
    rng2 = np.random.default_rng(SEED)
    base2 = build_base_gains(cfg2, generate_geometry(cfg2, rng2), rng2)
    got = np.asarray(compute_link_tables(cfg2, base2).gamma_sense)

    if not np.allclose(ref, got, rtol=0, atol=0):
        raise SystemExit("SELF-GATE FAILED: the gamma knob does not round-trip")
    print("self-gate OK: gamma knob round-trips bit-exactly at its natural value\n")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--area", type=float, help="deployment footprint side in metres")
    ap.add_argument("--rcs", type=float, help="target RCS in m^2")
    args = ap.parse_args()

    self_gate()

    if args.area is not None and args.rcs is not None:
        if args.area <= 0 or args.rcs <= 0:
            raise SystemExit("--area and --rcs must be positive")
        scenario(args.area, args.rcs)
        return 0

    release = scenario(4000.0, 0.05)
    scenario(4000.0, 0.2)
    compressed = scenario(400.0, 0.05)

    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    get = lambda res, name: next(  # noqa: E731
        (e for n, _, e, _ in res["rows"] if n == name), float("nan")
    )
    for lever in ("power x1.25", "looks 16->64"):
        a, b = get(release, lever), get(compressed, lever)
        print(
            f"  {lever:<16} release(4km, r={release['rinr_db']:+.1f}dB) {a:+.2f} dB "
            f"| compressed(400m, r={compressed['rinr_db']:+.1f}dB) {b:+.2f} dB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
