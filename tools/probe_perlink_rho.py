"""Admission test for P1-3 (identifiability-aware selection): does ``rho`` have
*spread across links*, and is that spread **new information**?

The angular probe (``ANGULAR_IDENTIFIABILITY_PROBE.md``) answered "the DD overlap
is mostly an artefact of the DD-only abstraction; a small array recovers it".
That recovery is necessary but **not sufficient** for P1-3: a selector that
maximises ``rho`` only does work if

(A) ``rho`` actually differs from link to link for the *same* target -- if every
    link sits at 0.6, picking by ``rho`` is picking at random;
(B) that difference is not already visible to the **existing** selector, which
    ranks links by sensing SINR.  If ``rho`` is monotone in SINR, the current
    selector already picks the good ones and adding ``rho`` changes nothing.

Both are measured here, on the *unweighted* per-link values -- the recorded
scene-level figure (``rho_weighted`` 0.0080 -> 0.470/0.656/0.800) is a
power-weighted average over links and cannot answer either question.

Pre-registered constants (fixed before any number was seen; see
``SELF_CONSISTENCY`` for the check that they do not contradict each other, which
is the defect the previous probe shipped with)::

    M_LIST            = (4, 8, 16)   receive array sizes; 1 = the DD-only baseline
    PRIMARY_AXIS      = 0            body axis x, fixed -- not chosen after the fact
    SPREAD_PASS       = 0.20         p90 - p10 of per-link rho, per target
    SPREAD_FAIL       = 0.05
    RHO_S_MAX         = 0.70         |Spearman(rho, gamma_sense)| above which the
                                     existing selector already sees it
    K_SELECT          = (1, 2, 3)    selection budget for the headroom readout

Decision: P1-3 is worth building iff **(A) passes for at least one M in
{4, 8, 16}** and **(B) passes**.  Everything else is a readout.

Caveat registered in advance: ``rho`` saturates at 1, so at large ``M`` the
spread can shrink from the *ceiling* rather than from uniformity.  That is why
(A) is evaluated per ``M`` and the maximum is taken, and why the M=1 column is
reported as the reference and not as a candidate.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/probe_perlink_rho.py --trials 12 --out results_perlink_rho
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import probe_angular_identifiability as pa
from isac_sim.receiver import cancellation as cx
from isac_sim.core.config import Config, apply_preset
from isac_sim.sensing.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.scenario.prior import perturbed_geometry

# --------------------------------------------------------------------------
# Pre-registered constants
# --------------------------------------------------------------------------
M_LIST = (1, 4, 8, 16)
CANDIDATE_M = (4, 8, 16)
AXES = (0, 1)
PRIMARY_AXIS = 0
SPREAD_PASS = 0.20
SPREAD_FAIL = 0.05
RHO_S_MAX = 0.70
K_SELECT = (1, 2, 3)

SELF_CONSISTENCY = (
    "PASS (spread >= %.2f) and FAIL (spread < %.2f) are separated by a grey band, "
    "not the same condition; (B) is complementary by construction (|rho_s| <= %.2f "
    "vs > %.2f). The previous probe shipped a rule whose PASS trigger equalled its "
    "FAIL threshold, so it could only ever FAIL." % (
        SPREAD_PASS, SPREAD_FAIL, RHO_S_MAX, RHO_S_MAX)
)


def link_templates(cfg: Config, geom_belief, base, receiver: int,
                   us: Sequence[float]) -> List[Dict]:
    """Every believed echo at this receiver, keeping the illuminator index.

    Mirrors ``probe_angular_identifiability.scene_templates`` but carries ``i``,
    because "per-link" here means per ``(illuminator i, receiver j)``.  The
    ``(doppler, delay)`` unpacking order is the one that probe got wrong on its
    first run; it is kept identical here on purpose.
    """
    out: List[Dict] = []
    g_proc = float(cfg.waveform.N * cfg.waveform.L)
    sense = float(cfg.radio.rho * cfg.radio.P_default)
    for i in range(cfg.scale.M):
        if i == receiver:
            continue
        for q in range(cfg.scale.Q):
            gain = float(base.target_gain[i, receiver, q])
            if gain <= 0.0:
                continue
            doppler, delay = cx.target_link_offset(cfg, geom_belief, i, receiver, q)
            out.append({
                "kernel": cx.kernel_vector(cfg, doppler, delay),
                "u": float(us[q]),
                "i": int(i),
                "q": int(q),
            })
    return out


SAMPLE_VERIFY = 8         # kept only to document the failure below; unused now


def per_link_rho(cfg: Config, geom_belief, base, receiver: int, m_rx: int,
                 axis: int) -> List[Dict[str, float]]:
    """``rho`` of every ``(i, q)`` echo at this receiver against all *other*
    targets' echoes there.

    Scale-invariant, so the amplitude outer product that the scene-level figure
    uses is deliberately omitted: the question here is "what would the selector
    see on this link", and weighting by power would fold the SINR back into the
    quantity that is supposed to be tested against the SINR in (B).
    """
    us = pa.axis_us(geom_belief, receiver, cfg.scale.Q, axis)
    items = link_templates(cfg, geom_belief, base, receiver, us)
    if not items:
        return []
    kernels = [it["kernel"] for it in items]
    us_list = [it["u"] for it in items]
    owners = [it["q"] for it in items]
    gram = pa.dd_gram(kernels) * pa.array_gram(m_rx, us_list)
    # One pseudo-inverse per column, and that is deliberate: a 1000x faster
    # Schur-complement route (``rho_c = 1 / inv(corr)[c,c]``, one inverse for
    # every column) was implemented here, guarded by re-deriving a sample of 8 of
    # the 140 columns the slow way, and **shipped wrong numbers anyway**.
    # Measured on this scene at M=8: per column 0.988 vs 0.346, 0.501 vs 0.050,
    # 0.304 vs 0.113 -- median 0.909 (slow) against 0.374 (fast), because forming
    # ``inv`` of the 140-template correlation matrix is not backward stable
    # enough to read off a reciprocal.  The guard declared 68% of configurations
    # "agreed" (max deviation 4.5e-8) because it only looked at every 17th
    # column; the columns it looked at happened to be the ones that agree.
    # A sampled certificate is not a certificate when the error is
    # column-dependent, so the shortcut is gone rather than gated.
    rows: List[Dict[str, float]] = []
    for idx, owner in enumerate(owners):
        others = [j for j, o in enumerate(owners) if o != owner]
        rows.append({
            "receiver": int(receiver),
            "illuminator": int(items[idx]["i"]),
            "target": int(owner),
            "rho": pa.escape_from_gram(gram, idx, others),
        })
    return rows


def _ranks(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    r = np.empty(len(values), dtype=float)
    r[order] = np.arange(len(values), dtype=float)
    return r


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.vdot(ra, ra) * np.vdot(rb, rb)))
    return float(np.vdot(ra, rb) / denom) if denom > 0.0 else float("nan")


def _headroom(values: Sequence[float], k: int) -> Tuple[float, float]:
    """(mean of the k best, mean of the k around the median rank)."""
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    k = min(k, len(v))
    top = float(v[:k].mean())
    mid = len(v) // 2
    lo = max(mid - k // 2, 0)
    return top, float(v[lo:lo + k].mean())


def target_rows(trial: int, m_rx: int, axis: int, links: List[Dict[str, float]],
                gamma: np.ndarray) -> List[Dict[str, float]]:
    """Per-target aggregation of one (trial, M, axis) configuration."""
    out: List[Dict[str, float]] = []
    by_q: Dict[int, List[Dict[str, float]]] = {}
    for row in links:
        by_q.setdefault(int(row["target"]), []).append(row)
    for q, rows in sorted(by_q.items()):
        rho = [r["rho"] for r in rows]
        sinr = [float(gamma[r["illuminator"], r["receiver"], q]) for r in rows]
        by_rx: Dict[int, List[float]] = {}
        for r in rows:
            by_rx.setdefault(int(r["receiver"]), []).append(r["rho"])
        # Receiver-level spread: the selector decides *which receiver to fuse*,
        # so the spread that matters is between receivers (each summarised by the
        # median over its illuminators).  This must be computed on links pooled
        # over receivers -- doing it inside one receiver's loop gives a
        # single-element group and silently yields NaN, which is how the first
        # run reported it.
        rx_level = [float(np.median(v)) for v in by_rx.values()]
        rec: Dict[str, float] = {
            "trial": trial, "m_rx": m_rx, "axis": axis, "target": q,
            "n_links": len(rows),
            "n_receivers": len(rx_level),
            "rho_mean": float(np.mean(rho)),
            "rho_p10": float(np.percentile(rho, 10)),
            "rho_p50": float(np.percentile(rho, 50)),
            "rho_p90": float(np.percentile(rho, 90)),
            "spread_p90_p10": float(np.percentile(rho, 90) - np.percentile(rho, 10)),
            "rho_min": float(min(rho)),
            "rho_max": float(max(rho)),
            "spread_receiver_level": (float(np.percentile(rx_level, 90))
                                      - float(np.percentile(rx_level, 10)))
            if len(rx_level) >= 3 else float("nan"),
            "spearman_rho_sinr": _spearman(rho, sinr),
        }
        for k in K_SELECT:
            top, mid = _headroom(rho, k)
            rec["top%d" % k] = top
            rec["medrank%d" % k] = mid
            rec["headroom%d" % k] = top - mid
        # Versus *random* choice, whose expectation is the mean over links: this
        # is the number that says what a rho-aware selector gains over one that
        # ignores rho entirely.
        rec["headroom_vs_mean1"] = rec["top1"] - rec["rho_mean"]
        out.append(rec)
    return out


def _write(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _med(rows: List[Dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) == r.get(key)]
    vals = [v for v in vals if np.isfinite(v)]
    return st.median(vals) if vals else float("nan")


def report(rows: List[Dict[str, float]]) -> None:
    print("  %-5s %-5s %10s %10s %10s %10s %10s %10s" % (
        "M", "axis", "rho p10", "rho p50", "rho p90", "spread", "rx spread",
        "rho_s(SINR)"))
    for m in M_LIST:
        for ax in AXES:
            sub = [r for r in rows if r["m_rx"] == m and r["axis"] == ax]
            if not sub:
                continue
            print("  %-5d %-5d %10.4f %10.4f %10.4f %10.4f %10.4f %10.3f" % (
                m, ax, _med(sub, "rho_p10"), _med(sub, "rho_p50"),
                _med(sub, "rho_p90"), _med(sub, "spread_p90_p10"),
                _med(sub, "spread_receiver_level"), _med(sub, "spearman_rho_sinr")))

    print()
    print("  selection headroom (median over targets/trials): top-k minus")
    print("  median-ranked-k; if this is ~0 the selector has nothing to choose.")
    print("  %-5s %-5s %10s %10s %10s %14s" % (
        "M", "axis", "k=1", "k=2", "k=3", "top1 - mean"))
    for m in M_LIST:
        for ax in AXES:
            sub = [r for r in rows if r["m_rx"] == m and r["axis"] == ax]
            if not sub:
                continue
            print("  %-5d %-5d %10.4f %10.4f %10.4f %14.4f" % (
                m, ax, _med(sub, "headroom1"), _med(sub, "headroom2"),
                _med(sub, "headroom3"), _med(sub, "headroom_vs_mean1")))

    print()
    best = max((_med([r for r in rows if r["m_rx"] == m and r["axis"] == PRIMARY_AXIS],
                     "spread_p90_p10") for m in CANDIDATE_M), default=float("nan"))
    spread_ok = best >= SPREAD_PASS
    spread_bad = best < SPREAD_FAIL
    rho_s = _med([r for r in rows if r["m_rx"] == max(CANDIDATE_M)
                  and r["axis"] == PRIMARY_AXIS], "spearman_rho_sinr")
    info_ok = abs(rho_s) <= RHO_S_MAX
    print("  (A) spread across links : best M spread = %.4f  -> %s" % (
        best, "PASS" if spread_ok else ("FAIL" if spread_bad else "GREY")))
    print("  (B) new information     : |Spearman(rho, SINR)| = %.3f (<= %.2f) -> %s" % (
        abs(rho_s), RHO_S_MAX, "PASS" if info_ok else "FAIL"))
    print()
    print("  verdict: P1-3 %s" % (
        "WORTH BUILDING" if (spread_ok and info_ok) else "NOT JUSTIFIED BY THIS MEASUREMENT"))
    print("  (" + SELF_CONSISTENCY + ")")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="paper-canonical",
                    help="must match the angular probe: the per-link numbers are "
                         "only comparable to its scene-level ones on the same scene")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-links", default="8",
                    help="comma list of M for which per-link rows are dumped")
    ap.add_argument("--out", default="results/perlink_rho")
    args = ap.parse_args(argv)

    # Same construction as the angular probe -- ``paper-canonical`` plus the two
    # scenario overrides.  Using a bare ``Config()`` here (which is what the first
    # run did) silently measures a different scene: the per-link median came out
    # at 0.30 where the angular probe's scene median is 0.94, and nothing about
    # that gap is a finding -- it is a different waveform and power budget.
    cfg = apply_preset(Config(), args.preset)
    cfg.geometry.area_xy = args.area
    cfg.detect.target_rcs = args.rcs

    print("per-link rho admission test (P1-3): %s, area %.0f m, RCS %.2f, %d trials"
          % (args.preset, args.area, args.rcs, args.trials))
    print("  %d UAVs, %d targets; array = half-wavelength ULA, unit-modulus "
          "steering" % (cfg.scale.M, cfg.scale.Q))
    print("  nuisance = every other target's believed echo at that receiver")
    print()

    dump_m = {int(x) for x in args.dump_links.split(",") if x.strip()}
    rows: List[Dict[str, float]] = []
    per_link: List[Dict[str, float]] = []

    for t in range(int(args.trials)):
        rng = np.random.default_rng([cfg.run.seed + args.seed, 10 ** 6 + t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
        tables = compute_link_tables(cfg, base)
        # Pool every receiver's links before aggregating: the receiver-level
        # spread is a between-receiver quantity and cannot be seen from one.
        pooled: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
        for j in range(cfg.scale.M):
            for m in M_LIST:
                for ax in AXES:
                    pooled.setdefault((m, ax), []).extend(
                        per_link_rho(cfg, belief, base, j, m, ax)
                    )
        for (m, ax), links in sorted(pooled.items()):
            if not links:
                continue
            rows.extend(target_rows(t, m, ax, links, tables.gamma_sense))
            if m in dump_m and ax == PRIMARY_AXIS:
                for r in links:
                    r = dict(r)
                    r.update({"trial": t, "m_rx": m, "axis": ax})
                    per_link.append(r)
        print("  trial %d/%d done" % (t + 1, args.trials), flush=True)

    _write(os.path.join(args.out, "perlink_rho_targets.csv"), rows)
    _write(os.path.join(args.out, "perlink_rho_links.csv"), per_link)
    print()
    report(rows)
    print()
    print("wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
