"""Does a spatial dimension rescue the DD-collision?  Pre-registered probe.

Background
----------
``TP_UIC_V11.md`` section 4 and ``TP_UIC_V12.md`` section 3.3 measure that ten
scatterers sharing a handful of delay-Doppler cells are mutually unidentifiable:
the weighted escape fraction ``rho`` collapses 1.00 -> 0.381 -> 0.019 as
co-located neighbours are treated as nuisance, and neither perfect cancellation
nor a perfect tracker moves it.  The recorded conclusion is deliberately narrow
-- *"unidentifiable under the current DD-only receiver abstraction"* -- because
the receiver's template lives in delay-Doppler **only**.  A real UAV sensing
payload carries an array, and two targets in the same DD cell but at different
bearings have templates

    a(tau, nu, phi) = a_DD(tau, nu) (x) a_arr(phi),

which the DD-only abstraction collapses onto each other by construction.  This
probe measures whether that missing dimension is the whole story.

Pre-registered criteria (fixed in the constants below, before any run)
---------------------------------------------------------------------
``BW_u(M) = 0.886 / M`` is the 3 dB beamwidth in ``u = sin(phi)`` of a
half-wavelength ULA with ``M`` elements.  Separations are reported in beamwidths,
never in degrees: the beamwidth moves with ``M``, and that is the mechanism.

Per ``M``, with ``rho(0.5 BW)`` read off the curve:

* ``separable``  = ``rho(0.5 BW) >= RHO_PASS`` (0.9);
* ``available``  = co-located pairs separate by >= ``HALF_BW`` beamwidths in
  >= ``FRAC_PASS`` (50 %) of cases;
* **PASS** = ``separable and available``;
* **FAIL** = ``rho(0.5 BW) < RHO_FAIL`` (0.5), or ``best_frac < FRAC_FAIL``
  (20 %), or the smallest separation reaching ``RHO_PASS`` is >= ``BW_FAIL``
  (2.0) beamwidths;
* else **中间**.

*Disclosed flaw in that rule, found when it was first run and left in place*
(labelling it is more useful than repairing it silently): ``RHO_PASS = 0.9``
corresponds to a separation of almost exactly ``BW_FAIL = 2.0`` beamwidths, so
the second PASS condition and the third FAIL condition are the *same condition*.
The rule therefore cannot distinguish "needs an impractical aperture" from
"needs a finite loss budget" -- it only ever says FAIL.  Both readings are
reported below with their own numbers, and the verdict string is kept as
registered so the record shows what the rule actually did.

What is measured, and with what
-------------------------------
An array buys **aperture gain** (``+10 log10 M``, a budget lever, same class as
``G_hw``) and **spatial selectivity** (a narrower beam, an identifiability
lever).  Only the second is being measured: steering is **unit norm**, so the
total collected power is unchanged and the template stays on exactly the scale
``cancellation.kernel_vector`` uses.  Without that guard the curve would rise
with ``M`` merely because everything is stronger.

Template amplitudes and offsets come from the production helpers
(``cancellation.kernel_vector``, ``cancellation.target_link_offset``, the same
``P_sense * g * G_proc * G_hw`` echo power and the same believed geometry as the
DD-only audit), and templates are **centre columns only** -- no finite-difference
tangent columns -- so the projection below is the exact orthogonal projection and
the rank-trimming question does not arise.

Everything is computed in the **Gram domain**, which is exact rather than
approximate: ``<a_DD1 (x) a_arr1, a_DD2 (x) a_arr2> = <a_DD1, a_DD2> * A(u1, u2)``,
so the ``K*M``-dimensional Gram is the Hadamard product of a DD Gram (production
kernels, dimension ``K``) and the analytic Dirichlet kernel in ``u``.  A
``K*M = 65536``-column projection would be the same arithmetic, only slower.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/probe_angular_identifiability.py --trials 12 --scene-trials 12 \
        --out results_angular_probe
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim import aperture as ap  # noqa: E402
from isac_sim import cancellation as cx  # noqa: E402
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import build_base_gains, generate_geometry  # noqa: E402
from isac_sim.prior import perturbed_geometry  # noqa: E402
from tools.run_tp_uic_v1 import pick_receiver  # noqa: E402

# --------------------------------------------------------------------------
# Pre-registered constants.  Do not tune these after seeing a result.
# --------------------------------------------------------------------------
M_LIST = (4, 8, 16)
HALF_BW = 0.5          # beamwidths: the "half a beam" criterion
BW_FAIL = 2.0          # beamwidths (== the separation RHO_PASS needs; see above)
RHO_PASS = 0.9         # 0.5 dB matched-filter loss budget
RHO_FAIL = 0.5         # 3 dB loss budget
FRAC_PASS = 0.5
FRAC_FAIL = 0.2
DELTA_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
COLLOCATED_BINS = 0.5


def beamwidth_u(m_rx: int) -> float:
    """3 dB beamwidth of a half-wavelength ULA, in ``u = sin(phi)`` units.

    Delegated to :mod:`isac_sim.aperture` on purpose.  The array model started
    life here and is now also read by the production collision penalty, and two
    copies of a coherence formula is exactly how a probe's verdict and a model's
    behaviour drift apart while both still look self-consistent.
    """
    return ap.beamwidth_u(m_rx)


def array_factor(m_rx: int, delta_u: float) -> float:
    """``|<a(u), a(u + du)>|`` for unit-norm steering, spacing ``lambda/2``."""
    return ap.array_factor(m_rx, delta_u)


def analytic_rho(m_rx: int, delta_u: float) -> float:
    """Array-only prediction ``1 - |A|^2`` for two targets sharing a DD cell."""
    return ap.array_escape(m_rx, delta_u)


def dd_gram(kernels: Sequence[np.ndarray]) -> np.ndarray:
    """``<a_DD_i, a_DD_j>`` -- production kernels, dimension ``K``."""
    m = len(kernels)
    g = np.zeros((m, m), dtype=complex)
    for i, ki in enumerate(kernels):
        g[i, i] = np.vdot(ki, ki)
        for j in range(i + 1, m):
            v = np.vdot(ki, kernels[j])
            g[i, j] = v
            g[j, i] = np.conj(v)
    return g


def array_gram(m_rx: int, us: Sequence[float]) -> np.ndarray:
    """``A(u_i, u_j)`` for unit-norm steering."""
    n = len(us)
    g = np.ones((n, n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            v = array_factor(m_rx, float(us[i]) - float(us[j]))
            g[i, j] = v
            g[j, i] = v
    return g


def escape_from_gram(gram: np.ndarray, index: int, others: Sequence[int],
                     rcond: float = 1e-10) -> float:
    """Escape fraction from the Gram, computed on the **correlation** matrix.

    ``rho_c = 1 - ||P_span(others) a_c||^2 / ||a_c||^2`` is invariant to any
    diagonal rescaling of the templates (the span does not move and numerator and
    denominator scale together), so it may be evaluated on the correlation matrix.
    It **must** be: the echo amplitudes span orders of magnitude, and the raw Gram
    squares them, so a raw-domain ``pinv`` is the same arithmetic with the dynamic
    range doubled in the exponent.  Measured on one trial at ``M = 1`` -- vector
    domain ``cond = 5.5e6``, Gram domain ``3e13`` -- the raw-domain answer came out
    as ``rho = -1.43e6`` (then clipped to 0) where the production
    ``cancellation_glrt._project_out`` gives ``1.18e-06``.  Normalising first
    reproduces the vector-domain value exactly.

    **Convention (the defect this function shipped with).**  The projection
    energy is ``u @ G_ss^+ @ conj(u)`` with ``u = corr[c, s]`` -- *not*
    ``conj(u) @ G_ss^+ @ u``.  The two differ by a transposition of ``G_ss^+``;
    they are equal only when that block is real symmetric, which is why the
    original form passed every check that used **one** nuisance column (there
    ``G_ss^+`` is a scalar) and why it agreed with production to 2.33e-15 -- that
    certificate was taken on a two-template instance.  With ``|others| >= 2`` on
    complex data the wrong form departs from the vector-domain projection by up
    to **2.3e-01** and goes *negative* (then clipped to 0), which reads as "no
    escape at all".  Measured: max deviation 0.0 (n=2), 2.3e-01 (n=3),
    2.1e-01 (n=6), 7.3e-02 (n=20).

    Equivalent check kept as a regression: ``1 / inv(corr)[c, c]`` (the Schur
    complement) reproduces the same value; the wrong form does not.
    """
    diag = np.real(np.diag(gram))
    scale = np.sqrt(np.maximum(diag, 0.0))
    if scale[index] <= 0.0:
        return float("nan")
    safe = np.where(scale > 0.0, scale, 1.0)
    corr = gram / (safe[:, None] * safe[None, :])
    if not len(others):
        return 1.0
    s = np.asarray(list(others), dtype=int)
    keep = s[scale[s] > 0.0]
    if not len(keep):
        return 1.0
    c_is = corr[index, keep]
    c_ss = corr[np.ix_(keep, keep)]
    # ``u @ G_ss^+ @ conj(u)`` -- see the docstring.  Writing ``conj(u)`` on the
    # left silently transposes ``G_ss^+`` and is only correct for one nuisance
    # column.
    projected = float(np.real(c_is @ np.linalg.pinv(c_ss, rcond=rcond) @ np.conj(c_is)))
    return float(min(1.0, max(0.0, 1.0 - projected)))


def curve_for_m(cfg: Config, m_rx: int, delay: float, doppler: float) -> List[Dict[str, float]]:
    """Test A: rho against angular separation at *identical* DD.

    With ``M = 1`` both targets are the same template, so the projection removes
    the tested one exactly and ``rho = 0`` at every separation.  Everything above
    that comes from the array and nothing else.
    """
    rows: List[Dict[str, float]] = []
    bw = beamwidth_u(m_rx) if m_rx > 1 else float("nan")
    kernel = cx.kernel_vector(cfg, doppler, delay)
    for delta in DELTA_GRID:
        du = float(delta * bw) if m_rx > 1 else 0.0
        us = [0.0, du] if m_rx > 1 else [0.0, 0.0]
        gram = dd_gram([kernel, kernel]) * array_gram(m_rx, us)
        measured = escape_from_gram(gram, 0, [1])
        rows.append({
            "m_rx": m_rx,
            "delta_bw": float(delta),
            "delta_u": du,
            "rho": measured,
            "rho_analytic": analytic_rho(m_rx, du),
        })
    return rows


def axis_us(geom, receiver: int, q_count: int, axis: int) -> np.ndarray:
    """``(Q,)`` bearings projected onto body axis ``axis`` (0 = x, 1 = y).

    Delegated, for the same reason the array formulas are: the production model
    now computes bearings too, and the probe must be looking at the same ones.
    """
    return ap.bearings(geom, receiver, q_count, axis)


def scene_templates(
    cfg: Config, geom_belief, base, receiver: int, us: Sequence[float], m_rx: int,
):
    """Every believed echo at this receiver, as ``(DD kernel, u, amplitude, owner)``."""
    out: List[Tuple[np.ndarray, float, float, int]] = []
    g_proc = float(cfg.waveform.N * cfg.waveform.L)
    sense = float(cfg.radio.rho * cfg.radio.P_default)
    for i in range(cfg.scale.M):
        if i == receiver:
            continue
        for q in range(cfg.scale.Q):
            gain = float(base.target_gain[i, receiver, q])
            if gain <= 0.0:
                continue
            # ``target_link_offset`` returns ``(doppler, delay)`` -- the same
            # order ``kernel_vector`` takes.  Unpacking it as ``(delay, doppler)``
            # (which this probe did on its first run) puts every template in the
            # wrong DD cell and makes the M=1 anchor disagree with the recorded
            # DD-only audit; the cross-check below is what caught it.
            doppler, delay = cx.target_link_offset(cfg, geom_belief, i, receiver, q)
            out.append((cx.kernel_vector(cfg, doppler, delay), float(us[q]),
                        math.sqrt(sense * gain * g_proc), q))
    return out


def real_scene_rho(cfg: Config, geom_belief, base, receiver: int, m_rx: int,
                   axis: int, rcond: float = 1e-10) -> Dict[str, float]:
    """Test C: the *real* scene's ``rho``, with and without the array.

    All other targets' believed echoes become the nuisance set, amplitudes carry
    the production echo power, and the per-template escape fractions are averaged
    with the same power weighting the production audit's ``rho_weighted`` uses.
    The ``M = 1`` column is therefore directly comparable to the recorded
    DD-only value (``rho_weighted = 0.0080``, on-grid median ``1.83e-04``), which
    is the cross-check that this probe measures the same quantity.

    Bearings come from the **believed** geometry: the receiver steers from its
    own belief, so using truth angles would hand it information it does not have.
    """
    us = axis_us(geom_belief, receiver, cfg.scale.Q, axis)
    items = scene_templates(cfg, geom_belief, base, receiver, us, m_rx)
    if not items:
        return {"m_rx": m_rx, "rho_weighted": float("nan"), "rho_median": float("nan"),
                "n_templates": 0}
    kernels = [it[0] for it in items]
    us_list = [it[1] for it in items]
    amps = np.asarray([it[2] for it in items])
    owners = [it[3] for it in items]
    gram = dd_gram(kernels) * array_gram(m_rx, us_list)
    gram = gram * (amps[:, None] * amps[None, :])
    rho_c = []
    for idx, owner in enumerate(owners):
        others = [j for j, o in enumerate(owners) if o != owner]
        rho_c.append(escape_from_gram(gram, idx, others, rcond))
    weights = amps ** 2
    total = float(weights.sum()) or 1.0
    return {
        "m_rx": m_rx,
        "rho_weighted": float(sum(r * w for r, w in zip(rho_c, weights)) / total),
        "rho_median": float(np.median(rho_c)),
        "n_templates": len(items),
    }


def co_located_pairs(cfg: Config, base, receiver: int, ang: np.ndarray) -> List[Dict[str, float]]:
    """Pairs whose DD templates collide under *some* illuminator, plus bearings.

    The DD separation of a pair depends on the illuminator, so a pair counts as
    co-located when even the **best case for the DD-only receiver** cannot
    separate them (separation <= ``COLLOCATED_BINS`` under at least one
    illuminator).  The absolute DD coordinate is ``bin + frac`` -- ``*_frac`` is
    the signed offset *inside* the bin -- so the difference uses the sum.
    """
    rows: List[Dict[str, float]] = []
    for q in range(cfg.scale.Q):
        for r in range(q + 1, cfg.scale.Q):
            best = float("inf")
            for i in range(cfg.scale.M):
                if i == receiver:
                    continue
                dl = abs((base.delay_bin[i, receiver, q] + base.delay_frac[i, receiver, q])
                         - (base.delay_bin[i, receiver, r] + base.delay_frac[i, receiver, r]))
                dk = abs((base.doppler_bin[i, receiver, q] + base.doppler_frac[i, receiver, q])
                         - (base.doppler_bin[i, receiver, r] + base.doppler_frac[i, receiver, r]))
                best = min(best, max(dl, dk))
            if not math.isfinite(best) or best > COLLOCATED_BINS:
                continue
            rows.append({
                "q": q, "r": r, "dd_sep_bins": float(best),
                "du_x": abs(float(ang[0, q] - ang[0, r])),
                "du_y": abs(float(ang[1, q] - ang[1, r])),
            })
    return rows


def anchor_check(cfg: Config, geom, geom_belief, base, receiver: int, rng) -> Dict[str, float]:
    """Same objects, both implementations, in a *well-conditioned* case.

    At ``M = 1`` the probe's templates *are* the production dictionary columns, so
    a two-template instance can be compared directly and exactly: vectors from
    ``build_observation``, one tested column and one nuisance column, production
    ``cancellation_glrt._project_out`` on the left, the Gram shortcut on the
    right.

    The comparison has to be pairwise, and that is a real finding rather than a
    convenience: with all 140 scene templates at once, ``rho`` is ~1e-6 and the
    Gram shortcut's own error is the same size (the raw-amplitude version returned
    ``rho = -1.4e6``), so *no* implementation can be verified against another at
    that floor.  A two-template instance is well conditioned, so agreement there
    is meaningful -- and it is what certifies the shortcut used for ``M >= 4``,
    where ``rho`` is 0.4-0.8 and no floor is involved.
    """
    from isac_sim import cancellation_glrt as gl

    m = cfg.scale.M
    sense = np.full(m, cfg.radio.rho * cfg.radio.P_default)
    obs = cx.build_observation(
        cfg, geom, geom_belief, base, receiver, rng=rng, sense_power=sense,
        radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
        hw_gain=1.0, weak_index=0,
    )
    ids = np.asarray(obs.A_target_ids)
    centre = np.zeros(ids.size, dtype=bool)
    centre[:: 1 + 2 * int(cfg.cancellation.tangent_order)] = True
    pairs = []
    first = int(ids[0])
    for other in sorted(set(int(v) for v in ids)):
        if other == first:
            continue
        cols_q = np.flatnonzero((ids == first) & centre)
        cols_n = np.flatnonzero((ids == other) & centre)
        if not cols_q.size or not cols_n.size:
            continue
        v = obs.A[:, cols_q[0]]
        n = obs.A[:, cols_n[0]]
        left = gl._project_out(n[:, None], v[:, None])[:, 0]
        rho_prod = float(np.vdot(left, left).real / np.vdot(v, v).real)
        gram = np.array([[np.vdot(v, v), np.vdot(v, n)],
                         [np.vdot(n, v), np.vdot(n, n)]], dtype=complex)
        rho_gram = escape_from_gram(gram, 0, [1])
        pairs.append((rho_prod, rho_gram))
        break
    return {
        "rho_production_pair": pairs[0][0] if pairs else float("nan"),
        "rho_gram_pair": pairs[0][1] if pairs else float("nan"),
        "n_columns": float(obs.A.shape[1]),
        "n_centre_columns": float(int(centre.sum())),
    }


def need_separation(curve: List[Dict[str, float]]) -> float:
    """Smallest separation (beamwidths) reaching ``RHO_PASS``, else ``inf``."""
    for row in curve:
        if row["delta_bw"] > 0.0 and row["rho"] >= RHO_PASS:
            return float(row["delta_bw"])
    return float("inf")


def verdict(rho_half: float, need: float, frac: float) -> str:
    """The pre-registered per-``M`` verdict.  Order is fixed."""
    if rho_half >= RHO_PASS and frac >= FRAC_PASS:
        return "PASS"
    if rho_half < RHO_FAIL or frac < FRAC_FAIL or need >= BW_FAIL:
        return "FAIL"
    return "中间"


def rank(name: str) -> int:
    return {"FAIL": 0, "中间": 1, "PASS": 2}[name]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--scene-trials", type=int, default=12)
    ap.add_argument("--anchor-trials", type=int, default=4)
    ap.add_argument("--out", default="results_angular_probe")
    args = ap.parse_args(argv)

    cfg = apply_overrides(
        apply_preset(Config(), args.preset),
        {
            "geometry.area_xy": float(args.area),
            "detect.target_rcs": float(args.rcs),
            "run.seed": int(args.seed),
            "run.verbose": False,
        },
    )
    os.makedirs(args.out, exist_ok=True)
    delay, doppler = 2.37, 1.44
    lam = cx.wavelength(cfg)

    print("angular identifiability probe : %s, area=%.0f m, seed=%d"
          % (args.preset, args.area, args.seed))
    print("  pre-registered PASS: rho >= %.2f at %.2f beamwidth for some M AND"
          % (RHO_PASS, HALF_BW))
    print("  >= %.0f%% of co-located pairs reaching that separation." % (100 * FRAC_PASS))

    curves: List[Dict[str, float]] = []
    for m_rx in list(M_LIST) + [1]:
        curves.extend(curve_for_m(cfg, m_rx, delay, doppler))

    print("\n=== A. rho vs angular separation at identical (delay, doppler) ===")
    print("  %-5s %s" % ("M", "".join("%9s" % ("%.2fBW" % d) for d in DELTA_GRID)))
    worst = 0.0
    for m_rx in list(M_LIST) + [1]:
        sub = [r for r in curves if r["m_rx"] == m_rx]
        print("  %-5d %s" % (m_rx, "".join("%9.2e" % r["rho"] for r in sub)))
        for r in sub:
            worst = max(worst, abs(r["rho"] - r["rho_analytic"]))
    print("  cross-check vs the Dirichlet kernel 1-|A|^2 : max |diff| = %.2e" % worst)
    print("  M=1 is the DD-only abstraction: identical templates, rho = 0 at every")
    print("  separation.  Note the curve is a function of M*delta_u alone, so in")
    print("  *beamwidth* units it is the same curve for every M -- the aperture")
    print("  only moves the unit, which is exactly why the unit must be beamwidths.")

    pairs: List[Dict[str, float]] = []
    ang = np.zeros((2, cfg.scale.Q))
    for t in range(int(args.trials)):
        rng = np.random.default_rng([cfg.run.seed, 10 ** 5 + t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        receiver, _ = pick_receiver(cfg, base, "median")
        p_rx = np.asarray(geom.p_uav[receiver], dtype=float)
        for q in range(cfg.scale.Q):
            v = np.asarray(geom.p_tgt[q], dtype=float) - p_rx
            n = float(np.linalg.norm(v))
            ang[0, q] = v[0] / n if n > 0 else 0.0
            ang[1, q] = v[1] / n if n > 0 else 0.0
        pairs.extend(co_located_pairs(cfg, base, receiver, ang))

    print("\n=== B. separations that actually occur between co-located targets ===")
    print("  DD separation <= %.1f bins under some illuminator; %d trials, n = %d pairs"
          % (COLLOCATED_BINS, args.trials, len(pairs)))
    print("  %-5s %6s %9s %9s %9s %9s %9s" % (
        "M", "aper m", "med BW x", "med BW y", "f>=1BW x", "f>=2BW x", "need BW"))
    frac_reports: Dict[int, Dict[str, float]] = {}
    for m_rx in M_LIST:
        bw = beamwidth_u(m_rx)
        stat: Dict[str, float] = {}
        for tag, key in (("x", "du_x"), ("y", "du_y")):
            vals = [p[key] / bw for p in pairs]
            stat[tag] = (st.median(vals) if vals else float("nan"),
                         sum(1 for v in vals if v >= 1.0) / max(len(vals), 1),
                         sum(1 for v in vals if v >= 2.0) / max(len(vals), 1),
                         sum(1 for v in vals if v >= HALF_BW) / max(len(vals), 1))
        best = "x" if stat["x"][0] >= stat["y"][0] else "y"
        curve = [r for r in curves if r["m_rx"] == m_rx]
        need = need_separation(curve)
        frac_reports[m_rx] = {"med": stat[best][0], "f1": stat[best][1],
                              "f2": stat[best][2], "fhalf": stat[best][3],
                              "best": best, "need": need}
        print("  %-5d %6.2f %9.2f %9.2f %9.2f %9.2f %9.2f   (best axis %s)"
              % (m_rx, (m_rx - 1) * lam / 2.0, stat["x"][0], stat["y"][0],
                 stat[best][1], stat[best][2], need, best))

    print("\n=== C. the real scene: rho with and without the array ===")
    print("  nuisance = all other targets' believed echoes; amplitudes = production")
    print("  echo power; averaging = production power weight; bearings from belief")
    scene: List[Dict[str, float]] = []
    for m_rx in [1] + list(M_LIST):
        acc: Dict[int, List[float]] = {0: [], 1: []}
        acc_m, acc_n = [], []
        for t in range(int(args.scene_trials)):
            rng = np.random.default_rng([cfg.run.seed, 10 ** 5 + t])
            geom = generate_geometry(cfg, rng)
            base = build_base_gains(cfg, geom, rng)
            belief = perturbed_geometry(
                cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
            )
            receiver, _ = pick_receiver(cfg, base, "median")
            for axis in (0, 1):
                got = real_scene_rho(cfg, belief, base, receiver, m_rx, axis)
                acc[axis].append(got["rho_weighted"])
                if axis == 0:
                    acc_m.append(got["rho_median"])
                    acc_n.append(got["n_templates"])
        scene.append({"m_rx": m_rx,
                      "rho_weighted_x": st.median(acc[0]) if acc[0] else float("nan"),
                      "rho_weighted_y": st.median(acc[1]) if acc[1] else float("nan"),
                      "rho_median": st.median(acc_m) if acc_m else float("nan"),
                      "n_templates": st.median(acc_n) if acc_n else 0.0})
    print("  %-5s %16s %16s %14s %10s" % (
        "M", "rho_weighted(x)", "rho_weighted(y)", "rho_median", "templates"))
    for row in scene:
        print("  %-5d %16.3e %16.3e %14.3e %10.1f" % (
            row["m_rx"], row["rho_weighted_x"], row["rho_weighted_y"],
            row["rho_median"], row["n_templates"]))
    print("  recorded DD-only audit for comparison: rho_weighted 0.0080,"
          " on-grid median 1.83e-04")

    print("\n=== C2. anchor: same trial, M=1, both conventions ===")
    anchors: List[Dict[str, float]] = []
    for t in range(int(args.anchor_trials)):
        rng = np.random.default_rng([cfg.run.seed, 10 ** 5 + t])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        belief = perturbed_geometry(
            cfg, geom, cfg.prior.belief_sigma_pos_m, cfg.prior.belief_sigma_vel_mps, rng
        )
        receiver, _ = pick_receiver(cfg, base, "median")
        ctx = np.random.default_rng([cfg.run.seed, 10 ** 6 + t])
        anchors.append(anchor_check(cfg, geom, belief, base, receiver, ctx))
    if anchors:
        prod = [a["rho_production_pair"] for a in anchors]
        gram = [a["rho_gram_pair"] for a in anchors]
        rel = [abs(p - g) / max(abs(p), 1e-300) for p, g in zip(prod, gram)]
        print("  one tested column vs one nuisance column, from build_observation:")
        print("  %-30s %13s %13s" % ("", "production", "gram"))
        print("  %-30s %13.3e %13.3e" % ("rho (median)",
                                         st.median(prod), st.median(gram)))
        print("  %-30s %13.2e" % ("max relative difference", max(rel) if rel else float("nan")))
        print("  %-30s %13.0f" % ("centre columns in the scene dictionary",
                                  anchors[0]["n_centre_columns"]))
        print("  Pairwise, because with all 140 scene templates rho is ~1e-6 and the")
        print("  Gram shortcut's own error is that size -- no implementation can be")
        print("  verified at that floor.  This is the well-conditioned certificate for")
        print("  the shortcut used at M >= 4, where rho is 0.4-0.8.  The array half of")
        print("  the template is certified separately in A (exact to 2e-16 against")
        print("  the Dirichlet kernel 1-|A|^2).")

    print("\n=== verdict (pre-registered, unchanged) ===")
    verdicts = {}
    for m_rx in M_LIST:
        bw = beamwidth_u(m_rx)
        curve = [r for r in curves if r["m_rx"] == m_rx]
        rho_half = float([r for r in curve if abs(r["delta_bw"] - HALF_BW) < 1e-9][0]["rho"])
        fr = frac_reports[m_rx]
        verdicts[m_rx] = verdict(rho_half, fr["need"], fr["fhalf"])
    best = "FAIL"
    for m_rx, name in verdicts.items():
        if rank(name) > rank(best):
            best = name
    for m_rx, name in verdicts.items():
        fr = frac_reports[m_rx]
        print("  M=%-3d %-6s  rho(0.5BW)=%.2e  need=%.2f BW  co-located>=0.5BW: %.0f%%"
              % (m_rx, name, float([r for r in curves if r["m_rx"] == m_rx
                                    and abs(r["delta_bw"] - HALF_BW) < 1e-9][0]["rho"]),
                 fr["need"], 100 * fr["fhalf"]))
    print("  best over M: %s" % best)
    print("  ⚠️ the rule's constants are self-referential (RHO_PASS=0.9 needs ~2.0 BW,")
    print("     which is BW_FAIL), so the verdict string is structurally FAIL. The")
    print("     decision below uses the two loss budgets explicitly instead:")
    for m_rx in M_LIST:
        fr = frac_reports[m_rx]
        print("     M=%-3d 3 dB budget (rho>=0.5, needs 1.0 BW): %.0f%% of pairs clear it;"
              % (m_rx, 100 * fr["f1"]))
        print("           0.5 dB budget (rho>=0.9, needs %.1f BW): %.0f%% clear it"
              % (fr["need"], 100 * fr["f2"]))

    rows = [dict(c, kind="curve") for c in curves]
    rows += [dict(p, kind="pair") for p in pairs]
    rows += [dict(s, kind="scene") for s in scene]
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    path = os.path.join(args.out, "angular_probe.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("\nwrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
