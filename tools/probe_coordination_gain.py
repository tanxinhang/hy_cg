"""Can inter-UAV coordination reduce interference and buy detection performance?

Locked scenario: 500 m footprint, RCS 0.2 m^2 (see ADVICE_002_INTEGRATION.md).

Why this question is worth asking *here*: at this working point ``rinr = +11.5 dB``,
i.e. the sensing denominator is interference dominated. Interference is therefore the
right lever, and "cancel before you transmit" is the documented ordering.

Freedom-of-existence check (does the model even have this knob?): YES --
``active_tx_mask`` is consumed in ``simulate.py:584/873`` and gated by
``interference.sense_gate_by_active_tx`` (``config.py:304``, default **False**).
But the mask is built *after* selection, which is exactly audit finding F1: the
selector never sees it. This probe measures what pre-selection coordination would
be worth, i.e. what the inter-UAV link could buy.

Self-gate: interference must fall monotonically as fewer nodes radiate. If it does
not, the probe is measuring something else and refuses to report.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac_sim.config import (  # noqa: E402
    apply_overrides,
    apply_preset,
    default_config,
    validate_config,
)
from isac_sim.fusion import predicted_pd_for_links  # noqa: E402
from isac_sim.model import compute_link_tables, build_base_gains, generate_geometry  # noqa: E402
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402

SEED = 20260916
AREA = 500.0
RCS = 0.2
PD_REQ = 0.95


def build(looks: int = 16, mask: np.ndarray | None = None, gate: bool = False,
          kappa_db: float | None = None):
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    cfg = apply_overrides(
        cfg,
        {
            "geometry.area_xy": AREA,
            "detect.target_rcs": RCS,
            "detect.n_looks": looks,
            "interference.sense_gate_by_active_tx": gate,
        },
    )
    if kappa_db is not None:
        cfg = apply_overrides(cfg, {"interference.direct_cancellation_db": kappa_db})
    validate_config(cfg)
    rng = np.random.default_rng(SEED)
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base, active_tx_mask=mask)
    plan = assign_fusion_nodes(cfg, base, tables, geom=geom)
    return cfg, base, tables, plan


def pick_worst(cfg, tables, plan):
    """Worst target and its best *delivering* view (m1 > 0), as in the lever probe."""
    from isac_sim.soft_channel import received_moments

    g = np.asarray(tables.gamma_sense)
    Q, M = g.shape[2], g.shape[0]
    best_by_q = []
    for q in range(Q):
        best, key = None, None
        for i in range(M):
            for j in range(M):
                if i == j or not np.isfinite(g[i, j, q]) or g[i, j, q] <= 0:
                    continue
                if float(received_moments(cfg, tables, (i, j), q, plan).m1) <= 0:
                    continue
                k = (float(received_moments(cfg, tables, (i, j), q, plan).m1), g[i, j, q])
                if key is None or k > key:
                    best, key = (i, j), k
        best_by_q.append(best if best else (0, 1))
    pds = [predicted_pd_for_links(cfg, tables, q, [best_by_q[q]], plan=plan)
           for q in range(Q)]
    worst_q = int(np.argmin(pds))
    return worst_q, best_by_q[worst_q], float(pds[worst_q]), best_by_q, pds


def rinr_at(tables, link, q):
    return float(np.asarray(tables.rinr)[link[0], link[1]])


def gamma_at(tables, link, q):
    return float(np.asarray(tables.gamma_sense)[link[0], link[1], q])


def mask_of(M: int, members) -> np.ndarray:
    m = np.zeros(M, dtype=bool)
    m[list(members)] = True
    return m


def main() -> int:
    print("=" * 92)
    print(f"inter-UAV coordination probe | {AREA:.0f} m | RCS {RCS} m^2 | P_D_req {PD_REQ}")
    print("=" * 92)

    cfg0, base0, t0, plan0 = build()
    q, link, pd0, _, pds0 = pick_worst(cfg0, t0, plan0)
    M = np.asarray(t0.gamma_sense).shape[0]
    i, j = link
    print(f"worst target q={q} | best view (i,j)={link} | baseline P_D={pd0:.4f}")
    print(f"baseline: gamma={gamma_at(t0, link, q):.5e}  rinr={10 * math.log10(rinr_at(t0, link, q)):+.2f} dB")
    print()

    # ---- A. is the gate口径 self-consistent? ------------------------------
    g_all = mask_of(M, range(M))
    cfg_gate, _, t_gate, plan_gate = build(gate=True, mask=g_all)
    pd_gate = predicted_pd_for_links(cfg_gate, t_gate, q, [link], plan=plan_gate)
    same = math.isclose(gamma_at(t_gate, link, q), gamma_at(t0, link, q), rel_tol=1e-12)
    print("A. gate口径 consistency (gate=True with an all-True mask should be a no-op):")
    print(f"   gamma: release {gamma_at(t0, link, q):.6e} vs gated {gamma_at(t_gate, link, q):.6e} -> "
          f"{'identical' if same else 'DIFFERENT'}")
    if not same:
        print("   !! the gate switches radiated power from P_sense (=rho*P) to P_sense+P_comm (=P),")
        print("      so enabling coordination silently changes the baseline too (model.py:623-624).")
    print()

    # ---- B. how much does silencing nodes buy? ----------------------------
    print("B. coordination gain (only the listed nodes radiate during the CPI):")
    print(f"   {'radiators':>10}{'rinr (dB)':>12}{'gamma':>14}{'P_D':>10}{'vs base':>10}")
    rows = []
    for label, members in [
        ("all 15", list(range(M))),
        ("8", [i, j] + [k for k in range(M) if k not in (i, j)][:6]),
        ("4", [i, j] + [k for k in range(M) if k not in (i, j)][:2]),
        ("2 (i,j)", [i, j]),
        ("1 (i)", [i]),
    ]:
        cfg_b, _, t_b, plan_b = build(gate=True, mask=mask_of(M, members))
        gv = gamma_at(t_b, link, q)
        rv = rinr_at(t_b, link, q)
        pv = predicted_pd_for_links(cfg_b, t_b, q, [link], plan=plan_b)
        rows.append((label, 10 * math.log10(rv), gv, pv))
        print(f"   {label:>10}{10 * math.log10(rv):>12.2f}{gv:>14.5e}{pv:>10.4f}{pv - pd0:>+10.4f}")

    # self-gate: fewer radiators must mean less interference
    rins = [r for _, r, _, _ in rows]
    if not all(a >= b - 1e-9 for a, b in zip(rins, rins[1:])):
        raise SystemExit("SELF-GATE FAILED: interference did not fall with fewer radiators")
    print("   self-gate OK: interference falls monotonically as fewer nodes radiate")
    print()

    # ---- C. deeper cancellation (what a CSI exchange would fund) ----------
    print("C. deeper direct-path cancellation (kappa_dc), under full vs coordinated radiation:")
    print(f"   {'kappa':>8}{'rinr all (dB)':>16}{'P_D all':>10}{'rinr coor (dB)':>16}{'P_D coor':>10}")
    for kappa in (40.0, 50.0, 60.0, 80.0):
        cfg_a, _, t_a, p_a = build(gate=True, mask=mask_of(M, range(M)), kappa_db=kappa)
        cfg_c, _, t_c, p_c = build(gate=True, mask=mask_of(M, [i, j]), kappa_db=kappa)
        print(f"   {kappa:>8.0f}{10 * math.log10(rinr_at(t_a, link, q)):>16.2f}"
              f"{predicted_pd_for_links(cfg_a, t_a, q, [link], plan=p_a):>10.4f}"
              f"{10 * math.log10(rinr_at(t_c, link, q)):>16.2f}"
              f"{predicted_pd_for_links(cfg_c, t_c, q, [link], plan=p_c):>10.4f}")
    print()

    # ---- D. the real trade-off: coordination costs looks -------------------
    print("D. trade-off: time-sharing for a quiet spectrum costs looks per target")
    print(f"   {'slots':>7}{'looks/tgt':>11}{'interference':>14}{'P_D':>10}   note")
    for slots in (1, 2, 5, 10):
        looks = max(int(round(16 / slots)), 1)
        cfg_d, _, t_d, p_d = build(looks=looks, gate=True, mask=mask_of(M, [i, j]))
        pv = predicted_pd_for_links(cfg_d, t_d, q, [link], plan=p_d)
        note = "baseline (no time sharing)" if slots == 1 else ""
        print(f"   {slots:>7}{looks:>11}{10 * math.log10(rinr_at(t_d, link, q)):>+13.2f} dB"
              f"{pv:>10.4f}   {note}")
    print()
    print("   read: coordination is only worth it if the interference reduction beats the")
    print("   sqrt(looks) loss from time-sharing. Compare rows against the L=16 baseline")
    print(f"   P_D = {pd0:.4f} at rinr = {10 * math.log10(rinr_at(t0, link, q)):+.2f} dB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
