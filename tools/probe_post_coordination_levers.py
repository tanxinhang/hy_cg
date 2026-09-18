"""Does coordination RE-ORDER the levers? (locked scenario 500 m / RCS 0.2)

Hypothesis, from the audit trail rather than from intuition:

  * The uncoordinated working point is interference limited (rinr ~ +11 dB).
  * Silencing the non-participating illuminators drops rinr below 0 dB, i.e. it
    moves the system into the NOISE-limited regime.
  * In that regime the denominator-class knobs revive: gamma(mP)/gamma(P) =
    m(1+r)/(1+mr) -> m as r -> 0, so transmit power stops being saturated.
  * Therefore the *ordering* of the remaining levers should differ between the two
    arms, and the cheapest one (a 1.25x power step) may be worth far more after
    coordination than before it.

This is the "cancel first, then raise power" ordering the model audit derived on a
different axis, now tested inside the coordination mechanism itself. If the
hypothesis holds, the remaining performance gap has a cheap handle; if it does not,
the gap is a pure link-budget boundary and should be reported as such.

Levers are measured on a FROZEN link (the worst target's best delivering view,
chosen once per arm) so link switching cannot contaminate the comparison. The
surrogate `predicted_pd_for_links` is used for ORDERING only -- the formal MC
experiment showed it overstates magnitudes by ~2x.
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
from isac_sim.coordination import illuminator_mask  # noqa: E402
from isac_sim.fusion import predicted_pd_for_links  # noqa: E402
from isac_sim.model import compute_link_tables, build_base_gains, generate_geometry  # noqa: E402
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402
from isac_sim.selection import select_c2f_adaptive  # noqa: E402
from isac_sim.soft_channel import received_moments  # noqa: E402

AREA = 500.0
RCS = 0.2
PENALTY = 0.2
SEED = 20260917


def make_cfg(penalty: float = 0.0, looks: int = 16, pscale: float = 1.0,
             gate: bool = False):
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    cfg = apply_overrides(
        cfg,
        {
            "geometry.area_xy": AREA,
            "detect.target_rcs": RCS,
            "detect.n_looks": looks,
            "radio.P_default": pscale,
            "interference.sense_gate_by_active_tx": gate,
            "selector.tx_penalty": penalty,
        },
    )
    validate_config(cfg)
    return cfg


def worst_view(cfg, base, geom, mask):
    """Worst target and its best DELIVERING view under this arm."""
    tables = compute_link_tables(cfg, base, active_tx_mask=mask)
    plan = assign_fusion_nodes(cfg, base, tables, geom=geom)
    return tables, plan


def pd_of(cfg, tables, plan, q, link):
    mom = received_moments(cfg, tables, link, q, plan)
    if float(mom.m1) <= 0.0:
        return None
    return float(predicted_pd_for_links(cfg, tables, q, [link], plan=plan))


def main() -> int:
    print("=" * 96)
    print(f"post-coordination lever ordering | {AREA:.0f} m | RCS {RCS} m^2")
    print("=" * 96)

    geom = generate_geometry(make_cfg(), np.random.default_rng(SEED))
    base = build_base_gains(make_cfg(), geom, np.random.default_rng(SEED))

    arms = {}
    for arm, penalty, coordinate in (("uncoordinated", 0.0, False),
                                     ("coordinated", PENALTY, True)):
        cfg = make_cfg(penalty=penalty, gate=coordinate)
        t0 = compute_link_tables(cfg, base, active_tx_mask=None)
        plan0 = assign_fusion_nodes(cfg, base, t0, geom=geom)
        selected, _, _ = select_c2f_adaptive(cfg, base, t0, plan=plan0)
        mask = illuminator_mask(selected, cfg.scale.M) if coordinate else None
        arms[arm] = (cfg, selected, mask)
        r = np.asarray(compute_link_tables(cfg, base, active_tx_mask=mask).rinr)
        r = r[np.isfinite(r) & (r > 0)]
        print(f"  {arm:>15}: radiators={cfg.scale.M if mask is None else int(mask.sum()):>3}  "
              f"median rinr = {10 * math.log10(np.median(r)):+.2f} dB  "
              f"({'INTERFERENCE limited' if np.median(r) > 1 else 'NOISE limited'})")
    print()

    results = {}
    for arm, (cfg, selected, mask) in arms.items():
        tables, plan = worst_view(cfg, base, geom, mask)
        n_tgt = cfg.scale.Q
        # worst target by its delivered 2-link bundle, then freeze one best view
        pds = []
        for q in range(n_tgt):
            links = [lk for lk in selected.get(q, [])
                     if float(received_moments(cfg, tables, lk, q, plan).m1) > 0]
            pds.append(float(predicted_pd_for_links(cfg, tables, q, links, plan=plan))
                       if links else 0.0)
        q = int(np.argmin(pds))
        links = [lk for lk in selected.get(q, [])
                 if float(received_moments(cfg, tables, lk, q, plan).m1) > 0]
        if not links:  # no delivered view: fall back to the best single raw view
            cand = [(float(np.asarray(tables.gamma_sense)[i, j, q]), (i, j))
                    for i in range(cfg.scale.M) for j in range(cfg.scale.M) if i != j]
            cand.sort(reverse=True)
            links = [cand[0][1]]
        link = links[0]
        results[arm] = dict(q=q, link=link, base_pd=float(pd_of(cfg, tables, plan, q, link) or 0.0),
                            mask=mask, active_q=q)

    # ---- lever sweep on each arm, frozen link ------------------------------
    for arm, r in results.items():
        q, link, mask = r["active_q"], r["link"], r["mask"]
        print("=" * 96)
        print(f"arm {arm}: worst target q={q}, frozen view {link}, "
              f"baseline P_D={r['base_pd']:.4f}")
        print("=" * 96)
        print(f"  {'lever':>18}{'rinr (dB)':>12}{'gamma x':>10}{'P_D':>10}{'vs arm base':>13}")
        for label, looks, pscale in (
            ("baseline", 16, 1.0),
            ("looks x4 (L=64)", 64, 1.0),
            ("looks x16 (L=256)", 256, 1.0),
            ("power x1.25", 16, 1.25),
            ("power x2", 16, 2.0),
            ("power x10", 16, 10.0),
            ("power1.25 + L64", 64, 1.25),
            ("power1.25 + L256", 256, 1.25),
        ):
            cfg2 = make_cfg(penalty=(PENALTY if mask is not None else 0.0), looks=looks,
                            pscale=pscale, gate=(mask is not None))
            tabs = compute_link_tables(cfg2, base, active_tx_mask=mask)
            plan2 = assign_fusion_nodes(cfg2, base, tabs, geom=geom)
            p = pd_of(cfg2, tabs, plan2, q, link)
            rr = float(np.asarray(tabs.rinr)[link[0], link[1]])
            gx = float(np.asarray(tabs.gamma_sense)[link[0], link[1], q])
            g0 = float(np.asarray(compute_link_tables(cfg2, base, active_tx_mask=mask).gamma_sense)[
                link[0], link[1], q])
            print(f"  {label:>18}{10 * math.log10(rr) if rr > 0 else float('nan'):>12.2f}"
                  f"{gx:>10.4f}{(p if p is not None else float('nan')):>10.4f}"
                  f"{((p - r['base_pd']) if p is not None else float('nan')):>+13.4f}")
        print()

    print("reading: if the power rows gain far more in the coordinated arm than in the")
    print("         uncoordinated one, coordination has moved the working point into the")
    print("         noise-limited regime and cheapest lever is no longer looks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
