"""Can mobility buy an easier observation geometry, and does placement already get it?

Claim under test
----------------
The simulator is a **single-interval snapshot**: ``generate_geometry`` draws
static positions and the velocity vectors ``v_uav`` / ``v_tgt`` only feed the
OTFS Doppler bins and the belief prediction -- no code path writes ``p + v*t``
back into the geometry.  So "move the fleet to a better place, then sense" is
mathematically identical to "start the fleet in that place".  Mobility has no
independent value here; **placement** does.

This script therefore prices the *placement* question directly, under three
architectures, all measured against the frozen ``target-local-v1`` operating
point with the model's own gain functions:

* ``compress``  -- shrink the whole formation (the already-measured axis).
                   Every length shrinks together: the two-way echo
                   ``target_gain = lam^2 rcs / ((4 pi)^3 d_iq^2 d_jq^2)`` is
                   ``R^-4`` while the one-way direct path (``path_gain`` with
                   ``detect.path_loss_exp = 2.0``) is ``R^-2``, so the net
                   sensing SINR can only follow ``R^-2`` -- 20 dB per decade.
* ``dispatch``  -- send only a two-UAV task group (one illuminator, one
                   receiver) to radius r around one target, leaving the rest of
                   the fleet at its 4 km stations.  The pair's *own* direct path
                   is now ``2r``, which is exactly as short as the two echo legs
                   -- so this cannot escape the same ``R^-2`` law.  Whatever the
                   pair gains on the echo it pays back as mutual interference.
* ``dispatch+cancel`` -- same geometry, but the direct-path residual is pushed
                   below the receiver noise floor.  Only then does the echo's
                   ``R^-4`` actually appear, i.e. moving closer stops being a
                   zero-sum trade between echo and self-interference.

The triangle inequality is the reason ``dispatch`` cannot win:
``d_ij <= d_iq + d_jq``, so two nodes within radius ``r`` of the target are at
most ``2r`` apart, and ``SINR = d_ij^2 / (d_iq^2 d_jq^2) <= 4 / r^2``.

Usage:
    python tools/probe_move_geometry.py --mc 8 --seed 10917
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isac_sim.config import apply_overrides, apply_preset, default_config  # noqa: E402
from isac_sim.model import (  # noqa: E402
    Geometry,
    build_base_gains,
    denominator_guard,
    noise_power,
    radar_hardware_gain,
)

PRESET = 'target-local-v1'
AREA_REF = 4000.0


def base_config():
    cfg = apply_preset(default_config(), PRESET)
    return apply_overrides(cfg, {'scale.M': 15, 'scale.Q': 1})


def geometry_compress(cfg, scale, rng, flatten=False):
    """Whole formation shrunk to a square of side ``scale * AREA_REF``.

    ``flatten`` also squeezes the *altitude* bands by the same factor, so the
    three-dimensional deployment shrinks instead of only its footprint.  This
    separates two very different ceilings: the one set by the horizontal spread
    and the one set by the 800-1200 m UAV / 700-1500 m target altitude bands,
    which do not move when only ``geometry.area_xy`` is changed.
    """
    side = scale * AREA_REF
    g = cfg.geometry
    hw_u = (g.h_uav_max - g.h_uav_min) / 2.0
    hc_u = (g.h_uav_max + g.h_uav_min) / 2.0
    hw_t = (g.h_target_max - g.h_target_min) / 2.0
    hc_t = (g.h_target_max + g.h_target_min) / 2.0
    if flatten:
        hw_u, hw_t = hw_u * scale, hw_t * scale
    p_uav = np.zeros((cfg.scale.M, 3))
    p_uav[:, :2] = rng.uniform(0.0, side, size=(cfg.scale.M, 2))
    p_uav[:, 2] = hc_u + rng.uniform(-hw_u, hw_u, size=cfg.scale.M)
    centre = np.array([side / 2.0, side / 2.0, hc_t + rng.uniform(-hw_t, hw_t)])
    p_tgt = centre[None, :]
    return p_uav, p_tgt


def geometry_dispatch(cfg, scale, rng):
    """Two-UAV task group at radius ``r`` around the target; fleet stays put.

    The target sits at the centre of the 4 km station area, so the rest of the
    fleet keeps a realistic ~2 km stand-off instead of collapsing onto the pair.
    """
    r = scale * AREA_REF
    g = cfg.geometry
    M = cfg.scale.M
    cx = cy = AREA_REF / 2.0
    p_uav = np.zeros((M, 3))
    # nodes 0 (illuminator) and 1 (receiver) bracket the target at radius r.
    # This is the best case the triangle inequality allows: d_01 = 2r exactly,
    # so the pair's own direct path is as short as its two echo legs.
    p_uav[0] = np.array([cx + r, cy, 0.0])
    p_uav[1] = np.array([cx - r, cy, 0.0])
    rest = M - 2
    p_uav[2:, 0] = rng.uniform(0.0, AREA_REF, size=rest)
    p_uav[2:, 1] = rng.uniform(0.0, AREA_REF, size=rest)
    p_uav[2:, 2] = rng.uniform(g.h_uav_min, g.h_uav_max, size=rest)
    p_uav[0, 2] = p_uav[1, 2] = rng.uniform(g.h_uav_min, g.h_uav_max)
    p_tgt = np.array([[cx, cy, rng.uniform(g.h_target_min, g.h_target_max)]])
    return p_uav, p_tgt


def measure(cfg, p_uav, p_tgt, seed, cancel_db=None):
    """Median sensing SINR over the bistatic links of the focus target.

    Mirrors ``model.compute_link_tables`` on the ``orthogonal`` phase
    (``P_rad_sense = P_sense``, no payload during the sensing observation) and
    ``interference.coupling = 'shared_spectrum'``.  The DD fractional penalty is
    deliberately left out: this is a propagation-geometry probe, and including
    it would mix in the target's delay-Doppler collision count.
    """
    if cancel_db is not None:
        cfg = apply_overrides(cfg, {'interference.direct_cancellation_db': float(cancel_db)})
    r = cfg.radio
    ic = cfg.interference
    M = cfg.scale.M

    geom = Geometry(p_uav=p_uav,
                    v_uav=np.zeros_like(p_uav),
                    p_tgt=p_tgt,
                    v_tgt=np.zeros_like(p_tgt))
    rng = np.random.default_rng([seed, 0])
    base = build_base_gains(cfg, geom, rng)

    n0 = noise_power(cfg)
    eps = denominator_guard(cfg, n0)
    kappa = 10.0 ** (-ic.direct_cancellation_db / 10.0)
    g_proc = cfg.waveform.N * cfg.waveform.L
    g_hw = radar_hardware_gain(cfg)

    p_sense = np.full(M, r.rho * r.P_default)
    p_rad = p_sense                                   # orthogonal phase
    field = p_rad @ base.direct_gain                  # (M,) at each receiver j
    residual = r.residual_self_factor * r.P_default + kappa * field
    denom = n0 + residual + eps                       # (M,)

    off = ~np.eye(M, dtype=bool)
    signal = p_sense[:, None] * base.target_gain[:, :, 0] * g_proc * g_hw
    sinr = np.where(off, signal / denom[None, :], np.nan)
    vals = sinr[np.isfinite(sinr)]
    if vals.size == 0:
        return {'best_sinr_db': float('nan')}

    # Isolate the strongest bistatic view the network can form and report its
    # own geometry and interference budget.  Averaging over the whole fleet
    # hides the mechanism whenever only a few nodes move.
    i_star, j_star = np.unravel_index(np.nanargmax(sinr), sinr.shape)
    d_iq = float(base.d_uav_tgt[i_star, 0])
    d_jq = float(base.d_uav_tgt[j_star, 0])
    d_ij = float(base.d_uu[i_star, j_star])
    r_eff = float(np.sqrt(max(d_iq * d_jq, 1e-12)))

    power = r.rho * r.P_default
    partner = kappa * power * float(base.direct_gain[i_star, j_star])
    rest_field = float(field[j_star]) - power * float(base.direct_gain[i_star, j_star])
    rest = kappa * max(rest_field, 0.0)
    return {
        'best_sinr_db': float(10.0 * np.log10(np.max(vals))),
        'p90_sinr_db': float(10.0 * np.log10(np.percentile(vals, 90))),
        'sinr_db': float(10.0 * np.log10(np.median(vals))),
        'r_eff_m': r_eff,
        'd_iq_m': d_iq,
        'd_jq_m': d_jq,
        'd_ij_m': d_ij,
        'echo_db': float(10.0 * np.log10(base.target_gain[i_star, j_star, 0] * g_proc * g_hw)),
        'partner_db': float(10.0 * np.log10(max(partner, 1e-300))),
        'rest_db': float(10.0 * np.log10(max(rest, 1e-300))),
        'n0_db': float(10.0 * np.log10(n0)),
        'uav_target_slant_mean_m': float(base.d_uav_tgt.mean()),
        'uav_uav_slant_mean_m': float(base.d_uu[off].mean()),
    }


def sweep(cfg, arch, scales, mc, seed, cancel_db=None, flatten=False):
    out = []
    for s in scales:
        accs = []
        for trial in range(mc):
            rng = np.random.default_rng([seed, trial, int(round(s * 1e6))])
            if arch.startswith('compress'):
                p_uav, p_tgt = geometry_compress(cfg, s, rng, flatten=flatten)
            else:
                p_uav, p_tgt = geometry_dispatch(cfg, s, rng)
            accs.append(measure(cfg, p_uav, p_tgt, seed, cancel_db))
        row = {'arch': arch, 'scale': float(s),
               'approach_m': float(s * AREA_REF),
               'cancel_db': cancel_db if cancel_db is not None else cfg.interference.direct_cancellation_db}
        for k in accs[0]:
            row[k] = float(np.nanmean([a[k] for a in accs]))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc', type=int, default=8)
    ap.add_argument('--seed', type=int, default=10917)
    ap.add_argument('--scales', type=float, nargs='+',
                    default=[1.0, 0.5, 0.25, 0.15, 0.1, 0.05, 0.02])
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    cfg = base_config()
    print(f'preset={PRESET}  interference_model={cfg.comm.interference_model}  '
          f'rho={cfg.radio.rho}  kappa_dc={cfg.interference.direct_cancellation_db} dB  '
          f'M={cfg.scale.M}  mc={args.mc}  seed={args.seed}')

    rows = []
    rows += sweep(cfg, 'compress', args.scales, args.mc, args.seed)
    rows += sweep(cfg, 'compress3d', args.scales, args.mc, args.seed, flatten=True)
    rows += sweep(cfg, 'dispatch', args.scales, args.mc, args.seed)
    for db in (40.0, 60.0, 80.0, 120.0):
        rows += sweep(cfg, 'dispatch', args.scales, args.mc, args.seed, cancel_db=db)

    hdr = (f"{'arch':>10s} {'cancel':>7s} {'approach':>9s} {'r_eff':>7s} {'d_ij':>7s} "
           f"{'echo':>8s} {'partner':>8s} {'rest':>8s} {'n0':>8s} {'bestSINR':>9s}")
    print('\n' + hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['arch']:>10s} {r['cancel_db']:>7.0f} {r['approach_m']:>9.0f} "
              f"{r['r_eff_m']:>7.0f} {r['d_ij_m']:>7.0f} "
              f"{r['echo_db']:>8.1f} {r['partner_db']:>8.1f} {r['rest_db']:>8.1f} "
              f"{r['n0_db']:>8.1f} {r['best_sinr_db']:>9.2f}")

    print('\n--- slopes vs the best view\'s own pair radius (per decade; R^-2 == 20, R^-4 == 40) ---')
    for arch, cancel in [('compress', 40.0), ('compress3d', 40.0), ('dispatch', 40.0),
                         ('dispatch', 60.0), ('dispatch', 80.0), ('dispatch', 120.0)]:
        sub = [r for r in rows if r['arch'] == arch and abs(r['cancel_db'] - cancel) < 1e-6]
        sub.sort(key=lambda r: r['r_eff_m'])
        if len(sub) < 2:
            continue
        x = np.log10([r['r_eff_m'] for r in sub])
        y_echo = np.array([r['echo_db'] for r in sub])
        y_ptn = np.array([r['partner_db'] for r in sub])
        y_sinr = np.array([r['best_sinr_db'] for r in sub])
        f = lambda v: float(np.polyfit(x, v, 1)[0])
        print(f'  {arch:>9s} kappa_dc={cancel:>5.0f} dB   '
              f'echo={f(y_echo):+7.2f}  partner_interf={f(y_ptn):+7.2f}  '
              f'SINR={f(y_sinr):+7.2f} dB/decade')

    if args.out:
        p = Path(args.out)
        p.mkdir(parents=True, exist_ok=True)
        (p / 'move_geometry.json').write_text(
            json.dumps({'preset': PRESET, 'mc': args.mc, 'seed': args.seed,
                        'rows': rows}, indent=1), encoding='utf-8')
        print(f'\nwrote {p / "move_geometry.json"}')


if __name__ == '__main__':
    main()
