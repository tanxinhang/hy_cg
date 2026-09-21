"""Is ``N``/``L`` one resolution lever, or two knobs wearing one name?

Written before the resolution axis of the identifiability question is swept, for
the same reason the angular probe was: a sweep is only interpretable if the thing
it changes is known.  Reads ``isac_sim`` only, writes a CSV.

The mappings under test (``isac_sim/sensing/model.py``)::

    delay_bin   = round(tau * L * delta_f)      valid when 0 <= l < L
    doppler_bin = round(nu  * N * T)            valid when -N/2 <= k < N/2
    bandwidth   = N * delta_f                   -> noise_power ~ B
    G_proc      = N * L                         (= B * CPI, physically right)
    K           = N * L

so the delay axis is quantised on ``L * delta_f`` while the noise bandwidth --
and hence the physical delay resolution ``1 / B`` -- is ``N * delta_f``.  **The
two coincide only at the default ``N == L == 64``**, which is why the model looks
self-consistent until someone sweeps.

Measured consequences at the 600 m work point (see
``ANGULAR_IDENTIFIABILITY_PROBE.md`` §七):

* ``L`` only (64 -> 256): B and the noise power do not move, ``G_proc`` x4, the
  delay cell goes 78.1 -> 19.5 m, and the fraction of colliding links falls
  11.9% -> 2.3%.  That is re-quantisation, not resolution.
* ``N`` only (64 -> 256): B x4 so the noise floor costs 6 dB, the Doppler cell
  goes 11.9 -> 3.0 m/s (a real gain, the CPI lengthens), but the delay cell does
  not move, so collisions only reach 2.7% -- a 4x *real* resolution gain buys
  less than pure re-quantisation of ``L`` does.

``collision_penalty = 1 / count**alpha`` (``model.py``) therefore tracks the
rounding grid, not the bandwidth.  Run ``--collisions`` for that table too.

Run::

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/probe_waveform_mapping.py --area 600 --out results_waveform_mapping
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from isac_sim.sensing import model as md
from isac_sim.core.config import Config

C = 3e8
LAM = 3e8 / 5.9e9

CASES: Tuple[Tuple[int, int], ...] = (
    (64, 64), (64, 128), (64, 256), (128, 64), (256, 64), (128, 128), (256, 256),
)


def _cfg(n: int, l: int, area: float) -> Config:
    cfg = Config()
    cfg.waveform.N = n
    cfg.waveform.L = l
    cfg.geometry.area_xy = area
    return cfg


def mapping_rows(area: float) -> List[Dict[str, float]]:
    """Pure algebra: what each axis is quantised on, and what it costs."""
    rows: List[Dict[str, float]] = []
    for n, l in CASES:
        cfg = _cfg(n, l, area)
        w = cfg.waveform
        b = md.bandwidth(cfg)
        rows.append({
            "N": n,
            "L": l,
            "area_m": area,
            "B_noise_MHz": b / 1e6,
            "noise_power_relative": b / (64 * w.delta_f),
            "G_proc": float(n * l),
            "K": n * l,
            "delay_cell_from_L_m": C / (2.0 * l * w.delta_f),
            "delay_cell_from_B_m": C / (2.0 * b),
            "doppler_cell_mps": LAM / (2.0 * n * w.T),
            "max_range_km": C / (2.0 * w.delta_f) / 1e3,
            "max_vel_mps": LAM * w.delta_f / 4.0,
            "self_consistent": int(n == l),
        })
    return rows


def collision_rows(area: float, seed: int = 12345) -> List[Dict[str, float]]:
    """The production collision penalty on one fixed scene, per (N, L).

    One geometry and one RNG seed for every case, so the only thing that moves
    between rows is the grid.  ``dd_collision_count`` is the number of targets
    sharing one *rounded* ``(delay_bin, doppler_bin)`` cell, and it multiplies
    the sensing signal as ``1 / count**alpha`` (``model.py``:900).
    """
    cfg0 = _cfg(64, 64, area)
    rng = np.random.default_rng([seed, 777])
    geom = md.generate_geometry(cfg0, rng)

    rows: List[Dict[str, float]] = []
    for n, l in CASES:
        cfg = _cfg(n, l, area)
        rng = np.random.default_rng([seed, 0])
        base = md.build_base_gains(cfg, geom, rng)
        cc = np.asarray(base.dd_collision_count, dtype=float)
        valid = np.asarray(base.valid_dd, dtype=bool)
        alpha = float(cfg.dd.dd_collision_alpha)
        penalty = 1.0 / (np.maximum(cc, 1.0) ** alpha)
        m = cc.shape[0]
        sel = (~np.eye(m, dtype=bool)[:, :, None]) & valid
        c, pen = cc[sel], penalty[sel]
        rows.append({
            "N": n,
            "L": l,
            "area_m": area,
            "collision_penalty_enabled": int(bool(cfg.dd.enable_dd_collision_penalty)),
            "frac_valid_links": float(valid.mean()),
            "frac_colliding_links": float((c > 1.0).mean()) if c.size else float("nan"),
            "collision_count_max": float(c.max()) if c.size else float("nan"),
            "penalty_median": float(np.median(pen)) if pen.size else float("nan"),
            "penalty_min": float(pen.min()) if pen.size else float("nan"),
            "penalty_loss_db_worst": float(-10.0 * np.log10(pen.min())) if pen.size else float("nan"),
        })
    return rows


def _write(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_mapping(rows: List[Dict[str, float]]) -> None:
    print("  %-9s %9s %9s %9s %12s %12s %11s %9s" % (
        "N x L", "B MHz", "noise x", "G_proc", "delay cell L", "delay cell B",
        "doppler cell", "N==L"))
    print("  %-9s %9s %9s %9s %12s %12s %11s %9s" % (
        "", "", "", "N*L", "m", "m", "m/s", ""))
    for r in rows:
        print("  %-9s %9.2f %9.2f %9.0f %12.1f %12.1f %11.1f %9s" % (
            "%dx%d" % (r["N"], r["L"]), r["B_noise_MHz"], r["noise_power_relative"],
            r["G_proc"], r["delay_cell_from_L_m"], r["delay_cell_from_B_m"],
            r["doppler_cell_mps"], "yes" if r["self_consistent"] else "NO"))


def _print_collisions(rows: List[Dict[str, float]]) -> None:
    print("  %-9s %11s %12s %12s %12s %8s" % (
        "N x L", "valid", "colliding", "count max", "pen med", "worst dB"))
    for r in rows:
        print("  %-9s %11.3f %12.3f %12.0f %12.4f %8.2f" % (
            "%dx%d" % (r["N"], r["L"]), r["frac_valid_links"],
            r["frac_colliding_links"], r["collision_count_max"],
            r["penalty_median"], r["penalty_loss_db_worst"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", type=float, default=600.0,
                    help="scene area in metres (600 = the low-RCS work point)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--collisions", action="store_true",
                    help="also evaluate the production collision penalty on one scene")
    ap.add_argument("--out", default="results_waveform_mapping")
    args = ap.parse_args(argv)

    w = Config().waveform
    print("waveform: delta_f %.0f kHz, T %.3f us, fc %.1f GHz, area %.0f m" % (
        w.delta_f / 1e3, w.T * 1e6, w.fc / 1e9, args.area))
    print()
    print("  the delay axis is quantised on L*delta_f; the noise bandwidth is")
    print("  N*delta_f.  Only N == L makes those the same statement.")
    print()
    rows = mapping_rows(args.area)
    _print_mapping(rows)
    print()
    print("  read: 'delay cell L' is what the grid resolves; 'delay cell B' is")
    print("  what 1/B says the waveform can resolve.  They differ by N/L.")
    _write(os.path.join(args.out, "waveform_mapping.csv"), rows)

    if args.collisions:
        crows = collision_rows(args.area, args.seed)
        print()
        print("scene collisions (one geometry, one seed; area %.0f m, alpha = %.2f):"
              % (args.area, Config().dd.dd_collision_alpha))
        _print_collisions(crows)
        _write(os.path.join(args.out, "waveform_collisions.csv"), crows)

    print()
    print("wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
