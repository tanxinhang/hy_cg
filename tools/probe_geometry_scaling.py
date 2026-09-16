"""Exact geometry factors behind the compact-deployment gain.

Compressing the formation shrinks two lengths at once:

* the two-way target path, ``target_gain = lam^2 * rcs / ((4 pi)^3 * d_iq^2 * d_jq^2)``
  (``isac_sim/model.py``), which the sensing signal rides, and
* the one-way illuminator-to-receiver direct path, ``path_gain`` with
  ``detect.path_loss_exp = 2.0``, which carries the dominant interference.

Because they shrink together, the SINR gain is far smaller than the raw
``R^-4`` radar path-loss advantage suggests.  This script reports the two
factors separately, always with the frozen ``target-local-v1`` config, so the
number can be quoted as a mechanism rather than an empirical curve fit.

Usage:
    python tools/probe_geometry_scaling.py --mc 100 --seed 10917
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config
from isac_sim.config import apply_overrides
from isac_sim.model import build_base_gains, generate_geometry


def factors(area, mc, seed):
    """Mean two-way target gain factor and one-way direct-path gain factor."""
    cfg = apply_overrides(config(seed, 8), {'geometry.area_xy': float(area)})
    target_terms, direct_terms = [], []
    slant_target, slant_uav = [], []
    for trial in range(mc):
        rng = np.random.default_rng([seed, trial])
        truth = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, truth, rng)
        m = base.target_gain.shape[0]
        off = ~np.eye(m, dtype=bool)
        # Mean over ordered (illuminator, receiver, target) triples, matching
        # how the sensing equation sums the field at the receiver.
        target_terms.append(float(base.target_gain.mean()))
        direct_terms.append(float(base.direct_gain[off].mean()))
        slant_target.append(float(base.d_uav_tgt.mean()))
        slant_uav.append(float(base.d_uu[off].mean()))
    return {'area_side_m': float(area),
            'target_gain_mean': float(np.mean(target_terms)),
            'direct_gain_mean': float(np.mean(direct_terms)),
            'uav_target_slant_mean_m': float(np.mean(slant_target)),
            'uav_uav_slant_mean_m': float(np.mean(slant_uav))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mc', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10917)
    parser.add_argument('--areas', type=float, nargs='+',
                        default=[4000.0, 3000.0, 2000.0, 1500.0, 1000.0, 800.0, 600.0])
    args = parser.parse_args()

    results = [factors(area, args.mc, args.seed) for area in args.areas]
    print(f'{"area":>7} {"d(uav,tgt)":>11} {"d(uav,uav)":>11} '
          f'{"2-way d^-4":>11} {"1-way d^-2":>11} {"net SINR":>9}')
    base = results[0]
    for row in results:
        two_way = 10 * np.log10(row['target_gain_mean'] / base['target_gain_mean'])
        one_way = 10 * np.log10(row['direct_gain_mean'] / base['direct_gain_mean'])
        print(f'{row["area_side_m"]:>7.0f} {row["uav_target_slant_mean_m"]:>11.1f} '
              f'{row["uav_uav_slant_mean_m"]:>11.1f} {two_way:>+11.2f} '
              f'{one_way:>+11.2f} {two_way - one_way:>+9.2f}')


if __name__ == '__main__':
    main()
