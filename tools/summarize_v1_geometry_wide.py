"""Summarize the deployment-distance sweep and pair it with the frozen runs.

Reads ``results_v1_geometry_wide`` (this sweep) and three pre-existing
directories that share the identical protocol (seed 10917, MC=100,
``target-local-v1`` preset, Gaussian replacement, edit-free):

* ``results_v1_rcs_600m``   -- 600 m square, RCS 0.05/0.1/0.2 m^2, K = 0/2/8
* ``results_v1_rcs_joint``  -- 4000 m square, RCS 0.05/0.1/0.2 m^2, K = 0/2/8
* ``results_v1_rcs_wide``   -- 4000 m square, RCS 1/5/20/50 m^2, K = 0/8

Because every directory draws the same 100 base scenarios, contrasts across
directories can be paired trial-by-trial rather than compared as two
independent samples.

Outputs, into ``results_v1_geometry_wide``:

* ``geometry_summary.csv``      -- tidy table, one row per (family, area, RCS,
  budget, method), for the compact curve and the distance axis.
* ``geometry_contrasts.json``   -- paired report-budget and deployment-size
  contrasts, plus the equivalent-RCS shift of the compact deployment.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import interval

MAIN = Path('results_v1_geometry_wide')
COMPACT_600 = Path('results_v1_rcs_600m')
JOINT_4KM = Path('results_v1_rcs_joint')
WIDE_4KM = Path('results_v1_rcs_wide')

METHODS = ['v1', 'joint', 'balanced', 'robust']
FIELDS = ['pd', 'pfa', 'reports', 'observations', 'active_targets', 'capture',
          'bits', 'delay_ms']
RCS_COMPACT = [0.05, 0.1, 0.2, 1.0, 5.0, 20.0, 50.0]
AREA_600 = 600.0
AREA_4000 = 4000.0
AXIS_RCS = 0.2
AXIS_BUDGET = 8


def load(directory, area):
    """Return {condition: {method: {trial: row}}} plus a {condition: overrides} map."""
    directory = Path(directory)
    rows = list(csv.DictReader((directory / 'trials.csv').open(encoding='utf8')))
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf8'))
    table = {}
    for row in rows:
        table.setdefault(row['condition'], {}).setdefault(row['method'], {})[
            int(row['trial'])] = row
    overrides = protocol['conditions']
    return {'dir': directory, 'area': area, 'table': table,
            'overrides': overrides, 'mc': protocol['mc']}


def series(loaded, condition, method):
    """Per-trial P_D indexed by trial, or None if the condition is absent."""
    per_method = loaded['table'].get(condition)
    if not per_method or method not in per_method:
        return None
    trials = per_method[method]
    return {trial: float(trials[trial]['pd']) for trial in sorted(trials)}


def mean_of(loaded, condition, method, field):
    per_method = loaded['table'].get(condition)
    if not per_method or method not in per_method:
        return None
    values = [float(r[field]) for r in per_method[method].values()]
    return float(np.mean(values))


def paired(a, b):
    """Paired difference a-b over the shared trial ids, with a 95% interval."""
    shared = sorted(set(a) & set(b))
    diff = [a[t] - b[t] for t in shared]
    return {'n': len(shared), 'mean': float(np.mean(diff)), 'ci95': interval(diff),
            'trials': shared}


def rcs_at_pd(curve, target):
    """Log-RCS interpolation: which RCS on ``curve`` gives this P_D?

    ``curve`` is a list of (rcs, pd) sorted by RCS.  Returns None when the
    target is outside the spanned range.
    """
    curve = sorted(curve)
    for (r0, p0), (r1, p1) in zip(curve, curve[1:]):
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            w = (target - p0) / (p1 - p0)
            return float(np.exp(np.log(r0) + w * (np.log(r1) - np.log(r0))))
    return None


def main():
    main_run = load(MAIN, None)
    compact_600 = load(COMPACT_600, AREA_600)
    joint_4km = load(JOINT_4KM, AREA_4000)
    wide_4km = load(WIDE_4KM, AREA_4000)

    # ---------------- tidy table ----------------
    rows = []
    for rcs in RCS_COMPACT:
        for budget in (0, 8):
            condition = f'rcs{rcs:g}_k{budget}'
            for method in METHODS:
                row = {'family': 'compact_600m', 'area_side_m': AREA_600,
                       'rcs_m2': rcs, 'report_budget': budget, 'method': method}
                got = any(mean_of(src, condition, method, 'pd') is not None
                          for src in (main_run, compact_600))
                if not got:
                    continue
                for field in FIELDS:
                    value = mean_of(main_run, condition, method, field)
                    if value is None:
                        value = mean_of(compact_600, condition, method, field)
                    row[field] = value
                rows.append(row)
    for rcs in RCS_COMPACT:
        for budget in (0, 8):
            condition = f'rcs{rcs:g}_k{budget}'
            for method in METHODS:
                if mean_of(joint_4km, condition, method, 'pd') is None and \
                        mean_of(wide_4km, condition, method, 'pd') is None:
                    continue
                row = {'family': 'baseline_4km', 'area_side_m': AREA_4000,
                       'rcs_m2': rcs, 'report_budget': budget, 'method': method}
                for field in FIELDS:
                    value = mean_of(joint_4km, condition, method, field)
                    if value is None:
                        value = mean_of(wide_4km, condition, method, field)
                    row[field] = value
                rows.append(row)
    distance_points = [AREA_600]
    for name, overrides in main_run['overrides'].items():
        if 'geometry.area_xy' in overrides and name.startswith('d'):
            distance_points.append(float(overrides['geometry.area_xy']))
    distance_points.append(AREA_4000)
    for area in sorted(set(distance_points)):
        row = {'family': 'distance_axis', 'area_side_m': area, 'rcs_m2': AXIS_RCS,
               'report_budget': AXIS_BUDGET, 'method': 'v1'}
        if area == AREA_600:
            src, condition = compact_600, f'rcs{AXIS_RCS:g}_k{AXIS_BUDGET}'
        elif area == AREA_4000:
            src, condition = joint_4km, f'rcs{AXIS_RCS:g}_k{AXIS_BUDGET}'
        else:
            src, condition = main_run, f'd{area:g}_rcs{AXIS_RCS:g}_k{AXIS_BUDGET}'
        if mean_of(src, condition, 'v1', 'pd') is None:
            continue
        for field in FIELDS:
            row[field] = mean_of(src, condition, 'v1', field)
        rows.append(row)
    with (MAIN / 'geometry_summary.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['family', 'area_side_m', 'rcs_m2',
                                               'report_budget', 'method'] + FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- contrasts ----------------
    out = {'note': 'Paired over the 100 shared base scenarios (seed 10917). '
                   'Exploratory single-geometry screen, not an equivalence claim.',
           'compact_minus_4km': [], 'compact_k8_minus_k0': [],
           'distance_minus_4km': [], 'equivalent_rcs': []}

    # 600 m vs 4 km at matched RCS, both methods of interest
    for rcs in RCS_COMPACT:
        for budget in (0, 8):
            condition = f'rcs{rcs:g}_k{budget}'
            for method in ('v1', 'joint', 'balanced'):
                compact = series(main_run, condition, method) or \
                    series(compact_600, condition, method)
                baseline = series(joint_4km, condition, method) or \
                    series(wide_4km, condition, method)
                if compact is None or baseline is None:
                    continue
                contrast = paired(compact, baseline)
                out['compact_minus_4km'].append({
                    'rcs_m2': rcs, 'report_budget': budget, 'method': method,
                    'pd_600m': float(np.mean(list(compact.values()))),
                    'pd_4000m': float(np.mean(list(baseline.values()))),
                    **{k: contrast[k] for k in ('n', 'mean', 'ci95')},
                    'family_ci95': interval(
                        [compact[t] - baseline[t] for t in contrast['trials']],
                        1 - .05 / 36)})

    # remote-report gain inside the compact deployment
    for rcs in RCS_COMPACT:
        for method in ('v1', 'joint', 'balanced', 'robust'):
            high = series(main_run, f'rcs{rcs:g}_k8', method) or \
                series(compact_600, f'rcs{rcs:g}_k8', method)
            low = series(main_run, f'rcs{rcs:g}_k0', method) or \
                series(compact_600, f'rcs{rcs:g}_k0', method)
            if high is None or low is None:
                continue
            contrast = paired(high, low)
            out['compact_k8_minus_k0'].append({
                'rcs_m2': rcs, 'method': method, 'pd_k8': float(np.mean(list(high.values()))),
                'pd_k0': float(np.mean(list(low.values()))),
                **{k: contrast[k] for k in ('n', 'mean', 'ci95')},
                'family_ci95': interval(
                    [high[t] - low[t] for t in contrast['trials']], 1 - .05 / 28)})

    # P_D versus deployment side at fixed RCS / budget
    baseline_axis = series(joint_4km, f'rcs{AXIS_RCS:g}_k{AXIS_BUDGET}', 'v1')
    for area in sorted(set(distance_points)):
        if area == AREA_600:
            point = series(compact_600, f'rcs{AXIS_RCS:g}_k{AXIS_BUDGET}', 'v1')
        elif area == AREA_4000:
            point = baseline_axis
        else:
            point = series(main_run, f'd{area:g}_rcs{AXIS_RCS:g}_k{AXIS_BUDGET}', 'v1')
        if point is None:
            continue
        entry = {'area_side_m': area, 'rcs_m2': AXIS_RCS,
                 'report_budget': AXIS_BUDGET, 'pd': float(np.mean(list(point.values())))}
        if baseline_axis is not None and area != AREA_4000:
            contrast = paired(point, baseline_axis)
            entry.update({k: contrast[k] for k in ('n', 'mean', 'ci95')})
        out['distance_minus_4km'].append(entry)

    # How much RCS does the compact deployment buy back?  Match P_D between the
    # compact curve and the 4 km curve by interpolating on log-RCS.
    for budget in (0, 8):
        compact_curve, baseline_curve = [], []
        for rcs in RCS_COMPACT:
            condition = f'rcs{rcs:g}_k{budget}'
            compact = series(main_run, condition, 'v1') or series(compact_600, condition, 'v1')
            baseline = series(joint_4km, condition, 'v1') or series(wide_4km, condition, 'v1')
            if compact is not None:
                compact_curve.append((rcs, float(np.mean(list(compact.values())))))
            if baseline is not None:
                baseline_curve.append((rcs, float(np.mean(list(baseline.values())))))
        for rcs, pd_compact in compact_curve:
            if not (min(p for _, p in baseline_curve) <= pd_compact
                    <= max(p for _, p in baseline_curve)):
                continue
            rcs_equivalent = rcs_at_pd(baseline_curve, pd_compact)
            if rcs_equivalent is None:
                continue
            out['equivalent_rcs'].append({
                'report_budget': budget, 'rcs_compact_m2': rcs,
                'pd_compact': pd_compact, 'rcs_equivalent_4km_m2': rcs_equivalent,
                'rcs_ratio': rcs_equivalent / rcs,
                'equivalent_gain_db': 10 * float(np.log10(rcs_equivalent / rcs))})

    (MAIN / 'geometry_contrasts.json').write_text(
        json.dumps(out, indent=2), encoding='utf8')
    print('wrote', MAIN / 'geometry_summary.csv', 'and', MAIN / 'geometry_contrasts.json')
    print('rows in summary:', len(rows))


if __name__ == '__main__':
    main()
