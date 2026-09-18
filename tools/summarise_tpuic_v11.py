"""Reduce a TP-UIC V1.1 output directory into the markdown tables of TP_UIC_V11.md.

Usage:
    python tools/summarise_tpuic_v11.py results_tp_uic_v11

The directory is expected to contain (as written by tools/run_tp_uic_v11.py):

    arms.csv           per (trial, arm): matched H1/H0 GLRT statistics and the
                       canceller diagnostics shipped alongside them
    masking.csv        per (trial, arm, n_nuisance): the masking curve
    identifiability.csv per (trial, arm): on-grid / manifold / off-grid rho
    single_target.csv  per (trial, arm): single-target mechanism validation

Reduction rules -- kept identical for every run so two runs stay comparable:

  * ``C_IC``    -- median over trials of ``kappa_db`` (dB), the *effective*
                   direct-cancellation depth the residmodel reads back
  * ``eta_surv``-- median of the realised weak-echo survival fraction
  * ``rho_w``   -- median of the weighted escape fraction (1 = fully separable)
  * ``T_H1``    -- median matched-filter statistic under H1
  * ``P_D`` / ``P_FA`` -- pooled detection / false-alarm rates over all trials
                   at the analytic CFAR threshold (p_fa from the run)

Arms are emitted in the order the driver declares them, so each table reads
top-down from the naive baselines to the full method.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ARMS = (
    "no_ic",
    "fixed_kappa",
    "plain_ls",
    "ridge_ls",
    "protected_ls",
    "tp_uic_stage1",
    "tp_uic_full",
    "perfect_channel",
)

# Display order for the single-target table keeps the same reading.
DISPLAY = {
    "no_ic": "no-IC",
    "fixed_kappa": "fixed 40 dB",
    "plain_ls": "plain LS",
    "ridge_ls": "ridge LS",
    "protected_ls": "protected LS",
    "tp_uic_stage1": "TP-UIC stage-1",
    "tp_uic_full": "TP-UIC full",
    "perfect_channel": "perfect channel",
}


def _read(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    v = row.get(key, "")
    if v in ("", "nan", "None"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _i(row, key, default=0):
    v = row.get(key, "")
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _med(values):
    vals = [v for v in values if v == v]  # drop nan
    return statistics.median(vals) if vals else float("nan")


def _fmt(x, nd=4):
    if x != x:
        return "--"
    if x != 0.0 and abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _auc(h1, h0):
    """Threshold-free separability: P(T_H1 > T_H0) over all cross pairs.

    Reported because the *analytic* CFAR level is not calibrated in the masked
    regime (see the runs) and the H0 pool is small.  AUC needs neither a
    threshold nor a covariance model: 0.5 means the arm cannot tell the two
    hypotheses apart at all, 1.0 means it always can.
    """
    a = [v for v in h1 if v == v]
    b = [v for v in h0 if v == v]
    if not a or not b:
        return float("nan")
    wins = 0.0
    for x in a:
        for y in b:
            wins += 1.0 if x > y else (0.5 if x == y else 0.0)
    return wins / (len(a) * len(b))


# --------------------------------------------------------------------------
def table_arms(rows):
    """Matched H1/H0 GLRT in the full multi-target scenario."""
    by_arm = OrderedDict((a, []) for a in ARMS)
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    lines = [
        "| arm | C_IC (dB) | eta_surv | rho_w | T_H1 | thr | P_D | P_FA | AUC | ncp_best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in by_arm:
        rs = by_arm[arm]
        if not rs:
            continue
        t1 = [_f(r, "t_h1") for r in rs]
        t0 = [_f(r, "t_h0") for r in rs]
        thr = [_f(r, "thr_h1") for r in rs]
        # detection columns are already in the belief/truth dictionary used
        det_h1 = [_i(r, "det_h1") for r in rs]
        det_h0 = [_i(r, "det_h0") for r in rs]
        p_d = sum(det_h1) / len(det_h1) if det_h1 else float("nan")
        p_fa = sum(det_h0) / len(det_h0) if det_h0 else float("nan")
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                DISPLAY.get(arm, arm),
                _fmt(_med([_f(r, "kappa_db") for r in rs]), 2),
                _fmt(_med([_f(r, "eta_survive") for r in rs]), 3),
                _fmt(_med([_f(r, "rho_weighted") for r in rs]), 4),
                _fmt(_med(t1), 2),
                _fmt(_med(thr), 2),
                _fmt(p_d, 3),
                _fmt(p_fa, 3),
                _fmt(_auc(t1, t0), 3),
                _fmt(_med([_f(r, "ncp_best") for r in rs]), 3),
            )
        )
    return "\n".join(lines)


def table_single(rows):
    """Single-target mechanism validation: the canceller is not the bottleneck."""
    by_arm = OrderedDict((a, []) for a in ARMS)
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    lines = [
        "| arm | C_IC (dB) | eta_surv | rho_w | T_H1 | thr | P_D | ncp_best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in by_arm:
        rs = by_arm[arm]
        if not rs:
            continue
        det = [_i(r, "detected") for r in rs]
        p_d = sum(det) / len(det) if det else float("nan")
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                DISPLAY.get(arm, arm),
                _fmt(_med([_f(r, "kappa_db") for r in rs]), 2),
                _fmt(_med([_f(r, "eta_survive") for r in rs]), 3),
                _fmt(_med([_f(r, "rho_weighted") for r in rs]), 4),
                _fmt(_med([_f(r, "t_h1") for r in rs]), 2),
                _fmt(_med([_f(r, "threshold") for r in rs]), 2),
                _fmt(p_d, 3),
                _fmt(_med([_f(r, "ncp_best") for r in rs]), 3),
            )
        )
    return "\n".join(lines)


def table_masking(rows):
    """rho_weighted versus the number of nearest co-located targets taken as nuisance."""
    counts = sorted({_i(r, "n_nuisance_targets") for r in rows})
    by_arm = OrderedDict((a, []) for a in ARMS)
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    header = "| arm | " + " | ".join("n=%d" % n for n in counts) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in counts) + " |"
    lines = [header, sep]
    for arm in by_arm:
        rs = by_arm[arm]
        if not rs:
            continue
        cells = []
        for n in counts:
            vals = [_f(r, "rho_weighted") for r in rs if _i(r, "n_nuisance_targets") == n]
            cells.append(_fmt(_med(vals), 3))
        lines.append("| %s | %s |" % (DISPLAY.get(arm, arm), " | ".join(cells)))
    return "\n".join(lines)


def table_ident(rows):
    """On-grid / manifold / off-grid identifiability."""
    by_arm = OrderedDict((a, []) for a in ARMS)
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    lines = [
        "| arm | rho_grid | rho_grid_min | rho_manifold | rho_manifold_min | xi_rel_grid | off_grid_median | off_grid_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in by_arm:
        rs = by_arm[arm]
        if not rs:
            continue
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                DISPLAY.get(arm, arm),
                _fmt(_med([_f(r, "rho_grid_median") for r in rs]), 5),
                _fmt(_med([_f(r, "rho_grid_min") for r in rs]), 5),
                _fmt(_med([_f(r, "rho_manifold_median") for r in rs]), 5),
                _fmt(_med([_f(r, "rho_manifold_min") for r in rs]), 5),
                _fmt(_med([_f(r, "xi_rel_grid") for r in rs]), 5),
                _fmt(_med([_f(r, "off_grid_median") for r in rs]), 5),
                _fmt(_med([_f(r, "off_grid_max") for r in rs]), 5),
            )
        )
    return "\n".join(lines)


def offgrid_table(rows):
    """The off-grid delta columns, pooled per arm."""
    by_arm = OrderedDict((a, []) for a in ARMS)
    keys = []
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
        for k in r:
            if k.startswith("off_") and k not in keys:
                keys.append(k)
    keys = sorted(keys)
    if not keys:
        return ""
    header = "| arm | " + " | ".join(k.replace("off_", "").replace("_", ",") for k in keys) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in keys) + " |"
    lines = [header, sep]
    for arm in by_arm:
        rs = by_arm[arm]
        if not rs:
            continue
        cells = [_fmt(_med([_f(r, k) for r in rs]), 6) for k in keys]
        lines.append("| %s | %s |" % (DISPLAY.get(arm, arm), " | ".join(cells)))
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    out = Path(argv[0])
    arms = _read(out / "arms.csv")
    single = _read(out / "single_target.csv")
    masking = _read(out / "masking.csv")
    ident = _read(out / "identifiability.csv")

    if arms:
        print("## A. Matched H1/H0 GLRT in the full multi-target scenario\n")
        print(table_arms(arms))
        print()
    if single:
        print("## B. Single-target mechanism validation\n")
        print(table_single(single))
        print()
    if masking:
        print("## C. Masking curve (rho_weighted vs #nuisance)\n")
        print(table_masking(masking))
        print()
    if ident:
        print("## D. Identifiability audit\n")
        print(table_ident(ident))
        print()
        off = offgrid_table(ident)
        if off:
            print("### D2. Off-grid deltas (median rho)\n")
            print(off)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
