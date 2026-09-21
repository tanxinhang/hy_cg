# RETIRED PREMISE (2026-09-20): this script swept / read the config field
# `interference.direct_cancellation_db` (kappa_dc).
# That field was DELETED: it asserted a fixed 40 dB direct-path cancellation with no
# receiver implementation behind it while propping up the whole SINR denominator.
# Direct-path cancellation is now only ever a MEASURED TP-UIC residual.  Running this
# script as-is will fail on the missing attribute -- kept as historical evidence only.
"""Reduce a TP-UIC arms_mc.csv into the markdown table used by TP_UIC_V1.md.

Usage:
    python tools/summarise_tpuic.py results_tp_uic_v1/arms_mc.csv

Reduction rules (kept identical for every run so two runs stay comparable):
  * ``C_IC``            -- median over trials of ``kappa_db`` (dB)
  * ``pred`` / ``cal``  -- median of ``kappa_pred_db`` and of its error
  * ``eta_prot``        -- median of the design-coverage fraction
  * ``eta_surv``        -- median of the realised weak-echo survival fraction
  * ``P_D`` / ``P_FA``  -- the run stores per-trial CFAR decisions, so these are
    the pooled detection / false-alarm rates over all (trial, target) pairs

Arms are emitted in the order the driver declares them, not alphabetically, so
the table reads top-down from the naive baselines to the full method.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import OrderedDict
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


def _f(row, key):
    v = row.get(key, "")
    if v in ("", "nan", "None"):
        return float("nan")
    return float(v)


def _median(values):
    vals = [v for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if not vals:
        return float("nan")
    return statistics.median(vals)


def _fmt(v, nd=2):
    if v != v:
        return "—"
    if v == float("inf"):
        return "inf"
    return "%.*f" % (nd, v)


def load(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else Path("results_tp_uic_v1/arms_mc.csv")
    rows = load(path)
    if not rows:
        print("empty: %s" % path)
        return 1

    order = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    for a in rows:
        if a["arm"] not in order:
            order.append(a["arm"])

    by_arm = OrderedDict((a, [r for r in rows if r["arm"] == a]) for a in order)

    print("source: %s   trials/arm: %s   receiver: %s"
          % (path, len(by_arm[order[0]]), rows[0].get("receiver", "?")))
    print()
    print("| arm | C_IC [dB] | pred [dB] | cal err [dB] | eta_prot | eta_surv |"
          " P_D | P_FA | noise enh [dB] | dim |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    for arm, rs in by_arm.items():
        c_ic = _median([_f(r, "kappa_db") for r in rs])
        pred = _median([_f(r, "kappa_pred_db") for r in rs])
        cal = _median([_f(r, "cal_err_db") for r in rs])
        ep = _median([_f(r, "eta_protect") for r in rs])
        es = _median([_f(r, "eta_survive") for r in rs])
        enh = _median([_f(r, "noise_enh_db") for r in rs])
        dim = _median([_f(r, "protect_dim") for r in rs])

        # Pooled CFAR rates: one H1 stat per trial, plus the weak-target id.
        h1, h0, weak = [], [], int(rs[0]["weak_target"])
        for r in rs:
            g = _f(r, "gate")
            h1.append(_f(r, "stat_h1") > g)
            if int(r["weak_target"]) == weak:
                h0.append(_f(r, "stat_h0") > g)
        p_d = sum(h1) / len(h1) if h1 else float("nan")
        p_fa = sum(h0) / len(h0) if h0 else float("nan")

        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (arm, _fmt(c_ic), _fmt(pred), _fmt(cal), "%.4f" % ep,
                 "%.4f" % es, "%.3f" % p_d, "%.3f" % p_fa, _fmt(enh, 1),
                 _fmt(dim, 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
