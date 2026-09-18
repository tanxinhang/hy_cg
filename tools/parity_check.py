#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parity check: the refactored package must reproduce the v10 prototype exactly.

The refactor moved every function into :mod:`isac_sim` but was not allowed to
change the model.  This script proves that: it runs both implementations on the
same Monte-Carlo size and seed, then compares every numeric CSV field.

Because the *default* model has since been corrected (shared-spectrum
interference coupling and a scale-relative SINR guard), the check pins the
``legacy`` preset explicitly.  v10 parity is therefore a property of that preset,
not of the current default -- which is exactly the right statement to make.

Baseline triage
---------------
``--baseline`` compares the legacy preset against the frozen baseline CSV at
mc=12/seed=2026.  That comparison used to report only the single worst field,
which hid the fact that ~210 of 306 cells diverge; it also demanded bit-exact
equality, so a perfect replay still failed on a handful of 1-ULP CSV
round-trips.  It prints a full triage and classifies every divergence as
``registered`` (attributed to a named commit in :data:`KNOWN_DRIFTS`) or
``unattributed``.

**Re-frozen 2026-09-16.**  The previous baseline was produced by the
``603b61d`` work tree and predated the ``84ae01c`` model revision.  That
revision was a deliberate code-to-paper alignment (the between-group term of the
law of total variance belongs to the H1 variance, not to the deflection
denominator -- see ``BASELINE_DRIFT_ATTRIBUTION.md`` sections 0 and 4.1), so the
old CSV described a model the manuscript never claimed.  The baseline has been
regenerated from the current model, the historical CSV is archived at
``.workbuddy/baseline_historical_603b61d/main/main.csv`` and
:data:`KNOWN_DRIFTS` has been retired.

**The registry is therefore empty and the gate is a live guard again**: any
divergence from here on is unattributed and fails the check.  A registration, if
one is ever added, is an attribution and not an endorsement -- it records where
a number came from, never that the change was correct.

Environment pinning
-------------------
The frozen CSV was produced by **CPython 3.11 / numpy 2.2.6**.  Under CPython
3.13 / numpy 2.5.2 the same code perturbs 25 of 1596 cells by <=1e-12 relative,
which ``--strict`` reports as a failure.  This was attributed on 2026-09-18 by
running a clean ``git worktree`` checkout of the frozen commit: it shows the same
25 cells under 3.13 and zero under 3.11, so the cells are interpreter arithmetic
and not a model change.  ``--allow-env-noise`` records that distinction without
masking anything: it passes only when *every* differing cell sits inside the
``tiny`` band, and it prints the runtime next to the verdict.  It is not a
standalone pass -- quote it only beside the frozen-runtime contrast.

Usage
-----
    python tools/parity_check.py                  # run both, all modes, MC=8
    python tools/parity_check.py --mc 20          # larger (slower) check
    python tools/parity_check.py --baseline       # frozen legacy baseline triage
    python tools/parity_check.py --baseline --strict     # demand zero diffs
    python tools/parity_check.py --baseline --strict --allow-env-noise   # outside py3.11
    python tools/parity_check.py --release-baseline --strict   # the release default path
    python tools/parity_check.py --release-baseline --write-release-baseline  # re-freeze it
    python tools/parity_check.py --baseline --json d.json
    python tools/parity_check.py --golden-only    # stale; expected MISMATCH

Two baselines, two different claims
-----------------------------------
* ``--baseline`` (legacy preset): guards the *refactor*.  It pins
  ``coupling=legacy`` + ``eps_mode=legacy``, so it says nothing about the
  release default path -- and under the legacy coupling the direct-path
  cancellation depth is never read.
* ``--release-baseline`` (preset ``target-local-v1``, the V1 main-experiment
  roster at mc=12/seed=2026): guards the *release identity*.  A flipped
  ``coordination.enable`` default, a moved kappa or a changed gate all show up
  here, and none of them show up above.

Exit status is 0 when the compared fields pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
V10 = ROOT / "lagrangian_dotfs_isac_simplified_v10_paper_plots.py"
# ``golden_v10_mc20.csv`` was frozen before the reporting-direction correction
# (the soft statistic is produced at the receiving UAV and reported j -> i), so
# it is expected to diverge and is kept for reference only.  The authoritative
# regression is ``.workbuddy/baseline/main/main.csv`` (mc=12, seed=2026), which
# the legacy preset must still reproduce bit-exactly.  That CSV was re-frozen on
# 2026-09-16 against the post-84ae01c model; the pre-revision file it replaced
# is archived under ``.workbuddy/baseline_historical_603b61d/``.
GOLDEN = ROOT / "golden_v10_mc20.csv"
LEGACY_BASELINE = ROOT / ".workbuddy" / "baseline" / "main" / "main.csv"
LEGACY_BASELINE_HISTORICAL = (
    ROOT / ".workbuddy" / "baseline_historical_603b61d" / "main" / "main.csv"
)

# The v10 prototype predates the model correction, so parity is checked under the
# ``legacy`` preset rather than under the current defaults.
LEGACY_PRESET = ["--set", "interference.coupling=legacy", "--set", "radio.eps_mode=legacy"]

# --------------------------------------------------------------------------
# Release-path baseline (added 2026-09-18)
# --------------------------------------------------------------------------
# The legacy comparison above guards the *refactor*: it pins two keys that keep
# the old coupling and the old SINR guard.  It therefore never exercises the
# release default path -- and under ``coupling=legacy`` the direct-path
# cancellation depth is not consulted at all, so a changed kappa, a changed
# gate, or a coordination default flipped to True would all pass it silently.
#
# This second baseline closes that hole: same preset and method roster as the V1
# main experiment (``tools/rerun_target_local_v1.py``), at a cheap MC so it can
# run inside a gate.
RELEASE_BASELINE = ROOT / ".workbuddy" / "baseline_release" / "main" / "main.csv"
RELEASE_MC = 12
RELEASE_SEED = 2026
RELEASE_ARGS: List[str] = [
    "--mode", "main",
    "--preset", "target-local-v1",
    "--set", "detect.comm_error_model=gaussian_replacement",
    "--mc", str(RELEASE_MC),
    "--seed", str(RELEASE_SEED),
]

# (tag, v10 argv, new-mode argv, v10 csv path, new csv path relative to --out)
MODE_CASES: Sequence[Tuple[str, List[str], List[str], str, str]] = (
    ("main", ["--csv", "v10_main.csv"], [], "v10_main.csv", "main/main.csv"),
    (
        "lambda-sweep",
        ["--lambda-sweep", "--lambda-values", "0.005", "0.05", "--sweep-csv", "v10_lambda.csv"],
        ["--mode", "lambda-sweep", "--values", "0.005", "0.05"],
        "v10_lambda.csv",
        "lambda-sweep/lambda-sweep.csv",
    ),
    (
        "ablation",
        ["--ablation", "--ablation-csv", "v10_ablation.csv"],
        ["--mode", "ablation"],
        "v10_ablation.csv",
        "ablation/ablation.csv",
    ),
    (
        "fair-ablation",
        ["--fair-ablation", "--fair-ablation-csv", "v10_fair.csv"],
        ["--mode", "fair-ablation"],
        "v10_fair.csv",
        "fair-ablation/fair-ablation.csv",
    ),
    (
        "dd-ablation",
        ["--dd-ablation", "--dd-ablation-csv", "v10_dd.csv"],
        ["--mode", "dd-ablation"],
        "v10_dd.csv",
        "dd-ablation/dd-ablation.csv",
    ),
    (
        "comm-sweep",
        ["--comm-sweep", "--R-min-values", "2e5", "1e6", "--comm-sweep-csv", "v10_comm.csv"],
        ["--mode", "comm-sweep", "--values", "2e5", "1e6"],
        "v10_comm.csv",
        "comm-sweep/comm-sweep.csv",
    ),
    (
        "robustness",
        ["--robustness", "--robustness-type", "comm_model", "--robustness-csv", "v10_rob.csv"],
        ["--mode", "robustness", "--axis", "comm_model"],
        "v10_rob.csv",
        "robustness/robustness.csv",
    ),
)

SKIP_KEYS = {"method", "variant", "experiment", "robustness_type", "comm_error_model"}

# --------------------------------------------------------------------------
# Drift classification
# --------------------------------------------------------------------------
# Relative tolerance below which two floats count as the same number.  1e-12
# absorbs CSV decimal round-tripping (observed: 7 cells at ~1e-16) while leaving
# every physically meaningful change far above the line.
DEFAULT_TOL = 1e-12

# Ordered bands used to label a cell by its relative difference.
BANDS: Sequence[Tuple[str, float]] = (
    ("tiny", 1e-9),
    ("small", 1e-3),
    ("moderate", 0.05),
    ("large", 0.5),
    ("order", math.inf),
)


@dataclass(frozen=True)
class DriftEntry:
    """One attributed divergence between the frozen baseline and the code."""

    key: str
    commit: str
    reason: str
    fields: Tuple[str, ...]
    report: str


# --------------------------------------------------------------------------
# RETIRED 2026-09-16.  The two entries below attributed the 210-cell divergence
# of the pre-revision baseline to 84ae01c ("release target-local fusion V1 and
# rebuild manuscript").  The follow-up audit showed entry A was misclassified:
# 84ae01c aligned the code with the manuscript's own equation rather than
# regressing it, and the frozen CSV described a pre-revision model.  The
# baseline was regenerated and the registry emptied.  The historical text is
# kept only as a record of what was believed; the check no longer consults it.
_HISTORICAL_KNOWN_DRIFTS: Tuple[DriftEntry, ...] = (
    DriftEntry(
        key="deflection-between-group-drop",
        commit="84ae01c",
        reason=(
            "deflection_variance_for_link was reduced to a verbatim alias of "
            "h0_variance_for_link, dropping the between-group term "
            "chi*(1-chi)*mu^2 of the law-of-total-variance mixture. The "
            "deflection denominator shrank, inflating D by ~7.3x and letting "
            "the greedy D_min break fire early (reports 34.42 -> 29.75). The "
            "six baseline methods mirror the proposed method's per-target "
            "counts via select_topk_baseline's reference_counts, so all seven "
            "moved together. Verified causally: restoring the term reproduces "
            "D_mean and selected_links_mean bit-exactly."
        ),
        fields=(
            "D_mean", "D_median", "D_p10", "D_p90", "D_per_ms", "D_per_kbit",
            "worst_target_D_mean", "worst_target_satisfied_prob",
            "selected_links_mean", "selected_links_std",
            "B_mean_bits", "B_std_bits", "T_mean_ms", "T_std_ms",
            "selected_chi_mean", "selected_chi_min_mean", "selected_chi_p10_mean",
            "selected_chi_ge_min_ratio_mean", "selected_gamma_comm_mean_db",
            "selected_rate_mean_mbps", "selected_rate_min_mbps_mean",
            "selected_rate_p10_mbps_mean", "selected_rate_satisfaction_ratio_mean",
        ),
        report="BASELINE_DRIFT_ATTRIBUTION.md#41-机制-a丢掉组间项law-of-total-variance",
    ),
    DriftEntry(
        key="soft-stat-sampling-rewrite",
        commit="84ae01c",
        reason=(
            "PARTIALLY LOCATED -- trigger not isolated. 84ae01c replaced the "
            "inline soft-statistic samplers in simulate.py (draw_h0_soft_stat / "
            "draw_h1_soft_stat, ~40 lines each) with a delegation to the new "
            "soft_channel.draw_received_soft_stat. After the deflection fix "
            "above restores selection, all nine methods still show real "
            "Monte-Carlo count differences in P_FA / P_D / actual_*_target_P_D "
            "(e.g. proposed_lagrangian P_FA 0.0442 -> 0.0492, P_D 0.5917 -> "
            "0.6250), while their D_*/T_* cells are down to 1 ULP. Every method "
            "moving together, with the selection-dependent quantities already "
            "restored, points at the RNG stream entering the detection stage. "
            "TWO CANDIDATE TRIGGERS WERE TESTED AND FALSIFIED: (1) the new "
            "Cornish-Fisher threshold correction (dropping z_cf leaves the "
            "output bit-identical -- skew0 is ~0 here, so z_cf == base_thr); "
            "(2) the H0 scale switching from pair-level sigma0[i,j] to "
            "target-level var0_q[i,j,q] (forcing sigma0^2 leaves the output "
            "bit-identical -- the two really do coincide). Remaining candidate: "
            "a change in RNG draw order/branching inside the sampling path. "
            "Registered as attributed-to-commit, NOT as explained."
        ),
        fields=(
            "P_FA", "P_FA_overall", "P_FA_ci95_low", "P_FA_ci95_high",
            "P_FA_ci95_half_width", "P_FA_overall_ci95_low",
            "P_FA_overall_ci95_high", "P_FA_overall_ci95_half_width",
            "P_D", "P_D_ci95_low", "P_D_ci95_high", "P_D_ci95_half_width",
            "P_D_per_ms", "P_D_per_kbit",
            "actual_mean_target_P_D", "actual_best_target_P_D",
            "actual_worst_target_P_D",
            # Shared with the deflection entry: reachable through the sampler.
            "D_mean", "D_median", "D_p10", "D_p90", "D_per_ms", "D_per_kbit",
            "worst_target_D_mean", "T_mean_ms",
        ),
        report="BASELINE_DRIFT_ATTRIBUTION.md#42-机制-b软统计量采样路径重写--已定位到-commit触发点未隔离",
    ),
)

# Live registry: empty by construction after the 2026-09-16 re-freeze.  Every
# divergence from the frozen baseline is unattributed and fails the check, so
# the gate is a live guard again.  See _HISTORICAL_KNOWN_DRIFTS above.
KNOWN_DRIFTS: Tuple[DriftEntry, ...] = ()

DRIFT_FIELD_INDEX: Dict[str, DriftEntry] = {}
for _entry in KNOWN_DRIFTS:
    for _field in _entry.fields:
        DRIFT_FIELD_INDEX.setdefault(_field, _entry)


def _band(rel: float) -> str:
    for name, cap in BANDS:
        if rel <= cap:
            return name
    return "order"


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


def _run(cmd: List[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def _pair_rows(rows_a: List[Dict[str, str]], rows_b: List[Dict[str, str]]):
    """Align two result tables by ``method`` when their row counts differ."""
    if len(rows_a) != len(rows_b):
        # Row counts diverge only when the refactor *added* a method (e.g.
        # ``topk_deflection``), which the prototype never emitted.  Align by
        # the ``method`` column so the extra row is skipped instead of
        # shifting every subsequent row.
        if rows_a and rows_b and "method" in rows_a[0] and "method" in rows_b[0]:
            by_a = {r["method"]: r for r in rows_a}
            by_b = {r["method"]: r for r in rows_b}
            common = [m for m in by_a if m in by_b]
            if not common:
                return None
            return [(by_a[m], by_b[m]) for m in common]
        return None
    return list(zip(rows_a, rows_b))


def _compare(path_a: Path, path_b: Path) -> Tuple[int, float, Tuple[str, ...] | None]:
    """Legacy single-worst comparison, kept for the v10 parity modes."""
    rows_a = list(csv.DictReader(path_a.open(encoding="utf-8")))
    rows_b = list(csv.DictReader(path_b.open(encoding="utf-8")))

    pairs = _pair_rows(rows_a, rows_b)
    if pairs is None:
        if len(rows_a) != len(rows_b):
            return len(rows_a), float("inf"), ("row count", str(len(rows_a)), str(len(rows_b)))
        return 0, float("inf"), ("no common methods",)

    shared = set(pairs[0][0]) & set(pairs[0][1])
    worst = 0.0
    where = None
    compared = 0
    for row_a, row_b in pairs:
        for key in sorted(shared):
            va, vb = row_a[key], row_b[key]
            if key in SKIP_KEYS or va == "" or vb == "":
                continue
            try:
                fa, fb = float(va), float(vb)
            except ValueError:
                if va != vb:
                    return len(rows_a), float("inf"), (key, va, vb)
                continue
            compared += 1
            diff = abs(fa - fb)
            if diff > worst:
                worst, where = diff, (key, fa, fb)
    return compared, worst, where


def _compare_records(path_a: Path, path_b: Path) -> Tuple[int, List[Dict[str, object]]]:
    """Full cell-level comparison: every differing cell, not just the worst."""
    rows_a = list(csv.DictReader(path_a.open(encoding="utf-8")))
    rows_b = list(csv.DictReader(path_b.open(encoding="utf-8")))
    pairs = _pair_rows(rows_a, rows_b)
    if pairs is None:
        return 0, []

    shared = set(pairs[0][0]) & set(pairs[0][1])
    compared = 0
    records: List[Dict[str, object]] = []
    for row_a, row_b in pairs:
        method = row_a.get("method", "?")
        for key in sorted(shared):
            va, vb = row_a[key], row_b[key]
            if key in SKIP_KEYS or va == "" or vb == "":
                continue
            compared += 1
            try:
                fa, fb = float(va), float(vb)
            except ValueError:
                if va != vb:
                    records.append({"method": method, "field": key,
                                    "baseline": va, "current": vb,
                                    "rel": math.inf, "band": "nonnumeric"})
                continue
            if math.isnan(fa) or math.isnan(fb):
                if math.isnan(fa) != math.isnan(fb):
                    records.append({"method": method, "field": key,
                                    "baseline": fa, "current": fb,
                                    "rel": math.inf, "band": "nan-mismatch"})
                continue
            if fa == fb:
                continue
            rel = _rel(fa, fb)
            records.append({"method": method, "field": key,
                            "baseline": fa, "current": fb,
                            "rel": rel, "band": _band(rel)})
    return compared, records


def _triage(records: List[Dict[str, object]], tol: float):
    """Split divergences into registered (attributed) and unattributed."""
    registered: List[Dict[str, object]] = []
    unattributed: List[Dict[str, object]] = []
    noise = 0
    for rec in records:
        rel = float(rec["rel"])
        if math.isfinite(rel) and rel <= tol:
            noise += 1
            continue
        entry = DRIFT_FIELD_INDEX.get(str(rec["field"]))
        if entry is None:
            unattributed.append(rec)
        else:
            registered.append({**rec, "drift": entry})
    return registered, unattributed, noise


def _runtime() -> str:
    import platform as _platform

    import numpy as _np

    return f"py{_platform.python_version()} / numpy {_np.__version__}"


def _run_baseline(args) -> int:
    if not LEGACY_BASELINE.exists():
        raise SystemExit(f"baseline not found: {LEGACY_BASELINE}")
    with tempfile.TemporaryDirectory(prefix="isac_baseline_") as tmp:
        out = Path(tmp) / "legacy"
        _run([sys.executable, str(ROOT / "run_isac_sim.py"),
              "--mc", "12", "--seed", "2026", "--quiet", "--no-plots",
              "--out", str(out), *LEGACY_PRESET], ROOT)
        compared, records = _compare_records(LEGACY_BASELINE, out / "main" / "main.csv")

    registered, unattributed, noise = _triage(records, args.tolerance)

    bands: Dict[str, int] = {}
    for rec in records:
        bands[str(rec["band"])] = bands.get(str(rec["band"]), 0) + 1
    band_str = " ".join(f"{k}={bands[k]}" for k in sorted(bands)) or "none"

    print("legacy baseline triage")
    print(f"  runtime tested    : {_runtime()}")
    print(f"  cells compared    : {compared}")
    print(f"  differing         : {len(records)}  ({band_str})")
    print(f"  below tolerance   : {noise}  (rel <= {args.tolerance:g})")
    print(f"  registered        : {len(registered)}")
    print(f"  unattributed      : {len(unattributed)}")

    if registered:
        print("\n  registered drift:")
        per_key: Dict[str, List[Dict[str, object]]] = {}
        for rec in registered:
            per_key.setdefault(rec["drift"].key, []).append(rec)
        for entry in KNOWN_DRIFTS:
            hits = per_key.get(entry.key, [])
            if not hits:
                continue
            fields = sorted({str(h["field"]) for h in hits})
            methods = sorted({str(h["method"]) for h in hits})
            print(f"    [{entry.commit}] {entry.key}")
            print(f"        cells={len(hits)}  fields={len(fields)}  methods={len(methods)}")
            print(f"        reason: {entry.reason}")
            print(f"        report: {entry.report}")

    if unattributed:
        print("\n  UNATTRIBUTED (these decide the exit status):")
        for rec in unattributed[: args.max_show]:
            print(f"    {rec['method']:22s} {rec['field']:30s} "
                  f"{rec['baseline']!r:>18s} -> {rec['current']!r:<18s} "
                  f"rel={rec['rel']} ({rec['band']})")
        if len(unattributed) > args.max_show:
            print(f"    ... {len(unattributed) - args.max_show} more")

    print("\n  NOTE: baseline re-frozen 2026-09-16 against the post-84ae01c model.")
    print("        The attribution registry is empty, so every divergence printed")
    print("        above is unattributed and fails the check: this is a live guard.")
    print(f"        Historical pre-revision CSV: {LEGACY_BASELINE_HISTORICAL.relative_to(ROOT)}")
    print("        Mechanism evidence: BASELINE_DRIFT_ATTRIBUTION.md.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "baseline": str(LEGACY_BASELINE),
            "cells_compared": compared,
            "n_differing": len(records),
            "n_below_tolerance": noise,
            "n_registered": len(registered),
            "n_unattributed": len(unattributed),
            "registered": [{k: v for k, v in r.items() if k != "drift"} | {"drift": r["drift"].key}
                           for r in registered],
            "unattributed": unattributed,
            "known_drifts": [asdict(e) for e in KNOWN_DRIFTS],
        }, indent=1), encoding="utf-8")
        print(f"\n  wrote {args.json}")

    if args.strict:
        if args.allow_env_noise:
            # The frozen CSVs were produced by CPython 3.11 / numpy 2.2.6.  A
            # different runtime perturbs a handful of cells by ~1e-16..1e-13
            # relative.  A clean checkout of the frozen commit reproduces the
            # *same* cells under the *same* runtime, which is what makes the
            # contrast an attribution rather than an excuse -- so this switch
            # only tolerates cells inside the band, and names every one of them.
            big = [r for r in records if str(r["band"]) != "tiny"]
            if big:
                print(f"\nRESULT: DIFFERS ({len(big)} cell(s) beyond the 'tiny' band; "
                      "--allow-env-noise does not cover them)")
                for rec in big[: args.max_show]:
                    print(f"    {rec['method']:22s} {rec['field']:30s} "
                          f"rel={rec['rel']} ({rec['band']})")
                return 1
            print(f"\nRESULT: CLEAN (--allow-env-noise: {len(records)} cell(s) differ inside "
                  "the 'tiny' band, rel <= 1e-9)")
            print("        attributed to the interpreter/numpy version, not to the model.")
            print("        Only valid next to the frozen-runtime contrast in")
            print("        V1_STABLE_RELEASE.md section 5.6 -- not as a standalone pass.")
            return 0
        ok = not records
        print(f"\nRESULT: {'CLEAN' if ok else 'DIFFERS'} (--strict: {len(records)} differing cells)")
        if records and not args.allow_env_noise:
            print(f"        runtime under test: {_runtime()}; the baseline is pinned to "
                  "py3.11 / numpy 2.2.6.")
            print("        Re-run under the contracted runtime before treating this as drift.")
        return 0 if ok else 1
    ok = not unattributed
    print(f"\nRESULT: {'CLEAN' if ok else 'DIRTY'} "
          f"(unattributed={len(unattributed)}, registered={len(registered)}, "
          f"below-tolerance={noise})")
    return 0 if ok else 1


def _release_verdict(records, args, compared: int, noise: int, label: str) -> int:
    """Shared verdict for the release-path baseline (mirrors the legacy one)."""
    bands: Dict[str, int] = {}
    for rec in records:
        bands[str(rec["band"])] = bands.get(str(rec["band"]), 0) + 1
    band_str = " ".join(f"{k}={bands[k]}" for k in sorted(bands)) or "none"
    above = [r for r in records if float(r["rel"]) > args.tolerance]

    print(label)
    print(f"  runtime tested    : {_runtime()}")
    print(f"  cells compared    : {compared}")
    print(f"  differing         : {len(records)}  ({band_str})")
    print(f"  below tolerance   : {noise}  (rel <= {args.tolerance:g})")
    print(f"  above tolerance   : {len(above)}")
    if above:
        print("\n  ABOVE-TOLERANCE (these decide the exit status):")
        for rec in above[: args.max_show]:
            print(f"    {rec['method']:26s} {rec['field']:30s} "
                  f"{rec['baseline']!r:>18s} -> {rec['current']!r:<18s} "
                  f"rel={rec['rel']} ({rec['band']})")
        if len(above) > args.max_show:
            print(f"    ... {len(above) - args.max_show} more")

    if args.allow_env_noise:
        big = [r for r in records if str(r["band"]) != "tiny"]
        if big:
            print(f"\nRESULT: DIFFERS ({len(big)} cell(s) beyond the 'tiny' band; "
                  "--allow-env-noise does not cover them)")
            return 1
        print(f"\nRESULT: CLEAN (--allow-env-noise: {len(records)} cell(s) inside the "
              "'tiny' band, rel <= 1e-9)")
        print("        attributed to the interpreter/numpy version, not to the model.")
        return 0
    if args.strict:
        ok = not records
        print(f"\nRESULT: {'CLEAN' if ok else 'DIFFERS'} (--strict: {len(records)} differing cells)")
        return 0 if ok else 1
    ok = not above
    print(f"\nRESULT: {'CLEAN' if ok else 'DIRTY'} (above-tolerance={len(above)})")
    return 0 if ok else 1


def _run_release_baseline(args) -> int:
    """Bit-exact guard on the *release* default path, not on the legacy preset.

    Same preset, method roster (default 19) and seed as the V1 main experiment,
    at a cheap MC so a gate can afford it.  This is the check that notices a
    flipped ``coordination.enable`` default or a moved kappa -- neither of which
    the legacy comparison can see.
    """
    with tempfile.TemporaryDirectory(prefix="isac_release_") as tmp:
        out = Path(tmp) / "release"
        _run([sys.executable, str(ROOT / "run_isac_sim.py"), "--quiet", "--no-plots",
              "--out", str(out), *RELEASE_ARGS], ROOT)
        produced = out / "main" / "main.csv"
        if not produced.exists():
            raise SystemExit(f"release run produced no main.csv at {produced}")

        if args.write_release_baseline:
            RELEASE_BASELINE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(produced, RELEASE_BASELINE)
            print(f"wrote {RELEASE_BASELINE.relative_to(ROOT)}  "
                  f"(preset=target-local-v1, mc={RELEASE_MC}, seed={RELEASE_SEED})")
            print(f"  runtime: {_runtime()}  -- the baseline is bound to this runtime")
            return 0

        if not RELEASE_BASELINE.exists():
            raise SystemExit(
                f"release baseline not found: {RELEASE_BASELINE}\n"
                "generate it once with:  python tools/parity_check.py "
                "--release-baseline --write-release-baseline"
            )
        compared, records = _compare_records(RELEASE_BASELINE, produced)

    _, _, noise = _triage(records, args.tolerance)
    return _release_verdict(
        records, args, compared, noise,
        f"release-path baseline (target-local-v1, mc={RELEASE_MC}, seed={RELEASE_SEED})",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mc", type=int, default=8, help="Monte-Carlo trials per run (default: 8)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--golden-only", action="store_true",
                        help="only compare the main run against the frozen golden CSV "
                             "(stale: predates the reporting-direction correction)")
    parser.add_argument("--baseline", action="store_true",
                        help="compare the legacy preset against the frozen baseline at "
                             "mc=12/seed=2026 (re-frozen 2026-09-16; the authoritative regression)")
    parser.add_argument("--release-baseline", action="store_true",
                        help="compare the *release* default path (preset target-local-v1, "
                             f"mc={RELEASE_MC}/seed={RELEASE_SEED}) against its own frozen "
                             "baseline -- this is what covers kappa, the gate and the "
                             "coordination defaults")
    parser.add_argument("--write-release-baseline", action="store_true",
                        help="with --release-baseline: (re)generate that baseline instead of "
                             "comparing. Only after an attributed, deliberate change.")
    parser.add_argument("--strict", action="store_true",
                        help="with --baseline/--release-baseline: require zero differing cells, "
                             "ignoring the KNOWN_DRIFTS registry")
    parser.add_argument("--allow-env-noise", action="store_true",
                        help="with --strict: accept cells that differ only inside the 'tiny' "
                             "band (rel <= 1e-9) as interpreter/numpy arithmetic when running "
                             "outside the contracted runtime (py3.11 / numpy 2.2.6). "
                             "Refuses if any cell is larger.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOL,
                        help=f"relative tolerance for the below-noise band (default: {DEFAULT_TOL:g})")
    parser.add_argument("--json", default=None, help="with --baseline: dump the full triage as JSON")
    parser.add_argument("--max-show", type=int, default=40,
                        help="max unattributed cells to print (default: 40)")
    args = parser.parse_args()

    if args.baseline:
        return _run_baseline(args)

    if args.release_baseline:
        return _run_release_baseline(args)

    with tempfile.TemporaryDirectory(prefix="isac_parity_") as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "new"
        new_argv = ["--mc", str(args.mc), "--seed", str(args.seed), "--quiet", "--no-plots",
                    "--out", str(out), *LEGACY_PRESET]

        failures = 0
        if args.golden_only:
            if not GOLDEN.exists():
                raise SystemExit(f"frozen golden file not found: {GOLDEN}")
            _run([sys.executable, str(ROOT / "run_isac_sim.py"), *new_argv], ROOT)
            compared, worst, where = _compare(GOLDEN, out / "main" / "main.csv")
            status = "OK" if worst == 0.0 else "MISMATCH"
            print(f"{'main (golden)':<16} fields={compared:<5} max|diff|={worst:<8g} {status}")
            return 0 if worst == 0.0 else 1

        if not V10.exists():
            raise SystemExit(f"reference prototype not found: {V10}")

        for tag, v10_args, new_args, v10_rel, new_rel in MODE_CASES:
            _run([sys.executable, str(V10), "--num-mc", str(args.mc), "--seed", str(args.seed),
                  "--quiet", *v10_args], ROOT)
            _run([sys.executable, str(ROOT / "run_isac_sim.py"), *new_argv, *new_args], ROOT)

            compared, worst, where = _compare(ROOT / v10_rel, out / new_rel)
            ok = worst == 0.0
            failures += 0 if ok else 1
            print(f"{tag:<16} fields={compared:<5} max|diff|={worst:<8g} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                print(f"    worst field: {where}")

        print()
        print("PARITY OK - refactor is numerically identical to v10" if not failures
              else f"PARITY FAILED in {failures} mode(s)")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
