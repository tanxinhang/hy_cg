"""Move out-of-band / hardware-gain artifacts into an archive directory.

The project accumulated 183 ``results*`` artifacts spanning several abandoned
scenarios.  Two families are no longer admissible:

  A. any run whose radar hardware gain G_hw != 0 dB and >= 15 dB
  B. any run whose footprint is > 800 m OR whose target RCS is > 1 m^2

This script NEVER deletes: it moves to ``_archive/<stamp>/`` so the evidence is
recoverable.  Run with ``--dry-run`` (default) to inspect the plan first.

Manual overrides encode findings that no automated probe can recover (the
artifact carries no config, or its real parameters only appear in a .log).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_CSV = ROOT / "tools" / "_cleanup_scan.csv"

# --------------------------------------------------------------------------
# Manual overrides.  "KEEP" wins over everything (protects new-scenario
# evidence); "DELETE" adds artifacts the probe could not classify.
# reason strings are copied into the manifest for auditability.
# --------------------------------------------------------------------------
OVERRIDE_KEEP = {
    # --- new-scenario core evidence, G_hw = 0 dB -------------------------
    "results/coord_600m_g0k60": "G_hw=0 dB control arm (kappa=60)",
    "results_coord_800m_g0k60": "G_hw=0 dB control arm (kappa=60)",
    "results/coordination_600m_rcs0.1": "0.9500 coordination result, G_hw=0",
    "results_coordination_800m_rcs0.05": "coordination result, G_hw=0",
    "results_coordination_smoke": "coordination smoke, G_hw=0",
    "results_hw_vs_algo": "600 m/0.1 hw-vs-algo matrix summary",
    "results_hw_vs_algo_800": "800 m/0.05 hw-vs-algo matrix summary",
    "results/lowrcs_main_sparse": "negative control (gate off), MC=1000",
    "results/lowrcs_main_gate": "release-calibre gate arm, MC=1000",
    "results_lowrcs_main_kappa60": "release kappa sweep, MC=1000",
    "results_lowrcs_main_kappa80": "release kappa sweep, MC=1000",
    "results/lowrcs_500_800_main": "release main comparison, 600 m/0.1",
    "results_lowrcs_kappa600_kcurve.png": "figure",
    "results_release_kappa600.png": "figure",
    # --- in-band, G_hw = 0 ------------------------------------------------
    "results_radar_compact_800m_pilot.csv": "800 m compact, net_gain=0 only",
    "results_report_budget_800m_coarse.csv": "800 m compact, net_gain=0 only",
    "results_report_budget_800m_k10.csv": "800 m compact, net_gain=0 only",
    "results_v1_power_c2f_benchmark": "RCS 0.05/0.1/0.2, no hardware gain",
    # empty run (header-only trials.csv) from the same sweep tool as
    # results_v1_lowrcs_500_800; keep so the sweep history stays complete.
    "results_v1_lowrcs_sweep": "empty sweep stub, same tool as the in-band runs",
    "results_v1_rho_coordination": "400/600 m, RCS 0.05/0.2, G_hw=0",
    "results_v1_lowrcs_shortfall": "400 m, RCS 0.05/0.1; 'required_gain' is a "
                                   "measured shortfall, not an applied gain",
    # --- unclassifiable / empty: keep, the cost of a wrong delete is high --
    "results_v1_joint_small": "no geometry in artifacts -> keep",
    "results_v1_convergence": "no geometry in artifacts -> keep",
    "results_v1_move_geometry": "no geometry in artifacts -> keep",
    "results_v1_convergence.log": "log of a kept run",
    "results_v1_lowrcs_cheap.log": "log of a kept run",
    "results_v1_lowrcs_shortfall.log": "log of a kept run",
    "results_v1_lowrcs_sweep.log": "log of a kept run",
    "results_v1_rho_coordination.log": "log of a kept run",
    "results_v1_rho_coordination_smoke": "smoke of a kept run",
    "results_v1_rho_pd_check.log": "log of a kept run",
    "results_active_information_v15_system_smoke": "smoke, 800 m/0.05 family",
    "results_active_system_v15_smoke": "smoke, 800 m/0.05 family",
    "results_joint_bundle_v12_smoke": "smoke, 800 m/0.05 family",
    "results_scientific_gates_v16": "no geometry in artifacts -> keep",
}

OVERRIDE_DELETE = {
    # s2 preset is 2000 m AND sweeps net_gain up to 30 dB: hits both A and B.
    "results_radar_gain_s2_coarse.csv": "B(2000 m) + A(net_gain 10/20/30 dB)",
    "results_radar_gain_s2_refined.csv": "B(2000 m) + A(net_gain 12.5/15/17.5 dB)",
}

# A-class arms below this are retained as "hardware can be dialled down"
# evidence rather than deleted.
A_DELETE_MIN_DB = 15.0

A_KEEP_NOTE = {
    "results/coord_600m_g5k40": "5 dB down-shift control (< 15 dB threshold)",
}


def measure_band(d: Path):
    """Read the ACTUAL swept values from a trials CSV, if it has them.

    Sweep tools snapshot ``base_config`` with the library default RCS (50 m^2)
    even when every real trial used 0.05-0.2 m^2, so a config-only probe
    mis-files them as out-of-band.  Measured columns always win.
    """
    areas, rcss = set(), set()
    for csv_path in sorted(d.rglob("*.csv"))[:8]:
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                if not rd.fieldnames:
                    continue
                acol = next((c for c in rd.fieldnames
                             if c.lower() in ("area_m", "area_xy", "area")), None)
                rcol = next((c for c in rd.fieldnames
                             if c.lower() in ("rcs_m2", "target_rcs", "rcs")), None)
                if not acol and not rcol:
                    continue
                for i, row in enumerate(rd):
                    if i > 20000:
                        break
                    if acol and row.get(acol):
                        try:
                            areas.add(float(row[acol]))
                        except ValueError:
                            pass
                    if rcol and row.get(rcol):
                        try:
                            rcss.add(float(row[rcol]))
                        except ValueError:
                            pass
        except Exception:
            continue
    return areas, rcss


def load_plan():
    rows = list(csv.DictReader(open(SCAN_CSV, encoding="utf-8")))
    plan = []
    for r in rows:
        name = r["path"]
        cat, reason = r["category"], r["reasons"]
        # Measured swept range overrides the config-based probe.  A run whose
        # every trial is inside 500-800 m x RCS<=1 is never out-of-band, no
        # matter what its base_config snapshot claims.
        p = ROOT / name
        if p.is_dir():
            areas, rcss = measure_band(p)
            if (areas or rcss) and not any(a > 800 for a in areas) \
                    and not any(c > 1 for c in rcss):
                if cat == "B-out-of-band":
                    plan.append((name, "KEEP",
                                 f"measured band area={sorted(areas)} "
                                 f"rcs={sorted(rcss)} (base_config rcs=50 is "
                                 f"a stale snapshot)"))
                    continue
        if name in OVERRIDE_KEEP:
            plan.append((name, "KEEP", "override: " + OVERRIDE_KEEP[name]))
            continue
        if name in OVERRIDE_DELETE:
            plan.append((name, "DELETE", OVERRIDE_DELETE[name]))
            continue
        if name in A_KEEP_NOTE:
            plan.append((name, "KEEP", "A-class but " + A_KEEP_NOTE[name]))
            continue
        if cat == "A-hardware-gain":
            plan.append((name, "DELETE", "A: " + reason))
        elif cat == "B-out-of-band":
            plan.append((name, "DELETE", "B: " + reason))
        elif cat == "C-keep":
            plan.append((name, "KEEP", "in-band, G_hw=0"))
        else:
            plan.append((name, "KEEP", "unclassified -> keep by default"))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually move; otherwise print the plan")
    ap.add_argument("--archive", default=None,
                    help="archive root (default _archive/<YYYY-MM-DD>)")
    args = ap.parse_args()

    plan = load_plan()
    dels = [p for p in plan if p[1] == "DELETE"]
    keeps = [p for p in plan if p[1] == "KEEP"]

    print(f"plan: {len(plan)} artifacts -> DELETE {len(dels)}, KEEP {len(keeps)}")
    print()

    if args.apply:
        arch = Path(args.archive) if args.archive else \
            ROOT / "_archive" / datetime.now().strftime("%Y-%m-%d")
        arch.mkdir(parents=True, exist_ok=True)
        manifest = []
        moved = 0
        for name, _, reason in dels:
            src = ROOT / name
            if not src.exists():
                print(f"  MISSING {name}")
                continue
            dst = arch / name
            if dst.exists():
                print(f"  EXISTS  {name} (skipped)")
                continue
            # copy first, verify, only then remove the source
            if src.is_dir():
                shutil.copytree(src, dst)
                n = sum(1 for _ in dst.rglob("*") if _.is_file())
                s = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
                if n == 0:
                    print(f"  EMPTY   {name} (kept in place)")
                    shutil.rmtree(dst)
                    continue
                shutil.rmtree(src)
            else:
                shutil.copy2(src, dst)
                if dst.stat().st_size != src.stat().st_size:
                    print(f"  SIZE-MISMATCH {name} (kept in place)")
                    dst.unlink()
                    continue
                s = dst.stat().st_size
                src.unlink()
            moved += 1
            manifest.append({"path": name, "reason": reason, "bytes": s})
        mf = arch / "MANIFEST.csv"
        with open(mf, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "reason", "bytes"])
            w.writeheader()
            w.writerows(manifest)
        total = sum(m["bytes"] for m in manifest)
        print(f"\nmoved {moved} artifacts ({total/1024/1024:.1f} MB) -> {arch}")
        print(f"manifest: {mf}")
        print(f"\nremaining results* at top level: "
              f"{len(list(ROOT.glob('results*')))}")
    else:
        print("===== DELETE =====")
        for name, _, reason in dels:
            print(f"  {name:58s} {reason[:62]}")
        print(f"\n===== KEEP ({len(keeps)}) =====")
        for name, _, reason in keeps:
            print(f"  {name:58s} {reason[:62]}")
        print("\n(dry run; re-run with --apply to move into _archive/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
