"""Cross-tool parity check and map table for the 500-800 m low-RCS scenario.

Two jobs
--------
1. **Parity.** The sweep's ``base`` cell at ``--areas 600`` must reproduce
   ``tools/audit_v1_rcs_600m.py``'s ``rcs*_k8`` + ``method=v1`` rows exactly:
   same preset, same overrides, same seed, same trial index.  If this fails the
   scenario numbers must not be published.
2. **Map.** One row per (area, rcs) with the three statistics that are easy to
   confuse:

   ``mean_target``    mean over trials, then over targets  (average performance)
   ``worst_target``   mean over trials of the per-trial min over targets
   ``MIN_worst``      min over trials of the per-trial min over targets
                      (the tail; the coordination line's statistic differs)

   plus the best software cell and the hardware reference cell.
"""
import argparse
import csv
import json
from pathlib import Path

SOFTWARE_CELLS = ["looks64", "looks128", "maxmin", "maxmin_looks64",
                  "maxmin_looks128"]
FIELDS = ["pd", "pfa", "reports", "observations"]


def per_target(row):
    return [float(v) for v in row["per_target"].split(";")]


def parity_check(directory, reference, rcs_values, tol=0.0):
    """Sweep ``base`` @600 m vs the released 600 m table, cell by cell."""
    sweep = {}
    for row in csv.DictReader((directory / "trials.csv").open(encoding="utf8")):
        if row["label"] == "base" and float(row["area_m"]) == 600.0:
            sweep[(float(row["rcs_m2"]), int(row["trial"]))] = row
    ref = {}
    for row in csv.DictReader((reference / "trials.csv").open(encoding="utf8")):
        if row["condition"].endswith("_k8") and row["method"] == "v1":
            ref[(row["condition"][3:-3], int(row["trial"]))] = row

    checked = mismatched = 0
    missing = []
    for rcs in rcs_values:
        key = f"{rcs:g}"
        trials = sorted(t for (r, t) in sweep if r == rcs)
        if not trials:
            print(f"  MISSING: no base rows at 600 m for rcs={key}")
            missing.append((rcs, None))
            continue
        for trial in range(trials[-1] + 1):
            a, b = sweep.get((rcs, trial)), ref.get((key, trial))
            if a is None or b is None:
                missing.append((rcs, trial))
                continue
            for field in FIELDS:
                checked += 1
                if abs(float(a[field]) - float(b[field])) > tol:
                    mismatched += 1
                    print(f"  MISMATCH rcs={rcs:g} trial={trial} {field}: "
                          f"{a[field]} vs {b[field]}")
    print(f"parity: {checked - mismatched}/{checked} fields identical"
          f" ({mismatched} mismatches, {len(missing)} missing pairs)")
    return mismatched == 0 and not missing


def map_table(directory, rcs_values, areas):
    rows = list(csv.DictReader((directory / "trials.csv").open(encoding="utf8")))
    groups = {}
    for row in rows:
        groups.setdefault((float(row["area_m"]), float(row["rcs_m2"]),
                           row["label"]), []).append(row)

    out = []
    for area in areas:
        for rcs in rcs_values:
            entry = {"area_m": area, "rcs_m2": rcs}
            cells = {}
            for label in set(k[2] for k in groups if k[0] == area and k[1] == rcs):
                group = groups[(area, rcs, label)]
                targets = [per_target(r) for r in group]
                n = len(targets)
                cells[label] = {
                    "n": n,
                    "mean_target": sum(sum(t) / len(t) for t in targets) / n,
                    "worst_target": sum(min(t) for t in targets) / n,
                    "min_worst": min(min(t) for t in targets),
                    "pfa": sum(float(r["pfa"]) for r in group) / n,
                }
            entry["cells"] = cells
            if "base" in cells:
                software = [(v["mean_target"], k) for k, v in cells.items()
                            if k in SOFTWARE_CELLS]
                if software:
                    best = max(software)
                    entry["best_software"] = best[1]
                    entry["best_software_gain"] = (
                        best[0] - cells["base"]["mean_target"])
                if "gain15" in cells:
                    entry["gain15_gain"] = (cells["gain15"]["mean_target"]
                                            - cells["base"]["mean_target"])
            out.append(entry)

    header = (f"| RCS (m²) | 距离 (m) | base 平均 | base 最差目标 | base MIN "
              f"| 最好软件旋钮 | Δ平均 | +15 dB 硬件 | Δ平均 | P_FA |")
    print(header)
    print("|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for entry in out:
        c = entry["cells"]
        if "base" not in c:
            print(f"| {entry['rcs_m2']:g} | {entry['area_m']:g} | (缺) |")
            continue
        base = c["base"]
        software = entry.get("best_software", "—")
        sdelta = entry.get("best_software_gain")
        gdelta = entry.get("gain15_gain")
        print(f"| {entry['rcs_m2']:g} | {entry['area_m']:g} "
              f"| {base['mean_target']:.4f} | {base['worst_target']:.4f} "
              f"| {base['min_worst']:.4f} | {software} "
              f"| {'—' if sdelta is None else f'{sdelta:+.4f}'} "
              f"| {c.get('gain15', {}).get('mean_target', float('nan')):.4f} "
              f"| {'—' if gdelta is None else f'{gdelta:+.4f}'} "
              f"| {base['pfa']:.4f} |")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path,
                        default=Path("results_v1_lowrcs_500_800"))
    parser.add_argument("--ref", type=Path, default=Path("results_v1_rcs_600m"))
    parser.add_argument("--areas", type=float, nargs="+",
                        default=[500.0, 600.0, 700.0, 800.0])
    parser.add_argument("--rcs", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--json-out", type=Path, default=None,
                        help="defaults to <dir>/map.json so that two scenarios "
                             "writing to different --dir never share one "
                             "summary file (overwriting each other)")
    args = parser.parse_args()
    if args.json_out is None:
        args.json_out = args.dir / "map.json"

    print("=== parity: sweep base @600 m vs results_v1_rcs_600m rcs*_k8 ===")
    protocol = json.loads((args.dir / "protocol.json").read_text(encoding="utf8"))
    if protocol.get("scenario", "paper-vertical") != "paper-vertical":
        # The released 600 m table shares the 4 km vertical geometry.  A sweep on
        # another scenario is a different physical configuration, so an exact
        # match is not expected and must not be reported as a failure.
        print(f"  SKIPPED: this sweep ran scenario="
              f"{protocol.get('scenario')!r}, the reference table is "
              f"'paper-vertical'.  Compare at config level instead.")
        ok = None
    else:
        ok = parity_check(args.dir, args.ref, args.rcs)
    print()
    print("=== map: 500-800 m x RCS 0.05-0.2, seed 10917, MC=100 ===")
    out = map_table(args.dir, args.rcs, args.areas)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf8")
    print()
    print("parity_ok =", ok, "->", args.json_out)


if __name__ == "__main__":
    main()
