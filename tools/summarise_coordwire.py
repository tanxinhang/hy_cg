"""Tabulate the coordinated-radiation wiring experiment.

Reads every ``results_coordwire_*/main/main.csv`` (one arm per directory) and
writes ``results_coordwire/summary.csv`` plus a paired comparison against the
``A_off`` baseline arm.

Discipline notes (these are the traps this script is written to avoid):

* Every arm shares the seed, the MC count, the preset and the method name, so
  the arms are paired by construction.  Differences are therefore reported as
  *paired* per-trial differences from ``trials.csv``, never as differences of
  two absolute summary numbers taken from different runs.
* ``B_gate`` is a regression guard, not an arm: it must be bit-identical to
  ``A_off``.  The equality check lives here so the claim is re-checked every
  time the table is rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM_DIR = ROOT / "results/coordwire"

ARM_LABEL = {
    "A_off": "baseline (release, no gate)",
    "B_gate": "gate only (regression guard)",
    "C_coord": "coordination",
    "H_coord_cap12": "coordination + max_tx_nodes=12",
    "D_coord_cap8": "coordination + max_tx_nodes=8",
    "G_coord_cap6": "coordination + max_tx_nodes=6",
    "F_coord_cap4": "coordination + max_tx_nodes=4",
    "J_coord_cap3": "coordination + max_tx_nodes=3",
    "K_coord_cap2": "coordination + max_tx_nodes=2",
    "I_coord_pen005": "coordination + tx_penalty=0.05",
    "L_coord_pen01": "coordination + tx_penalty=0.10",
    "M_coord_cap1": "coordination + max_tx_nodes=1",
    "N_s7777_off": "baseline (seed 7777)",
    "O_s7777_cap3": "coordination + max_tx_nodes=3 (seed 7777)",
}

# Arms are only comparable within one seed stream: the pairing must never mix
# seed 2026 with seed 7777 (the absolute numbers move by ~0.01 between seeds,
# which is exactly the size of the effect one would wrongly attribute to a knob).
BASELINE_BY_SEED = {
    "2026": "A_off",
    "7777": "N_s7777_off",
}


def arm_seed(tag: str) -> str:
    return "7777" if "s7777" in tag else "2026"

# Metrics that are read from main.csv.  Kept explicit so a renamed column is a
# loud KeyError rather than a silently-zero column.
METRIC_COLUMNS = {
    "P_D": "P_D",
    "P_FA": "P_FA",
    "bits": "B_mean_bits",
    "delay_ms": "T_mean_ms",
    "obs": "selected_observations_mean",
    "worst_pd": "actual_worst_target_P_D",
    "n_tx": "coordination_n_tx_mean",
    "rounds": "coordination_rounds_mean",
    "converged": "coordination_converged_rate",
    "evals": "selector_score_evaluations_mean",
}


def read_main(arm_dir: Path) -> dict[str, float]:
    path = arm_dir / "main" / "main.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    row = rows[0]
    out: dict[str, float] = {}
    for key, col in METRIC_COLUMNS.items():
        if col in row:
            try:
                out[key] = float(row[col])
            except (TypeError, ValueError):
                out[key] = math.nan
    return out


def read_trials(arm_dir: Path) -> list[dict[str, str]]:
    path = arm_dir / "main" / "trials.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def paired_pd_diff(base_rows, arm_rows):
    """Per-trial paired difference of per-target detection counts.

    Returns ``(delta, n_trials)`` where ``delta`` is the mean over trials of
    (detected_arm - detected_base) / total_targets, i.e. the paired change in
    P_D.  Reported with a normal-approximation 95% CI over trials.
    """
    if not base_rows or not arm_rows:
        return math.nan, math.nan, 0
    n = min(len(base_rows), len(arm_rows))
    diffs = []
    for i in range(n):
        try:
            db = float(base_rows[i]["detected"])
            da = float(arm_rows[i]["detected"])
            tot = float(base_rows[i]["total_targets"])
        except (KeyError, TypeError, ValueError):
            continue
        if tot <= 0:
            continue
        diffs.append((da - db) / tot)
    if not diffs:
        return math.nan, math.nan, 0
    mean = sum(diffs) / len(diffs)
    if len(diffs) > 1:
        var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        se = math.sqrt(var / len(diffs))
    else:
        se = math.nan
    return mean, 1.96 * se, len(diffs)


def bit_exact(a_rows, b_rows) -> tuple[int, int]:
    if not a_rows or not b_rows:
        return 0, 0
    compared = 0
    diffs = 0
    for ra, rb in zip(a_rows, b_rows):
        for key in ra:
            compared += 1
            if ra[key] != rb[key]:
                diffs += 1
    return compared, diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ARM_DIR / "summary.csv"))
    args = ap.parse_args()

    arms = sorted(
        p for p in ARM_DIR.parent.glob("results_coordwire_*") if (p / "main" / "main.csv").exists()
    )
    baseline_cache: dict[str, tuple[list[dict[str, str]], dict[str, float]]] = {}

    def baseline_for(seed: str):
        if seed not in baseline_cache:
            d = ARM_DIR.parent / f"results_coordwire_{BASELINE_BY_SEED[seed]}"
            baseline_cache[seed] = (read_trials(d), read_main(d))
        return baseline_cache[seed]

    rows = []
    for arm_dir in arms:
        tag = arm_dir.name[len("results_coordwire_"):]
        seed = arm_seed(tag)
        base_rows, base_metrics = baseline_for(seed)
        metrics = read_main(arm_dir)
        trial_rows = read_trials(arm_dir)
        # The fixed-point diagnostics live in trials.csv (main.csv is written
        # through an explicit column list and only gained these columns after
        # the first arms had already been produced).
        for key, col in (
            ("n_tx", "coordination_n_tx"),
            ("rounds", "coordination_rounds"),
            ("converged", "coordination_converged"),
        ):
            vals = [float(r[col]) for r in trial_rows if col in r and r[col] != ""]
            if vals:
                metrics[key] = sum(vals) / len(vals)
        delta, half_width, n_trials = paired_pd_diff(base_rows, trial_rows)
        rows.append({
            "seed": seed,
            "arm": tag,
            "label": ARM_LABEL.get(tag, tag),
            **{k: metrics.get(k, math.nan) for k in METRIC_COLUMNS},
            "paired_delta_P_D": delta,
            "paired_ci95_halfwidth": half_width,
            "n_paired_trials": n_trials,
        })

    ARM_DIR.mkdir(exist_ok=True)
    fieldnames = ["seed", "arm", "label"] + list(METRIC_COLUMNS) + [
        "paired_delta_P_D", "paired_ci95_halfwidth", "n_paired_trials",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"arms: {len(rows)}  -> {args.out}")
    bases = {BASELINE_BY_SEED[s] for s in BASELINE_BY_SEED}
    for seed in ("2026", "7777"):
        group = [r for r in rows if r["seed"] == seed]
        if not group:
            continue
        group.sort(key=lambda r: r["P_D"])
        print(f"\n--- seed {seed} (paired against {BASELINE_BY_SEED[seed]}) ---")
        hdr = (
            f"{'arm':16s} {'P_D':>7s} {'worst':>7s} {'P_FA':>6s} {'bits':>9s} "
            f"{'n_tx':>5s} {'conv':>5s} {'P_D/kbit':>8s} {'dP_D(paired)':>20s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in group:
            d = r["paired_delta_P_D"]
            if r["arm"] in bases:
                ds = "(baseline)"
            else:
                ds = f"{d:+.4f} +/- {r['paired_ci95_halfwidth']:.4f}"
            print(
                f"{r['arm']:16s} {r['P_D']:7.4f} {r['worst_pd']:7.4f} {r['P_FA']:6.4f} "
                f"{r['bits']:9.1f} {r['n_tx']:5.2f} {r['converged']:5.2f} "
                f"{r['P_D'] / (r['bits'] / 1000.0):8.4f} {ds:>20s}"
            )

    gate_dir = ARM_DIR.parent / "coordwire_B_gate"
    if gate_dir.exists():
        # Read A_off explicitly: ``base_rows`` is the *last* baseline bound in
        # the loop above, which is the seed-7777 one.
        compared, diffs = bit_exact(
            read_trials(ARM_DIR.parent / "results/coordwire_A_off"),
            read_trials(gate_dir),
        )
        print()
        print(
            f"regression guard (B_gate vs A_off): {compared} fields compared, "
            f"{diffs} differences -> {'BIT-EXACT' if diffs == 0 else 'REGRESSION'}"
        )


if __name__ == "__main__":
    main()
