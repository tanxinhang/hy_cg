"""Diagnose capture inactivity and belief/truth mask-ranking agreement.

This is a failure-oriented probe for the semantic-reset experiment.  It
enumerates every non-empty illumination mask up to ``--max-tx-nodes`` and
compares the belief objective used for planning with an independent held-out
truth objective.  Capture discount is toggled without redundancy so its actual
decision effect is isolated.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.scenario.belief import (  # noqa: E402
    BeliefState,
    belief_capture_probability_lower_bound,
    belief_capture_sigma_points,
    belief_dd_std_bins,
)
from isac_sim.sensing.model import build_base_gains, generate_geometry  # noqa: E402
from tools.run_joint_tpuic_coordination import (  # noqa: E402
    build_config,
    evaluate_state,
)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _selected_code(selected) -> str:
    normalized = {
        str(q): [list(map(int, link)) for link in links]
        for q, links in selected.items()
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--max-tx-nodes", type=int, default=3)
    ap.add_argument("--out", default="results_tpuic_certificate_diagnostic")
    args = ap.parse_args(argv)
    if not 1 <= args.max_tx_nodes <= args.m:
        ap.error("--max-tx-nodes must lie in [1, M]")

    # Fields consumed by the shared configuration builder.
    defaults = {
        "rounds": 0,
        "max_protected_targets": args.q,
        "covariance_protection": False,
        "tpuic_arm": "tp_uic_full",
        "evaluation_tpuic_arm": "",
        "adaptive_risk_slack": 0.001,
        "belief_error_in_cres": False,
        "fusion_rule": "max_in_rate",
    }
    for key, value in defaults.items():
        setattr(args, key, value)
    cfg = build_config(args)

    rng = np.random.default_rng([cfg.run.seed, int(args.trial)])
    geom_true = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, geom_true, rng)
    belief = BeliefState.from_truth(cfg, geom_true, rng)
    geom_belief = belief.as_geometry(geom_true)
    base_belief = build_base_gains(
        cfg,
        geom_belief,
        rng,
        channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    belief_std = belief_dd_std_bins(cfg, geom_belief, belief)
    capture_probability = belief_capture_probability_lower_bound(cfg, belief_std)
    capture_sigma = belief_capture_sigma_points(cfg, geom_belief, belief)
    valid_capture = np.asarray(capture_probability, dtype=float)[base_belief.valid_dd]

    cache = {}
    rows = []
    for size in range(1, int(args.max_tx_nodes) + 1):
        for active in combinations(range(int(args.m)), size):
            mask = np.zeros(int(args.m), dtype=bool)
            mask[list(active)] = True
            states = {}
            for label, aware in (("baseline", False), ("capture", True)):
                states[label] = evaluate_state(
                    cfg,
                    trial=int(args.trial),
                    geom_true=geom_true,
                    geom_belief=geom_belief,
                    base_truth=base_truth,
                    base_belief=base_belief,
                    belief_dd_std=belief_std,
                    capture_probability=capture_probability,
                    capture_sigma_points=capture_sigma,
                    mask=mask,
                    receiver="tpuic",
                    capture_aware=aware,
                    redundancy_aware=False,
                    min_links=2,
                    capability_reps=1,
                    capability_source="predicted",
                    capability_retention_source="predicted_risk",
                    heldout_capability=True,
                    capability_cache=cache,
                    tpuic_arm="tp_uic_full",
                )
            baseline, capture = states["baseline"], states["capture"]
            rows.append({
                "trial": int(args.trial),
                "mask": json.dumps(list(active), separators=(",", ":")),
                "n_tx": int(size),
                "belief_objective": float(baseline.belief_objective),
                "truth_objective": float(baseline.truth_objective),
                "belief_worst_pd": float(np.min(baseline.belief_pd)),
                "truth_worst_pd": float(np.min(baseline.truth_pd)),
                "selected": _selected_code(baseline.selected),
                "capture_belief_objective": float(capture.belief_objective),
                "capture_truth_objective": float(capture.truth_objective),
                "capture_selected": _selected_code(capture.selected),
                "capture_changed_selection": bool(
                    _selected_code(baseline.selected)
                    != _selected_code(capture.selected)
                ),
            })

    belief_values = np.asarray([r["belief_objective"] for r in rows], dtype=float)
    truth_values = np.asarray([r["truth_objective"] for r in rows], dtype=float)
    best_belief = int(np.argmax(belief_values))
    best_truth = int(np.argmax(truth_values))
    reference = int(np.argmax([r["n_tx"] for r in rows]))
    belief_delta = belief_values - belief_values[reference]
    truth_delta = truth_values - truth_values[reference]
    nonzero = (np.abs(belief_delta) > 1e-12) | (np.abs(truth_delta) > 1e-12)
    sign_agreement = float(np.mean(
        np.sign(belief_delta[nonzero]) == np.sign(truth_delta[nonzero])
    )) if np.any(nonzero) else float("nan")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "mask_certificate_diagnostic.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "trial": int(args.trial),
        "n_masks": len(rows),
        "spearman_belief_truth": _spearman(belief_values, truth_values),
        "sign_agreement_vs_reference": sign_agreement,
        "reference_mask": rows[reference]["mask"],
        "best_belief_mask": rows[best_belief]["mask"],
        "best_truth_mask": rows[best_truth]["mask"],
        "best_masks_agree": best_belief == best_truth,
        "capture_changed_masks": int(sum(
            bool(r["capture_changed_selection"]) for r in rows
        )),
        "capture_probability": {
            "min": float(np.min(valid_capture)),
            "p10": float(np.quantile(valid_capture, 0.10)),
            "median": float(np.median(valid_capture)),
            "p90": float(np.quantile(valid_capture, 0.90)),
            "max": float(np.max(valid_capture)),
            "fraction_below_0_9": float(np.mean(valid_capture < 0.9)),
            "fraction_below_0_5": float(np.mean(valid_capture < 0.5)),
        },
    }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote", os.path.abspath(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
