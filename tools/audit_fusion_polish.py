"""Paired, isolated V1 fixed-set fusion certification screen."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isac_sim.core.config import Config, apply_preset, apply_overrides
from experiments.flow.simulate import run_one_trial

METHOD = "proposed_c2f_adaptive_pd_fusion_polish"


def trial_job(job):
    condition, index, seed = job
    cfg = apply_preset(Config(), "target-local-v1")
    cfg.run.seed = seed
    overrides = {
        "nominal": {},
        "no_price": {"selector.lambda_c": 0.0},
        "belief_stress": {"prior.belief_sigma_pos_m": 500.0},
    }[condition]
    cfg = apply_overrides(cfg, overrides)
    results = run_one_trial(cfg, index, ["proposed_c2f_adaptive_pd", METHOD])
    old, new = results["proposed_c2f_adaptive_pd"], results[METHOD]
    cert = new.fusion_polish_certificate
    assert old.selected_links == new.selected_links
    assert np.all(cert.predicted_after >= cert.predicted_before)
    assert new.overhead_bits <= old.overhead_bits
    return {
        "condition": condition, "trial": index, "seed": seed,
        "pd_before": old.detected / old.total_targets,
        "pd_after": new.detected / new.total_targets,
        "pfa_before": old.false_alarm_overall / old.total_false_overall,
        "pfa_after": new.false_alarm_overall / new.total_false_overall,
        "reports_before": int(cert.reports_before.sum()),
        "reports_after": int(cert.reports_after.sum()),
        "report_lower_bound": int(cert.report_lower_bound.sum()),
        "already_at_count_bound": int(np.array_equal(cert.reports_before, cert.report_lower_bound)),
        "changed_targets": int(np.sum(old.reporting_plan.f_q != new.reporting_plan.f_q)),
        "candidate_pd_evaluations": cert.candidate_evaluations,
        "min_predicted_pd_change": float(np.min(cert.predicted_after-cert.predicted_before)),
        "fine_evaluations_before": old.fine_eval_c2f,
        "fine_evaluations_after": new.fine_eval_c2f,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--conditions", nargs="+", default=["nominal", "no_price"])
    parser.add_argument("--out", type=Path, default=Path("results_v1_fusion_polish_audit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, t, args.seed) for c in args.conditions for t in range(args.mc)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(trial_job, jobs))
    with (args.out / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for condition in args.conditions:
        group = [r for r in rows if r["condition"] == condition]
        stats = {key: float(np.mean([r[key] for r in group])) for key in rows[0] if key not in {"condition", "trial", "seed"}}
        diff = np.array([r["pd_after"] - r["pd_before"] for r in group])
        half = 1.96 * float(diff.std(ddof=1)) / np.sqrt(len(diff)) if len(diff)>1 else float("nan")
        stats["paired_pd_delta_ci95_normal"] = (
            [float(diff.mean()-half), float(diff.mean()+half)]
            if len(diff) > 1 and diff.std(ddof=1) > 0 else None
        )
        stats["paired_pd_delta_mean"] = float(diff.mean())
        stats["pd_difference_trials"] = int(np.count_nonzero(diff))
        stats["inference_note"] = "Exploratory paired normal interval; zero sample variance is not evidence of equivalence."
        stats["mc"] = len(group)
        stats["trials_with_report_reduction"] = sum(r["reports_after"] < r["reports_before"] for r in group)
        summary[condition] = stats
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol = dict(vars(args))
    cfg = apply_preset(Config(), "target-local-v1")
    cfg.run.seed = args.seed
    protocol["base_config"] = asdict(cfg)
    protocol["condition_overrides"] = {"nominal": {}, "no_price": {"selector.lambda_c": 0.0}, "belief_stress": {"prior.belief_sigma_pos_m": 500.0}}
    (args.out / "protocol.json").write_text(json.dumps(protocol, default=str, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
