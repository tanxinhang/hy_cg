"""Reproducible DD-waveform, report-mixture and selector-runtime audits.

These are component validations, not a rerun of the headline system experiment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isac_sim.scenario.belief import BeliefState
from isac_sim.core.config import Config, apply_preset, apply_overrides
from isac_sim.sensing.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.cooperation.reporting import assign_fusion_nodes
from experiments.selection import select_c2f, select_c2f_adaptive
from isac_sim.sensing.waveform import sweep_compare_analytic_vs_psf, waveform_llr_detection_check
from isac_sim.sensing.dd import leakage_1d


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results_isac_review_revision"))
    parser.add_argument("--runtime-trials", type=int, default=30)
    parser.add_argument("--error-model", choices=("erasure", "gaussian_replacement"), default="gaussian_replacement")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = apply_preset(Config(), "target-local-v1")
    cfg = apply_overrides(cfg, {"selector.score_mode": "detector_pd", "detect.comm_error_model": args.error_model})
    # The waveform helper represents a failed report by zero-mean replacement;
    # zero replacement variance implements exactly the released erasure law.
    wave_cfg = apply_overrides(cfg, {"detect.soft_error_sigma_scale": 0.0}) if args.error_model == "erasure" else cfg
    summary = {}
    capture = sweep_compare_analytic_vs_psf(cfg, 512, np.random.default_rng(913))
    rows = [{key: float(value[i]) for key, value in capture.items()} for i in range(512)]
    save_csv(args.out / "dd_energy.csv", rows)
    summary["dd"] = {
        f"{key}_{metric}": float(fun(np.abs(capture[key+"_analytic"]-capture[key+"_psf"])))
        for key in ("eta_c", "eta_loc")
        for metric, fun in (("max_abs_error", np.max), ("mean_abs_error", np.mean))
    }
    print("DD impulse-response audit complete", flush=True)

    wave_rows = []
    for chi in (1.0, 0.7, 0.3):
        for gamma in (0.5, 2.0):
            row = waveform_llr_detection_check(
                wave_cfg, raw_gamma=gamma, report_success=chi, n_trials=100_000,
                rng=np.random.default_rng(914),
            )
            for field in ("pd", "pfa"):
                p = row["empirical_"+field]
                # Wilson interval for independent waveform trials.
                z, n = 1.96, row["n_trials"]
                centre = (p+z*z/(2*n))/(1+z*z/n)
                half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                row[field+"_ci_low"], row[field+"_ci_high"] = centre-half, centre+half
            wave_rows.append(row)
    save_csv(args.out / "waveform_llr.csv", wave_rows)
    summary["waveform"] = {
        "max_pd_prediction_error": max(abs(r["predicted_pd"]-r["empirical_pd"]) for r in wave_rows),
        "max_exact_mixture_pd_error": max(abs(r["exact_mixture_pd"]-r["empirical_pd"]) for r in wave_rows),
        "max_exact_mixture_pfa_error": max(abs(r["exact_mixture_pfa"]-r["empirical_pfa"]) for r in wave_rows),
        "cases": len(wave_rows), "trials_per_case": 100000,
    }
    print("Waveform-derived LLR audit complete", flush=True)

    mixture_rows = []
    for chi in (0.3, 0.7, 1.0):
        for failure_scale in (0.0, 1.0, 3.0):
            rng = np.random.default_rng(915)
            gamma, looks, n = 0.5, cfg.detect.n_looks, 200_000
            delta = looks*gamma**2/(1+gamma)
            v0, v1 = looks*gamma**2/(1+gamma)**2, looks*gamma**2
            failure_v = failure_scale**2*v0
            for h in (0, 1):
                local = gamma/(1+gamma)*(rng.gamma(looks, 1+gamma*h, n)-looks)
                mixed = np.where(rng.random(n)<chi, local, rng.normal(0, np.sqrt(failure_v), n))
                mean = chi*delta*h
                var = chi*(v1 if h else v0)+(1-chi)*failure_v+chi*(1-chi)*(delta*h)**2
                mixture_rows.append({"chi":chi,"failure_scale":failure_scale,"hypothesis":h,
                    "mean_exact":mean,"mean_empirical":float(mixed.mean()),
                    "variance_exact":var,"variance_empirical":float(mixed.var()),
                    "mean_error_in_standard_errors":float(abs(mixed.mean()-mean)/np.sqrt(var/n)),
                    "relative_variance_error":float(abs(mixed.var()-var)/var),"samples":n})
    save_csv(args.out / "mixture_moments.csv", mixture_rows)
    summary["mixture"] = {"max_mean_error_in_standard_errors":max(r["mean_error_in_standard_errors"] for r in mixture_rows),
        "max_relative_variance_error":max(r["relative_variance_error"] for r in mixture_rows)}

    timing_rows = []
    for trial in range(args.runtime_trials+1):
        rng = np.random.default_rng([916,trial])
        truth = generate_geometry(cfg, rng)
        base_truth = build_base_gains(cfg, truth, rng)
        belief = BeliefState.from_truth(cfg, truth, rng).as_geometry(truth)
        base = build_base_gains(cfg, belief, rng, channel=base_truth, rcs_view="mean")
        tables = compute_link_tables(cfg, base)
        plan = assign_fusion_nodes(cfg, base, tables, belief)
        order = ("adaptive", "full") if trial % 2 else ("full", "adaptive")
        for method in order:
            start = time.perf_counter()
            if method == "adaptive":
                selected, _, stats = select_c2f_adaptive(cfg, base, tables, plan)
            else:
                selected, _, stats = select_c2f(cfg, base, tables, apply_to_all=True, plan=plan)
            elapsed = time.perf_counter()-start
            if trial:
                timing_rows.append({"trial":trial,"method":method,"seconds":elapsed,
                    "fine_evaluations":stats["fine_eval_c2f"],
                    "observations":sum(map(len, selected.values()))})
    save_csv(args.out / "selector_runtime.csv", timing_rows)
    summary["runtime"] = {method:{
        "median_seconds":float(np.median([r["seconds"] for r in timing_rows if r["method"]==method])),
        "mean_seconds":float(np.mean([r["seconds"] for r in timing_rows if r["method"]==method])),
        "mean_fine_evaluations":float(np.mean([r["fine_evaluations"] for r in timing_rows if r["method"]==method]))
    } for method in ("adaptive","full")}
    summary["runtime"]["scope"] = "Selector calls only; shared geometry/base tables and precomputed eta_fine excluded; one warmup; alternating order."
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    protocol={"config":asdict(cfg),"waveform_config":asdict(wave_cfg),"python":sys.version,"numpy":np.__version__,
        "platform":platform.platform(),"cpu":os.environ.get("PROCESSOR_IDENTIFIER",platform.processor()),
        "runtime_trials":args.runtime_trials,"dd_samples":512,"seed_dd":913,"seed_waveform":914,
        "seed_mixture":915,"seed_runtime":916,
        "sources":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("isac_sim").glob("*.py")}}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
