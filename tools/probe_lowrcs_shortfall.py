"""Analytic sensing-SINR shortfall of the 400-600 m low-RCS operating point.

Why a separate probe
--------------------
``tools/audit_v1_lowrcs_screen`` measures what each *available knob* actually
buys by running the Monte-Carlo detector.  That answers "does it work", but not
"how far away are we".  This probe answers the second question in closed form
from the derived LLR moments in ``isac_sim.llr``:

    delta(gamma) = L * gamma^2 / (1 + gamma)      E1 - E0 mean gap
    var0(gamma)  = L * gamma^2 / (1 + gamma)^2    H0 variance
    var1(gamma)  = L * gamma^2                    H1 variance
    d' = (sum delta - z * sqrt(sum var0)) / sqrt(sum var1),  z = Q^-1(P_FA)

then solves for the single scalar gain ``k`` applied to *every* selected link's
sensing SINR that lifts a given target to the required detection probability.
10*log10(k) is the shortfall in dB: the amount of extra sensing SNR the current
configuration is missing, from whichever source (more integration looks, more
radar power, antenna gain, or a larger RCS).

Caveats that make this a bound, not a result
--------------------------------------------
* Links are treated as independent (diagonal fused covariance). ``corr.enable``
  adds off-diagonal terms, so the probe is optimistic about the fused d'.
* The released detector is Monte-Carlo and calibrated, not Gaussian; a Gaussian
  tail is an approximation at these low per-link SINRs.
* The scalar gain is applied uniformly. The power optimisation does **not** act
  uniformly, so this is a diagnostic of the residual physics, not a prediction
  of what the optimiser will find.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import ndtr  # noqa: E402
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config  # noqa: E402
from tools.audit_v1_lowrcs_sweep import REPORT_CAP, SEED  # noqa: E402
from isac_sim.belief import BeliefState, belief_dd_std_bins  # noqa: E402
from isac_sim.config import apply_overrides, validate_config  # noqa: E402
from isac_sim.llr import llr_delta, llr_var0, llr_var1  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.reporting import assign_fusion_nodes  # noqa: E402
from isac_sim.selection import select_c2f_adaptive  # noqa: E402

FIELDS = ["area_m", "rcs_m2", "trial", "n_looks", "target", "n_links",
          "gamma_mean_db", "gamma_min_db", "d_prime", "pd_pred",
          "required_gain_db"]


def _d_prime(gammas, n_looks, z):
    delta = float(sum(llr_delta(g, n_looks) for g in gammas))
    var0 = float(sum(llr_var0(g, n_looks) for g in gammas))
    var1 = float(sum(llr_var1(g, n_looks) for g in gammas))
    if var1 <= 0.0:
        return float("-inf")
    return (delta - z * np.sqrt(var0)) / np.sqrt(var1)


def _required_gain(gammas, n_looks, z, target_pd):
    """Smallest scalar multiplier on every gamma that reaches ``target_pd``."""
    want = norm.ppf(target_pd)
    if _d_prime(gammas, n_looks, z) >= want:
        return 1.0
    lo, hi = 1.0, 1.0
    for _ in range(60):
        hi *= 2.0
        if _d_prime([g * hi for g in gammas], n_looks, z) >= want:
            break
    else:
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _d_prime([g * mid for g in gammas], n_looks, z) >= want:
            hi = mid
        else:
            lo = mid
    return hi


def job(spec):
    area, rcs, trial, looks = spec
    cfg = apply_overrides(
        config(SEED, REPORT_CAP),
        {"geometry.area_xy": float(area), "detect.target_rcs": float(rcs),
         "detect.n_looks": int(looks)})
    validate_config(cfg)
    z = float(norm.ppf(1.0 - cfg.detect.Pfa_target))
    rng = np.random.default_rng([SEED, trial])
    truth = generate_geometry(cfg, rng)
    truth_base = build_base_gains(cfg, truth, rng)
    belief = BeliefState.from_truth(cfg, truth, rng)
    geom = belief.as_geometry(truth)
    base = build_base_gains(cfg, geom, rng, channel=truth_base,
                            rcs_view="mean")
    coarse = compute_link_tables(cfg, base)
    plan = assign_fusion_nodes(cfg, base, coarse, geom)
    chosen, _d, _stats = select_c2f_adaptive(cfg, base, coarse, plan)

    gamma = np.asarray(coarse.gamma_sense, dtype=float)
    rows = []
    for q in range(cfg.scale.Q):
        links = chosen.get(q, [])
        gammas = [float(gamma[i, j, q]) for (i, j) in links]
        if not gammas:
            rows.append(dict(area_m=float(area), rcs_m2=float(rcs),
                             trial=trial, n_looks=int(looks), target=q,
                             n_links=0, gamma_mean_db=float("nan"),
                             gamma_min_db=float("nan"),
                             d_prime=float("-inf"), pd_pred=0.0,
                             required_gain_db=float("inf")))
            continue
        dp = _d_prime(gammas, int(looks), z)
        gain = _required_gain(gammas, int(looks), z,
                              cfg.detect.weak_pd_required)
        rows.append(dict(
            area_m=float(area), rcs_m2=float(rcs), trial=trial,
            n_looks=int(looks), target=q, n_links=len(gammas),
            gamma_mean_db=float(10.0 * np.log10(np.mean(gammas))),
            gamma_min_db=float(10.0 * np.log10(np.min(gammas))),
            d_prime=float(dp),
            pd_pred=float(ndtr(dp)) if np.isfinite(dp) else 0.0,
            required_gain_db=float(10.0 * np.log10(gain))
            if np.isfinite(gain) else float("inf")))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=60)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--areas", type=float, nargs="+",
                        default=[400.0, 600.0])
    parser.add_argument("--rcs", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--looks", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--out", type=Path,
                        default=Path("results_v1_lowrcs_shortfall"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)

    specs = [(area, rcs, trial, looks)
             for area in args.areas for rcs in args.rcs
             for trial in range(args.mc) for looks in args.looks]
    rows = []
    with (args.out / "shortfall.csv").open("w", newline="",
                                           encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(job, spec) for spec in specs]
            for index, future in enumerate(as_completed(futures), 1):
                batch = future.result()
                rows.extend(batch)
                writer.writerows(batch)
                handle.flush()
                if index % 20 == 0:
                    print(f"{index}/{len(specs)}", flush=True)

    summary = {}
    for area in args.areas:
        for rcs in args.rcs:
            for looks in args.looks:
                group = [r for r in rows if r["area_m"] == area
                         and r["rcs_m2"] == rcs and r["n_looks"] == looks]
                if not group:
                    continue
                # Target-wise aggregation, matching the simulator's definition
                # of worst-target P_D (mean per target, then min).
                per_target = {}
                for r in group:
                    per_target.setdefault(r["target"], []).append(r)
                target_pd = {q: float(np.mean([x["pd_pred"] for x in g]))
                             for q, g in per_target.items()}
                target_gain = {q: float(np.mean([x["required_gain_db"] for x in g]))
                               for q, g in per_target.items()}
                target_gamma = {q: float(np.mean([x["gamma_mean_db"] for x in g]))
                                for q, g in per_target.items()}
                critical = max(target_gain, key=lambda q: target_gain[q])
                summary[f"area{area:g}_rcs{rcs:g}_looks{looks}"] = {
                    "n_trials": len(group) // max(len(per_target), 1),
                    "pd_pred_mean": float(np.mean(list(target_pd.values()))),
                    "pd_pred_worst": float(min(target_pd.values())),
                    "critical_target": int(critical),
                    "required_gain_db_worst": float(target_gain[critical]),
                    "required_gain_db_median": float(
                        np.median(list(target_gain.values()))),
                    "gamma_mean_db": float(np.mean(list(target_gamma.values()))),
                    "gamma_min_db": float(min(target_gamma.values())),
                    "n_links_mean": float(np.mean([r["n_links"] for r in group])),
                }
    (args.out / "shortfall_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf8")
    print("Finished: " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
