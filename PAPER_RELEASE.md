# Paper release contract

> The current integrated theory/model/algorithm direction is documented in
> [V16_MECHANISM_STABLE_THEORY_MODEL_ALGORITHM.md](V16_MECHANISM_STABLE_THEORY_MODEL_ALGORITHM.md).
> V1.6 is mechanism-stable but not generalization-certified. Earlier release
> evidence remains historical unless the V1.6 claim--evidence boundary marks it
> as supported.

The only configuration authorized to generate headline paper results is
`PRESETS["paper-canonical"]` in `isac_sim/config.py`.  `tools/rerun_paper.py`
applies this preset before any experiment-specific override.

## Canonical system boundary

- Task: prior-assisted cooperative confirmation or re-detection after an
  upstream coarse-acquisition stage. Blind first acquisition is not claimed.
- Sensing: continuous multi-UAV OTFS sensing components over one shared
  propagation field, including residual direct-path near-far interference.
- Reporting: target-specific explicit fusion UAV `f_q`; soft statistics travel
  from echo receiver `j` to `f_q`.
- MAC: orthogonal serial reporting. Other report payloads do not co-interfere;
  continuous sensing-waveform leakage remains in the reporting SINR.
- Reliability and latency: finite-blocklength packet success probability and
  block latency `n/B_c`.
- Local statistic: centred independent-look energy LLR under a fast-fluctuation,
  Swerling-II-like model.
- RCS information: the link budget uses `E[sigma_q]`; look-to-look fluctuation is
  marginalized by the local LLR distribution, avoiding double counting.
- Belief information: scheduling uses `(xhat, P)`. The DD gate is obtained by
  propagating `P` through the bistatic delay/Doppler Jacobian.
- C2F: fine gain is the local-window energy `eta_loc`; interpolation parameters
  are legacy-only. The proposed selector refines only its shortlist, while all
  methods evaluate their selected reports with the same local DD estimator.
- Selection objective: detector-predicted weak-target utility, saturated at
  `P_D^req`, minus a per-remote-report price. The first-order
  `alpha_q * DeltaD` rule is an ablation.
- Threshold calibration: the Gaussian threshold includes a first-order
  Cornish--Fisher correction computed from the same post-report LLR moments.

## Claim boundaries

- Correlated Gaussian selection is not claimed to be generally submodular.
- A finite sampled audit is evidence only for the tested instances.
- No `(1-exp(-c))/c` guarantee is claimed under the combined per-target and
  total-link constraints.
- All-neighbor is an upper-resource reference, not a detection upper bound.
- The simulator consumes an upstream predicted belief; it does not implement a
  full multi-CPI tracker posterior update.
- With fixed FBL blocklength and serial reporting, every selected report has
  the same latency cost. For the same candidate utilities and fixed link count,
  cost-aware and exact-marginal greedy therefore have the same ranking. The
  proposed C2F selector can differ because it recomputes shortlisted utilities;
  no gain is attributed to delay pricing at this operating point.

## Release gates

1. `python -m pytest -q`
2. `python tools/run_scientific_gates_v16.py --out results_scientific_gates_v16`
3. Require G1--G2 `pass`, G3 `theory_pass`, G4 `infrastructure_pass`, and label G5 `pending`
   until the algorithm and endpoints are frozen and an independent geometry
   bank has been evaluated.
4. Run the equal-resource rescue controls (`baseline`, `looks_only`,
   `power_only`, `combined`) and the disjoint held-out aspect set before any
   architecture-gain claim.
5. `python tools/rerun_paper.py --mc 1000 --workers 4 --out results_release`
6. Regenerate manuscript figures only from `results_release`.
7. Replace every numerical claim inherited from commit `603b61d`; those values
   were produced by a different model and are not canonical-release evidence.
8. Compile the manuscript and visually inspect every page before tagging a
   submission release.

Legacy presets and outputs remain available for reproducibility and ablation,
but must be labelled as such.

## Staged status (2026-09-14)

- [x] Canonical configuration, validation, and consistency tests.
- [x] Canonical main comparison, 1000 paired trials.
- [x] Finite-instance structural audit (10 scenarios) and 50 exact-oracle cases.
- [x] Figure 2 regenerated from `results_release` and manuscript compiled.
- [ ] Canonical sensitivity, ablation, FBL, correlation, belief, and robustness
  sweeps at their final trial counts.
- [ ] Final page-by-page camera-ready inspection after all release figures have
  been regenerated.

## Experimental capability branch (not yet canonical)

`proposed_c2f_pd` replaces the idealized deflection-to-detection mapping used
for scheduling with a moment-matched prediction built from the implemented
post-report H0/H1 moments and Cornish--Fisher threshold.  It inherits the
standard C2F selector's per-trial total link count, so fixed-blocklength bits
and serial latency remain exactly matched.  A 400-trial screening run improved
P_D from 0.8762 to 0.8830 (paired improvement 0.0068, 95% CI 0.0015--0.0120)
at identical 15.09 reports, 9654.4 bits, and 16.0907 ms.  It is approximately
1.82x the Python selection time of canonical C2F in a single representative
scenario, but remains faster than full refinement.  This branch must complete
the 1000-trial main comparison and robustness/ablation gates before any
promotion into `paper-canonical` or manuscript claims.

`proposed_c2f_adaptive` is a separate architecture audit for coupling C2F to
the evolving greedy state. Its coarse stage performs a greedy rollout and
collects the current per-target alternative frontier at every commit; the fine
stage replays the same marginal rule on that dynamic pool. In a 100-trial
screen it used 58.58 rather than 206.49 fine evaluations (71.6% fewer), with
P_D 0.854 versus 0.858 for static C2F; the paired difference was unresolved
(95% CI for static minus adaptive: -0.0015--0.0095). Independent-observation
singleton caching reduced a representative Python selection time from 1238 ms
to 370 ms, versus 393 ms for static C2F. This result supports improved
screening/greedy integration and refinement efficiency, not a detection-gain
claim; the method remains non-canonical pending a larger paired run.

`proposed_c2f_adaptive_pd` combines the two experimental mechanisms: the
coarse stage records the evolving per-target greedy frontier, the fine stage
replays the detector-aligned marginal rule on that dynamic pool, and the total
number of reports is matched trial by trial to canonical C2F.  In the 1000-
trial candidate main run it achieved P_D=0.8721 versus 0.8655 for canonical
C2F (paired improvement 0.0066, 95% CI 0.0034--0.0098), 0.8603 for exact-
marginal greedy (paired improvement 0.0118, 95% CI 0.0075--0.0161), and 0.8436
for sensing-SINR (paired improvement 0.0285, 95% CI 0.0236--0.0334).  Every
constrained method used the same 15.378 reports, 9841.92 bits, and 16.4032 ms
on average, with identical trial-level resource counts.  The actual worst-
target P_D increased from 0.854 to 0.857 and the worst-target D-satisfaction
probability stayed at 0.879.  The dynamic pool required 122.92 rather than
213.47 fine evaluations per trial (42.4% fewer).  Caching detector moments
whose target-local greedy state has not changed reduced a representative
selection time from 1.81 s to 0.73 s without changing the selected set.

A separate MC=200 hard-cap screen used K=10, 15, and 20, corresponding under
the canonical fixed-blocklength serial MAC to caps of (6.4 kbit, 10.67 ms),
(9.6 kbit, 16.00 ms), and (12.8 kbit, 21.33 ms).  The combined method improved
mean P_D over canonical C2F by 0.009, 0.009, and 0.008, respectively, but its
post-hoc worst-target P_D was lower at K=10 and K=20.  A sum-log-P_D fairness
candidate was also screened and rejected because it did not repair that
worst-target result.  Consequently the combined method is the leading
candidate at the current operating point, but is not promoted to
`paper-canonical`: a standalone system-defined hard budget and a robust
per-trial outage/fairness gate are still required.

The belief-mode deflection summary was also corrected so refined truth tables
use only DD-gate-captured links and the truth geometry, matching the detector.
This changes reported D statistics but not the selected sets, P_D, P_FA, bits,
or delay.

### Packetization audit (not a model change)

The canonical accounting currently charges one full
`K_candidates * b_d = 4 * 160 = 640` bit packet for every selected observation
`(i,j,q)`.  Because the manuscript does not yet define the four packet items,
this is an explicit audit item rather than an assumed physical fact.  A 200-
trial audit of `proposed_c2f_adaptive_pd` found 15.65 reports but only 3.97
active reporting UAVs on average.  The busiest UAV carried 9.40 reports
(6.01 kbit) on average, 22.1 reports (14.14 kbit) at the trial-level 90th
percentile, and 38 reports in the most concentrated trial.  Thus fleet-average
bits per UAV are not a valid load-balance claim.

As an accounting-only counterfactual, packing up to four observations produced
by the same `(j,q)` into one padded packet reduced the mean from 10.013 to
7.558 kbit and from 16.688 to 12.597 ms on the first 200 canonical scenarios
(24.5% using the ratio of aggregate means).  This is not yet a valid detection
result: observations in one packet share a packet-success event, so aggregation
requires packet-level FBL reliability and correlated erasure sampling before
promotion.  The reusable audit is `tools/audit_packetization.py`.

Shorter fixed blocklengths were also screened without changing the 640-bit
payload definition.  On the same 200 scenarios, `n=2048` gave
`P_D=0.8710`, 10.013 kbit and 16.688 ms; `n=1536` gave `P_D=0.8605`,
10.707 kbit and 13.384 ms; and `n=1024` gave `P_D=0.8350`, 12.051 kbit and
10.043 ms.  Lower `n` reduced latency but also reduced the feasible reliable
edge set and induced more selected reports, increasing total bits.  Neither
shorter-block point is a Pareto improvement over the current candidate.

### Fusion-destination bottleneck audit (candidate, not a release change)

Under `max_in_rate`, all ten targets are assigned to one fusion UAV on average.
After correcting local-evidence accounting, the MC=100 fusion audit gives 6.65
remote reports and 7.093 ms, with one receiver-conflict slot per report.

A target-local candidate instead chooses the UAV nearest to each target's
*predicted* position (`nearest_target`; the historical configuration spelling
`nearest_centroid` remains an alias).  It uses the belief geometry and never
the current-CPI target truth.  In the MC=1000 main gate, the combined adaptive
C2F method achieved `P_D=0.9762` from 12.101 observations with 0.751 remote
reports, 0.481 kbit, and 0.801 ms. Its paired improvement over exact-marginal
greedy was 0.0351 (95% CI 0.0310--0.0392). Against Sensing-SINR, the paired
difference was -0.0021 (95% CI -0.0051--0.0009), so detection was statistically
tied while target-local C2F used 83.7% less payload at the same observation count.

The corrected MC=100 target-local screen averages 0.74 remote reports and 0.70
conflict slots because local observations create no MAC traffic. The distance
assignment remains a prediction-only heuristic rather than a jointly optimal
fusion and selection rule.
