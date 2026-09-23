# Receiver detector aggregation gate (2026-10-03)

## Question

Can the nonlinear uncertainty-aware receiver improve detection by aggregating
evidence across its fixed 3x3 delay--Doppler hypothesis bank, instead of using
only the largest local GLRT statistic?

## Frozen candidate

The baseline remains the neighbourhood maximum.  The only candidate is an
unweighted, parameter-free log-mean-exp statistic:

$$
T_{\mathrm{LME}}=\log\left(\frac{1}{L}\sum_{l=1}^L e^{T_l}\right).
$$

It interpolates between distributed evidence and the maximum without changing
the hypothesis grid, whitening, nuisance projection, receiver, or physical
observation.  Because its analytic null law is not the ordinary GLRT law, it
has no analytic threshold and requires calibration of the final statistic.

## Isolation and gate

- 20 independent train scenes select between `max` and `logmeanexp`.
- The candidate must exceed max train AUC by at least 0.01 to be selected.
- 20 disjoint calibration H0 scenes set a 5% split-conformal threshold.
- 40 further test scenes are evaluation-only.
- Final promotion would require at least +0.02 test AUC and no lower PD.
- Receiver 0, target 1, +30 dB direct boost, 0.1-bin scene-fixed DD error,
  nonlinear sigma-point replacement covariance, and the existing 3x3 grid are
  fixed throughout.

## Result

| Statistic | Train AUC |
|---|---:|
| Neighbourhood maximum | **0.5050** |
| Log-mean-exp | 0.4925 |

The train gate selects `max`; the new candidate is therefore not evaluated on
test for promotion.  The frozen max detector obtains on held-out test:

- AUC: 0.6894;
- split-conformal threshold: 16.335;
- empirical PFA: 0.025;
- empirical PD: 0.025.

## Decision

**Retain the neighbourhood maximum.**  Evidence aggregation does not improve
the development split, so a test-set rescue would be post-selection bias.  The
very low tail PD also indicates that the dominant limitation is target evidence
at this stress point, not how the nine local hypotheses are pooled.

The implementation remains available as an explicitly calibrated experimental
statistic.  It does not change the default max detector or its metadata.  A
future detector experiment should change an information-bearing component
(for example independent multi-CPI evidence or a physically derived target
prior), not add another aggregation temperature or enlarge the grid.

Machine-readable results are in
`data/neighbourhood_aggregation_gate_20261003/summary.json` and per-scene paired
scores are in `records.csv`.
