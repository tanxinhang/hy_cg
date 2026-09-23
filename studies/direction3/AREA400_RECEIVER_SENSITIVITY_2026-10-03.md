# 400 m receiver-scene sensitivity (2026-10-03)

## Controlled change

Only the horizontal geometry extent changes from 800 x 800 m to 400 x 400 m.
The master seed and scene identifiers are shared, so normalized location draws
are paired and horizontally rescaled.  UAV/target altitude and speed ranges,
six UAVs, three targets, receiver 0, target 1, RCS 0.05 m2, +30 dB direct
boost, 0.1-bin scene-fixed DD error, TP-UIC, nonlinear sigma-point replacement
covariance, the 3x3 detector, and the 20/20/40 train/calibration/test split are
unchanged.

This does not halve three-dimensional path lengths: UAV altitude remains
800--1200 m and target altitude remains 700--1500 m.  The intervention mainly
changes horizontal density and bistatic geometry.

## Result

| Horizontal area | Train max AUC | Test AUC | Conformal threshold | Test PFA | Test PD |
|---|---:|---:|---:|---:|---:|
| 800 x 800 m | 0.505 | 0.689 | 16.335 | 0.025 | 0.025 |
| 400 x 400 m | **0.635** | **0.774** | **13.028** | 0.025 | **0.275** |

The held-out test contains 40 scenes: detections increase from 1 to 11 while
false alarms remain 1.  Median H0/H1 statistics move from 5.64/7.75 to
6.82/10.02, increasing separation despite a higher H0 centre.

Using same-scene paired bootstrap resampling of the 40 test rows:

- AUC difference: +0.0856, exploratory 95% interval [-0.0656, +0.2338];
- PD difference at each area's independently calibrated threshold: +0.25,
  exploratory 95% interval [+0.10, +0.40].

The AUC interval remains wide and crosses zero.  The PD interval excludes zero
conditional on the two fitted thresholds, but does not include calibration-set
uncertainty.  Therefore this is strong sensitivity evidence, not a final
cross-seed claim.

The detector choice does not change.  At 400 m, train AUC is 0.635 for max and
0.6325 for log-mean-exp, so the preregistered selector retains max.

## Decision

The 400 m geometry is a substantially more detectable operating point and is
appropriate for the next receiver validation matrix.  It must not silently
replace the 800 m stress point: both should be retained as separate geometry
regimes.  The next test should repeat 400 m under independent master seeds and
then sweep direct boost/DD error without tuning the detector.

Machine evidence is in
`data/neighbourhood_aggregation_area400_gate_20261003/`.
