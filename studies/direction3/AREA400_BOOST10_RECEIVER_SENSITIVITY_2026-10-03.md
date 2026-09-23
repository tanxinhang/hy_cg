# 400 m / +10 dB direct-interference sensitivity (2026-10-03)

## Controlled change

Starting from the 400 x 400 m receiver experiment, only the artificial direct
gain boost changes from +30 dB to +10 dB.  Master seed, scene identifiers,
20/20/40 split, geometry draws, RCS, DD error, receiver, nonlinear covariance,
3x3 hypothesis bank and detector-selection rule are unchanged.  Each operating
point receives its own calibration-H0 conformal threshold.

## Result

| Operating point | Train max AUC | Test AUC | Threshold | Test PFA | Test PD |
|---|---:|---:|---:|---:|---:|
| 800 m / +30 dB | 0.5050 | 0.6894 | 16.335 | 0.025 | 0.025 |
| 400 m / +30 dB | 0.6350 | 0.7744 | 13.028 | 0.025 | 0.275 |
| 400 m / +10 dB | **0.6425** | **0.8225** | **12.331** | 0.025 | **0.350** |

Relative to the paired 400 m / +30 dB scenes, reducing the boost by 20 dB
changes test AUC by +0.0481 and PD by +0.075.  Same-scene bootstrap resampling
of the 40 test rows gives:

- AUC difference median +0.0463, exploratory 95% interval [+0.0175, +0.0881];
- PD difference median +0.075, exploratory 95% interval [-0.05, +0.20].

The paired AUC change is consistently positive in this sample.  The tail-PD
change remains uncertain because only 40 test scenes and 20 threshold scenes
are available.  Median H0/H1 statistics change from 6.82/10.02 at +30 dB to
5.72/10.51 at +10 dB: lower interference reduces the null statistic while the
target statistic increases slightly.

The detector selection remains unchanged.  At +10 dB, train AUC is 0.6425 for
max and 0.6325 for log-mean-exp, so the preregistered rule retains max.

## Decision

The +10 dB operating point is detectably easier and useful as a nominal
condition.  Keep +30 dB as a strong-interference stress condition.  Geometry
contraction from 800 to 400 m produced the larger observed PD movement, while
the subsequent 20 dB interference reduction produced a smaller but more
consistent AUC gain.

This remains a single-master-seed sensitivity result.  Independent-seed
replication is required before choosing +10 dB as the paper headline point.

Machine evidence is in
`data/neighbourhood_aggregation_area400_boost10_gate_20261003/`.
