# Sigma-point covariance scale gate (2026-10-01)

## Purpose

Test whether a train/H0-only scalar correction can improve the fixed physical
sigma-point residual covariance.  This is stage 2A after rejecting a learned
low-rank correction in the first-order local basis.

## Frozen protocol

- Receiver 0, target 1, direct-path boost +30 dB and DD error sigma 0.1 bin.
- One paired H0/H1 observation per independent scene.
- 20 train scenes select one scale from `{0.5, 0.75, 1, 1.5, 2}` by making
  median target-direction H0 inflation closest to one.
- 20 disjoint calibration scenes set each method's 5% split-conformal
  threshold.  Forty further disjoint scenes are used only for evaluation.
- Proceed only if the learned scale improves test AUC by at least 0.02, does
  not reduce PD at the conformal threshold, and has median held-out H0
  inflation in `[0.8, 1.25]`.

## Train-only selection

| Scale | Median train H0 inflation |
|---:|---:|
| 0.50 | 1.326 |
| 0.75 | **1.312** |
| 1.00 | 1.315 |
| 1.50 | 1.331 |
| 2.00 | 1.342 |

Scale 0.75 is selected, but the train objective is nearly flat: its advantage
over the physical scale 1 is only 0.003 in median inflation.  The data do not
show a strongly identifiable scalar correction.

## Held-out result

| Method | Scale | Threshold | PFA | PD | AUC | Median H0 inflation |
|---|---:|---:|---:|---:|---:|---:|
| Fixed physical sigma point | 1.00 | 11.947 | 0.050 | 0.050 | 0.6163 | 1.202 |
| Train/H0 learned scale | 0.75 | 12.306 | 0.050 | 0.025 | 0.6225 | 1.197 |

The AUC gain is 0.0063, below the required 0.02, and threshold PD decreases by
0.025.  The gate fails even though held-out covariance calibration remains in
range.

## Decision

**Stop scalar covariance learning and retain scale 1.**  The fixed sigma-point
model is both simpler and more stable at the operational tail.  Combined with
the failed first-order local-basis gate, these data do not justify fitting a
more flexible covariance correction from only 20 H0 scenes: the available
train signal is too flat and additional degrees of freedom would primarily
increase selection variance.

The next defensible step is not another learned-weight sweep.  It is an
independent-seed and stress-axis validation of the fixed nonlinear sigma-point
receiver against first-order covariance, conventional IC, and the
perfect-channel diagnostic.  This tests whether the physics-derived nonlinear
covariance itself can carry the TP-UIC contribution.

Machine-readable protocol and results are in
`data/sigma_covariance_scale_gate_20261001/summary.json`; paired per-scene
scores are in `records.csv`.
