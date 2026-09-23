# Local residual-basis mechanism gate (2026-09-23)

## Question

Can a fixed, receiver-known first-order local basis capture the effective
direct-path residual direction left by TP-UIC?  A positive answer is a
necessary condition for learning a low-rank residual covariance correction in
that basis.

## Preregistered gate

- Scenario: receiver 0, target 1, direct-path boost +30 dB.
- Six independent scenes and two observation realisations per scene.
- TP-UIC uses first-order, uncertainty-weighted interference tangents and the
  sigma-point direct-mismatch covariance.
- The simulator-truth direct residual is a diagnostic label only.  Candidate
  bases use only the estimated direct dictionary and belief target dictionary.
- Both label and basis are passed through the same TP-UIC residual map.
- Primary metric: projection-energy coverage after residual-covariance
  whitening.
- Proceed only if median coverage is at least 0.80 and lower quartile coverage
  is at least 0.50.

## Result

| Receiver-known residual basis | Median | Lower quartile | Minimum | Raw median |
|---|---:|---:|---:|---:|
| Direct templates/tangents | 0.00017 | 0.00003 | 0.00002 | 0.00329 |
| Direct + tested-target templates/tangents | 0.00328 | 0.00199 | 0.00016 | 0.01062 |
| Direct + all-target templates/tangents | 0.02394 | 0.01169 | 0.00125 | 0.04037 |

No candidate passes.  The failure is orders of magnitude below the gate, not a
sample-size ambiguity near the threshold.

## Decision

**Stop the first-order local-basis low-rank learning branch.**  Do not tune
weights, shrinkage, or ranks inside this basis: it does not contain the missing
direction in the detector geometry.

This result is consistent with the proposed nonlinear reframing.  The next
bounded experiment should represent the residual distribution using nonlinear
sigma-point deviations (or an explicit second-order alternative), fit any
calibration only on train/H0 data, and compare it against the already-effective
fixed sigma-point covariance.  It must not use the oracle residual direction
as an estimator input.

Machine-readable records and the exact protocol are in
`local_residual_basis_screen/summary.json` and `records.csv`.
