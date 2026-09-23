# Low-cost target-direction gain diagnostic

## Purpose and source

This calculation uses the existing 20 calibration and 40 held-out scene
records for receiver 1, target 1, RCS 0.05 m2, perfect direct-channel
cancellation and the true target DD centre template. It requires no new
waveform simulation. The target template and direct cancellation are oracle
information, so the result is a resource-screening diagnostic rather than a
deployable receiver claim.

For each scene the receiver records the GLRT rank and expected target
noncentrality $\lambda_q$. For a fixed single-look statistic with real degrees
of freedom $d=10$, the approximation is

$$
2T\mid H_1\sim\chi'^2_d(2g\lambda_q),\qquad
P_D(g)=\frac1N\sum_q
\Pr\{\chi'^2_d(2g\lambda_q)>2\tau_{\rm cal}\}.
$$

Here $g$ is an *effective target-direction information gain* and
$\tau_{\rm cal}=13.035$ is the observed H0 split-conformal threshold. The
ideal central chi-square gate would be 9.154, which is much lower; using it
would overstate the single-look detection rate. The calibrated threshold is
held fixed solely to invert a hypothetical target-only gain.

## Sanity check and estimate

At $g=1$, the model predicts 0.076 mean detection on the 20 calibration
scenes and 0.117 on the 40 held-out scenes. The actual held-out detection is
4/40 = 0.100. This agreement is a coarse aggregate check; it does not
validate per-scene probabilities or new resource levels.

Inverting the calibration-scene model gives:

| Desired mean PD | Required effective gain | Gain in dB | Model PD on held-out scene geometries at that gain |
|---:|---:|---:|---:|
| 0.50 | 13.0x | 11.1 dB | 0.617 |
| 0.80 | 33.7x | 15.3 dB | 0.823 |

With the gate fixed, bootstrapping the 20 calibration scene NCP values puts
the 0.80 gain estimate roughly between 13.4 and 16.7 dB. This omits
uncertainty in the conformal tail threshold itself and is not a confidence
interval for actual system performance.

## Decision

The calculation supports a resource-scale conclusion: small increments of
target-direction information are unlikely to make this work point a high-PD
detector. It does **not** say that 34 independent CPIs are needed. Changing
CPI count changes the statistic's degrees of freedom, temporal dependence,
communication/latency budget and H0 law; changing transmit power also changes
direct interference. Those interventions each require a new final-statistic
calibration. The next efficient physical test should select one feasible
resource intervention with predicted gain near this scale, then run a paired
anchor experiment rather than a grid. Geometry/receiver selection can be
screened cheaply from existing per-scene NCP maps before any waveform rerun.

Machine-readable calculation: `data/target_gain_requirement_rx1/result.json`.
