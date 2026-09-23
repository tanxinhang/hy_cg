# Does lowering direct interference improve target detection?

## Paired protocol

Receivers 0 and 1, target 1, target RCS 0.05 m2, fixed scene-scoped DD error,
uncertainty-weighted TP-UIC, sigma-point residual covariance and 3x3
neighbourhood GLRT were held fixed. The direct-path gain boost was changed
from +30 dB to 0 dB. Both arms use the same master seed, 20 calibration
scenes, 40 test scenes and one realization per scene. Geometry, target echo,
noise draw and DD estimation error are paired across boost levels; the direct
gain is the intervention. A separate rank-20 split-conformal threshold was
calibrated from each arm's H0 scores at target alpha 0.05.

| Receiver/score | +30 dB AUC | 0 dB AUC | +30 dB detections | 0 dB detections |
|---|---:|---:|---:|---:|
| UAV 0 | 0.575 | 0.604 | 4/40 | 4/40 |
| UAV 1 | 0.789 | 0.826 | 4/40 | 4/40 |
| Fixed normalized sum, UAVs 0+1 | 0.745 | 0.775 | 3/40 | 4/40 |

All six test configurations have 0/40 false alarms. Each final score has a
separately calibrated marginal future-scene H0 exceedance bound of 1/21,
conditional on exchangeability. The bound is for this preselected target and
score, not for all targets or a learned association rule.

The paired fused AUC increment is +0.030. A paired scene bootstrap 95%
interval is approximately [-0.003, +0.068]; for UAV 1 the increment is
+0.037 with interval approximately [-0.007, +0.093]. These intervals include
zero. The fused rule gains one detection, scene 51, and the single UAV 1
detects the same four scenes at both interference levels. The numerical
improvement is suggestive, not statistically established.

## Mechanism and limit

Lowering the direct gain by 30 dB reduces median actual structural direct
residual energy by approximately 970--990 times for these receivers. Target
echo energy and survival stay essentially unchanged. The sigma-point model
already represents much of that residual, so median target-direction
information rises much less: UAV 0's NCP unit changes 1.125 to 1.404 and
UAV 1's 0.678 to 0.715. Thus a 1000-fold energy reduction does not imply a
1000-fold detection improvement.

The strict alpha=0.05 gate uses the largest of only 20 calibration H0 scores.
It is sensitive to extreme calibration scenes, and 40 test scenes resolve PD
only in steps of 0.025. The low RCS work point and target-template/geometric
limits remain. This experiment answers the sensitivity question: reducing
direct interference can improve score separation, but it does not by itself
produce a high-PD system under the current budget and gate.

An initial four-calibration/eight-test scan at 0, 10, 20 and 30 dB showed a
similar broad tendency but was too small and nonmonotone to support a dose
response claim. Its conformal thresholds are infinite at alpha=0.05, so the
reported PD=0 in that scan is a sample-count artifact and is not compared
with the 20/40 experiment.
