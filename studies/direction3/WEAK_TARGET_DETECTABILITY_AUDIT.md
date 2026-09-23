# Is the low target detection rate a physical impossibility?

## Matched-scene diagnostic

Target 1 has RCS 0.05 m2. Receiver 1 was evaluated on the same 20
calibration and 40 independent test scenes at target PFA 0.05. All thresholds
are calibrated separately on each arm's H0 scores. The configurations below
share physical geometry, target echo realizations and noise seeds.

| Receiver and detector | Test AUC | Detections/40 | Test false alarms/40 |
|---|---:|---:|---:|
| TP-UIC, +30 dB direct boost, belief neighbourhood | 0.789 | 4 | 0 |
| TP-UIC, 0 dB direct boost, belief neighbourhood | 0.826 | 4 | 0 |
| Perfect direct-channel cancellation, belief neighbourhood | 0.819 | 4 | 0 |
| Perfect cancellation, true target DD, centre GLRT | 0.759 | 4 | 0 |

The true-DD centre detector is a diagnostic with test-truth access, but is not
a strict upper bound on the neighbourhood detector: they search different
template sets and have different null distributions/thresholds. Likewise,
perfect direct cancellation removes one impairment but is not an optimal
physical detector. No row proves an information-theoretic impossibility.

The exact same test scenes, 21, 31, 42 and 57, are detected by every row.
For perfect direct cancellation with the belief neighbourhood detector, the
median target-direction NCP unit is 20.4 in those four scenes but only 0.77
in the 36 missed scenes. Target echo energy spans more than 30-fold between
the 10th and 90th percentiles. Three detected scenes have especially strong
target echoes; scene 21 is an exception with a weak echo, so this association
is evidence of heterogeneity rather than a deterministic detection rule.

For TP-UIC at +30 dB, test H0 and H1 statistic means are 6.44 and 9.54,
while the rank-20 conformal threshold is 13.79. Reducing direct boost to
0 dB raises the H1 mean to 9.96, but the separately calibrated threshold is
14.77. The improved AUC therefore does not translate into more threshold
crossings. With only 20 calibration H0 scenes, the valid alpha=0.05 threshold
is the largest observed H0 score and is sensitive to the tail of that small
sample.

## Interpretation

There is a physical limitation **under the current resource and geometry
configuration**: many scene/receiver links deliver too little target evidence
for one observation to cross this conservative 5% gate. Direct interference
is not the sole explanation, because perfect direct cancellation and lowering
its power yield the same 4/40 detections. The low-RCS echo, path geometry,
finite observation budget and target-state uncertainty jointly matter.

This is not an innate inability to detect a 0.05 m2 target. The retained
target fraction is about 0.994, the target subspace is nearly identifiable,
and the NCP is positive in the missed scenes. With additional independent
looks or a higher-gain geometry/array, evidence can in principle accumulate;
the required resource increase has not yet been quantified. A useful next
experiment would vary independent CPI count or receive aperture while holding
RCS, global PFA and reporting budget fixed, with a separate calibration gate
for each predeclared resource level.

The current conclusion remains single receiver/target marginal. The full
network's conditional and multi-target detection limits have not been
established by this pilot.

A subsequent low-cost oracle-centre calculation used the existing per-scene
NCP values and the actual calibrated H0 gate. It predicts 0.117 detection on
held-out geometries at baseline, close to the observed 0.100, and estimates
that average PD 0.8 would need about 15.3 dB more effective target-direction
information under a fixed single-look score. This is a prioritization proxy,
not a CPI or power requirement; see `TARGET_GAIN_REQUIREMENT_DIAGNOSTIC.md`.
