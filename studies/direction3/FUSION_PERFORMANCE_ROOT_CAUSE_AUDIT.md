# Why the fixed two-UAV fusion did not improve detection

## What the experiment actually established

The statistic is a fixed normalized sum of local neighbourhood GLRT values
from receivers 0 and 1, for target 1 only. Twenty independent scenes supplied
one fused H0 score each for split-conformal calibration; forty different
scenes supplied final H0/H1 scores. The rank-20 threshold is 4.7538. Its
future-scene marginal H0 exceedance bound is 1/21 under exchangeability.
The test set has zero false alarms, but its 95% Wilson upper limit is 0.088.
This is a valid *fixed-score, single-target marginal* gate, not network-wide
family-wise control across all targets or a demonstrated conditional guarantee
within a fixed geometry/channel block.

## Direct cause of missed detections

The final two-UAV test has AUC 0.745 and detects 3/40 targets. Receiver 0
alone detects 4/40 (AUC 0.575); receiver 1 alone detects 4/40 (AUC 0.789).
These are separately calibrated decisions, so the one-count comparison is
descriptive rather than a general performance ordering.

The fused threshold is set by calibration scene 4: its H0 components are
1.9966 and 2.7573, summing to 4.7538. In test scene 21, receiver 1 has H1
score 3.4908, above its own normalized gate 2.7573, but receiver 0 contributes
only 1.1784. The sum 4.6692 misses the fused gate. Scene 52 is the mirror:
receiver 0 scores 2.8616, above its gate 2.4796, while receiver 1 contributes
1.7095; the sum 4.5711 also misses. These two scenes expose the cost of an
unweighted sum when one receiver has evidence and the other does not.

Calibration H0 components have low measured correlation (0.108); test H0
correlation is 0.061. The current experiment therefore has little evidence of
a strong H0 correlation to exploit. Test H1 scores correlate more strongly
(0.611), consistent with shared scene geometry making favourable target links
co-occur. These estimates are from only 20/40 scenes and are not stable
population parameters.

## Why this remains a weak tail-detection experiment

At alpha 0.05, 20 calibration scenes only support the maximum H0 score as a
finite conformal threshold. That conservative, high-variance extreme sets
the operational decision. Several test H1 fused scores fall just below it:
4.669, 4.571 and 4.508. AUC describes all H0/H1 rankings and therefore can
be 0.745 even when only 3/40 H1 scores cross this particular tail gate.

The observed AUC difference, fused minus receiver 1, is -0.0438. A paired
scene bootstrap interval is approximately [-0.136, 0.049]. The data cannot
establish that fusion is truly worse than receiver 1, only that this pilot
fails to show an improvement. Likewise, the 3/40 versus 4/40 detection counts
do not resolve a stable difference.

The physical setting is intentionally hard: target RCS 0.05 m2 and direct
interference boost +30 dB. Receiver 0's perfect-channel reference on this
same split detects only 6/40 at its own gate (AUC 0.590), so covariance repair
cannot alone create a high-PD sensing regime there. Receiver 1 has better
ranking, but its strict local gate still yields only 4/40 detections.

## Model and protocol gaps relative to the requested system

1. The subset {0,1} and equal normalized weights were fixed for this wiring
   test. No sensing association or correlation-aware weight optimization was
   evaluated. The local H0 means and variances differ despite rank
   normalization, so the sum is not a likelihood ratio or an optimal fusion
   rule.
2. There is one realization per scene. Although DD estimation error is scoped
   to a scene, this run does not test using pilots to learn a residual model
   and then detecting in the same fixed-estimate block. It tests marginal
   transfer to new scenes.
3. The benchmark seeds target/path phases separately for each receiver. The
   receivers share scene geometry and link budgets, but the run is not a
   coherent, common-scatterer multi-receiver waveform simulation. Statistical
   score fusion is a legitimate pilot, but its output cannot be interpreted
   as coherent network LLR fusion.
4. Reporting links, finite communication/quantization, association selection,
   other targets, and the final network alarm rule are absent. The 1/21 bound
   applies to this one preselected target and pair, not the requested global
   PFA of the complete multi-target network.

## Correct next experiment

Use three independent scene blocks: fit a budgeted non-enumerative subset and
H0-based fusion model on the first; calibrate its frozen final network score
on the second; evaluate PFA, PD and uncertainty on the third. Define the alarm
across all tested targets before calibration and include the allowed report
budget. Keep a single receiver and fixed sum as prespecified baselines.
An in-block pilot/data experiment is needed separately if the claim is
conditional PFA under one persistent channel-estimation error.
