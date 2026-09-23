# Fixed two-UAV fusion and final-statistic calibration

This step checks the post-fusion gate before learning a UAV subset or fusion
weights. Receivers 0 and 1, target 1, +30 dB direct boost, the sigma-point
residual covariance and the 3x3 target-neighbourhood detector were fixed in
advance. The fusion score is the sum of the two local GLRT statistics after
division by each local complex rank. The exact same 20 independent scene IDs
calibrate both receivers; 40 disjoint scene IDs are held out. Each scene
contributes one joint H0/H1 score.

The conformal threshold is the 20th of 20 calibration **fused H0** scores,
4.754. Under future-scene exchangeability, the marginal chance of exceeding
that threshold under H0 is at most 1/21 = 0.0476. This statement concerns
the fixed pair and target; it does not cover a search over UAV subsets,
targets, or fusion rules performed after looking at the calibration scores.

| Detector | Test AUC | Test false alarms | Test detections |
|---|---:|---:|---:|
| Receiver 0 alone | 0.575 | 0/40 | 4/40 |
| Receiver 1 alone | 0.789 | 0/40 | 4/40 |
| Fixed normalized sum, receivers 0+1 | 0.745 | 0/40 | 3/40 |

The fused score ranks H1 above H0 reasonably well but loses one tail detection
relative to either local detector at the strict conformal threshold. Its test
H0 receiver correlation is 0.061 in this sample. Consequently this is a
successful **post-fusion PFA wiring** check, not evidence that correlation
aware fusion or sensing association improves detection. The observed 0/40
false alarms has a 95% Wilson upper limit of about 0.088; the finite-sample
guarantee comes from the calibration rank and exchangeability, not that count.

The fusion script validates aligned scene IDs, one realization per scene,
matching physical/receiver settings, distinct receivers and absence of the
oracle truth-leaking covariance. A regression test changes only test H0
scores and confirms the threshold remains fixed.

The next association experiment needs three scene-disjoint blocks: one to
fit the non-enumerative subset and H0 covariance/fusion weights, one to
calibrate the resulting frozen final score, and one for final evaluation.
The present 20/40 split cannot estimate weights on its calibration block and
still claim the same split-conformal guarantee.
