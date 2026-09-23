# Fixed two-UAV threshold sensitivity

The fixed UAV {0,1}, target-1 normalized-sum statistic was re-evaluated from
saved scores without changing the receiver or using test data to define a
threshold. Each direct-boost arm has 20 independent calibration H0 scenes and
40 independent test scenes. Rank $k$ below is the $k$th smallest calibration
H0 score. For a continuous exchangeable future H0 score, its exceedance
probability is $(21-k)/21$.

| Direct boost | Calibration rank | Threshold | Distribution-free marginal PFA bound | Test false alarms | Test detections |
|---:|---:|---:|---:|---:|---:|
| +30 dB | 20/20 | 4.754 | 4.76% | 0/40 | 3/40 |
| +30 dB | 19/20 | 3.941 | 9.52% | 1/40 | 9/40 |
| 0 dB | 20/20 | 5.162 | 4.76% | 0/40 | 4/40 |
| 0 dB | 19/20 | 3.921 | 9.52% | 0/40 | 8/40 |

The current gate is high because alpha=0.05 with 20 calibration scenes
forces the maximum H0 score. Lowering to rank 19 is a legitimate
approximately-10%-PFA design choice, but is not a 5%-PFA detector. Its
0/40 false-alarm count at 0 dB has a 95% Wilson upper limit around 8.8%,
so the test count cannot restore the 5% claim.

At fixed alpha=0.05, the distribution-free route to a less extreme order
statistic is to increase the number of independent H0 calibration scenes.
For example, with 100 scenes the conformal rank is 96/100. That threshold
may be lower if the present maximum was an outlier, but a decrease is not
guaranteed. Repeated noise draws within one shared scene are not 100
independent scene samples and must not be counted as such.

An exact-alpha randomized rank-19/rank-20 gate would use rank 19 only 5% of
the time and rank 20 otherwise. Its expected detection increase is tiny and
it produces a random operational decision, so it is not recommended here.

All rows concern one preselected target and receiver pair. A multi-target
network-wide PFA requirement must calibrate the final all-target alarm score,
which may have a different threshold and detection tradeoff.
