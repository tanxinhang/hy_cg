# 400 m / +10 dB multi-CPI detector gate (2026-10-04)

## Bottleneck audit

Before changing the detector, a paired receiver headroom run compared plain LS,
current TP-UIC and perfect-channel cancellation at 400 m / +10 dB:

| Arm | AUC | PFA | PD | Target survival |
|---|---:|---:|---:|---:|
| Plain LS | 0.803 | 0.100 | 0.575 | 0.993 |
| Nonlinear TP-UIC | 0.804 | 0.100 | 0.575 | 0.995 |
| Perfect channel | 0.830 | 0.025 | 0.500 | 1.000 |

This independent-seed diagnostic leaves only 0.026 AUC between TP-UIC and the
perfect-channel reference, while plain LS and TP-UIC are nearly identical.
Cancellation tuning is therefore not the main available performance lever at
this operating point.  The different threshold PD ordering is not an oracle
ordering: each arm has its own calibration maximum, and only 20 calibration
scenes make the tail threshold noisy.

## Candidate

Use two independent receiver blocks for the same scene.  Each block performs
the complete frozen chain independently:

`TP-UIC -> nonlinear covariance -> whitening -> 3x3 Max-GLRT`.

The two final statistics are added noncoherently.  Geometry, target belief and
direct DD estimation error are held fixed within a scene; observation noise and
block phases are independently drawn.  The scene, not the CPI, remains the
exchangeable sample.  A new master seed (`20261004`) separates this validation
from the earlier 400 m development data.

Both one-CPI and two-CPI final statistics receive their own 20-scene
split-conformal calibration.  Forty independent scenes are used for test.
Promotion requires at least +0.02 test AUC and no lower PD.

## Result

| Detector | Train AUC | Test AUC | Threshold | Test PFA | Test PD |
|---|---:|---:|---:|---:|---:|
| One CPI | 0.800 | 0.674 | 10.735 | 0.050 | 0.225 |
| Two-CPI noncoherent sum | **0.828** | **0.739** | 17.969 | 0.075 | **0.525** |

Point-estimate changes are +0.065 AUC and +0.300 PD, so the preregistered gate
passes.  Same-scene bootstrap intervals over the 40 test scenes are:

- AUC difference: +0.065, exploratory 95% interval [-0.050, +0.176];
- PD difference: +0.300, exploratory 95% interval [+0.175, +0.450];
- PFA difference: +0.025, exploratory 95% interval [0.000, +0.075].

Wilson intervals are [0.014, 0.165] for one-CPI PFA (2/40), [0.026, 0.199]
for two-CPI PFA (3/40), [0.123, 0.375] for one-CPI PD (9/40), and [0.375,
0.671] for two-CPI PD (21/40).  Thus the tail-PD improvement is strong in this
sample, while the AUC gain and true PFA still need more independent scenes.
The conformal rank guarantee is marginal over a future exchangeable scene; it
does not require every batch of 40 tests to realize at most 5% false alarms.

## Cost and decision

**Promote two-CPI accumulation to the next validation stage, not to production.**
It is the first tested detector change in this sequence to pass the predefined
performance gate, but costs approximately twice the observation blocks,
receiver computation and latency.  It should be exposed as a performance--
latency mode, with one CPI retained as the low-latency baseline.

Next requirements:

1. repeat with independent master seeds and at least 40 calibration scenes to
   reduce maximum-threshold variance;
2. define the physical CPI duration/coherence assumption explicitly;
3. compare two CPI against two independent looks of plain LS and perfect
   channel, so the gain is attributed to evidence accumulation rather than
   TP-UIC;
4. only after those checks consider three or more CPI counts.

Machine evidence:

- `data/area400_boost10_multicpi_gate_20261004/`
- `data/area400_boost10_receiver_headroom_20261003/`
