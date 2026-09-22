# Scene-adaptive power trial on the frozen 600 m scene

The comparison reuses the same frozen scene, 25 looks, TP-UIC arm, report
budgets, 4.8 W fleet budget, and 1 W per-UAV cap. Only the power policy changes.

| Policy | Delivered PD | Worst PD | Reports | Latency |
|---|---|---:|---:|---:|
| Existing explicit vector | `[0.9834, 0.9124, 0.6066]` | 0.6066 | 4 | 1.067 ms |
| Scene geometry max-min | `[0.9917, 0.9431, 0.4938]` | 0.4938 | 2 | 0.533 ms |

The adaptive candidate is rejected: worst-target PD decreases by 0.1128. The
geometry surrogate improves targets 1/2 but underfunds the sensing/reporting
combination needed by target 3. This demonstrates that being scene-adaptive is
not sufficient; the accepted objective must be actual delivered PD.

No performance gain is claimed. The existing explicit vector remains the
incumbent for this frozen scene. A next low-complexity step, if pursued, should
use a small fixed-step coordinate search around that incumbent and accept only
strict improvements in worst delivered PD.
