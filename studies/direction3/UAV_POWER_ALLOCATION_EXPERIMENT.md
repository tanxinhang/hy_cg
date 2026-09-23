# UAV sensing/communication power experiment

## Protocol

- Trial 32, 600 m square region, Q=3, M=6, 25 looks (53.333 ms).
- 512-channel-use finite-blocklength reports with a 640-bit LLR payload.
- Each UAV transmits at most one report and each fusion UAV receives at most two.
- Per-UAV total power is split as
  `P_sense = rho * P_total` and `P_comm = (1-rho) * P_total`.
- Power is applied before receiver/TP-UIC construction and before communication
  SINR/FBL reliability calculation. It is not a post-processing PD multiplier.

## Total-power sweep at rho=0.8

| Total power per UAV (W) | Sense power (W) | Comm power (W) | Reports | Worst delivered PD |
|---:|---:|---:|---:|---:|
| 0.5 | 0.4 | 0.1 | 5 | 0.442672 |
| 1.0 | 0.8 | 0.2 | 5 | 0.806359 |
| 2.0 | 1.6 | 0.4 | 5 | 0.989933 |

Power is now an active performance lever. The 1 W point is close to the 0.8
boundary, while halving/doubling the same budget changes the worst-target PD by
a large amount.

## Fixed-1-W sensing/communication split

| rho | Sense power (W) | Comm power (W) | Reports | Worst delivered PD | Minimum looks for PD >= 0.8 |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.50 | 0.50 | 5 | 0.554855 | 54 |
| 0.65 | 0.65 | 0.35 | 5 | 0.698867 | 35 |
| 0.80 | 0.80 | 0.20 | 5 | 0.806359 | 25 |
| 0.95 | 0.95 | 0.05 | 1 | 0.673443 | 39 |

The split has a non-monotone effect. Raising rho from 0.5 to 0.8 strengthens
sensing enough to improve detection, but rho=0.95 leaves too little
communication power: only one report remains useful and the worst-target PD
falls. Of the preregistered coarse candidates, rho=0.8 is best; this does not
claim a continuous global optimum.

## Remaining limitation

All UAVs currently share the same total power and rho. The next optimization
should use per-UAV variables `P_j` and `rho_j`, constrained by individual and
fleet energy budgets, and should recompute sensing interference and FBL link
success after every accepted power update.
