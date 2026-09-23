# UAV-centric communication, target-count and area scaling experiment

## Protocol

- Trial 32, six UAVs, RCS 0.1, Swerling-II covariance LLR.
- 25 independent looks (53.333 ms), unless the reported minimum-look field is used.
- Each UAV may transmit at most one LLR report; each fusion UAV may receive at
  most two remote reports.
- Fusion starts at the strongest local UAV for each target. Reports are admitted
  by worst-target-PD marginal gain.
- Communication reliability uses the finite-blocklength normal approximation
  with a 640-bit LLR payload.
- Area is the side length of the square deployment region. The 400/500/600 m
  comparison keeps the 400 m per-UAV movement limit fixed.
- Area scaling uses two structural views per target. With more than three
  targets and six UAVs this is infeasible, so the target-count stress test uses
  one structural view per target.

## Finite-blocklength sweep at 600 m, Q=3

| Blocklength | Block latency (ms) | Selected reports | Worst delivered PD |
|---:|---:|---:|---:|
| 128 | 0.0667 | 0 | 0.583979 |
| 256 | 0.1333 | 2 | 0.583979 |
| 512 | 0.2667 | 5 | 0.806359 |
| 1024 | 0.5333 | 5 | 0.807835 |
| 2048 | 1.0667 | 5 | 0.807835 |

The scheduler avoids unreliable links, so the minimum reliability over every
reachable graph edge is not representative of the selected routes. At 128
channel uses no remote report is useful; at 512 channel uses the worst-target
requirement is first recovered.

## Area scaling, Q=3

| Area side (m) | Best-local worst PD | UAV-cooperative worst PD | Reports | Minimum looks for PD >= 0.8 |
|---:|---:|---:|---:|---:|
| 400 | 0.891782 | 0.987717 | 5 | 9 |
| 500 | 0.867522 | 0.983839 | 5 | 9 |
| 600 | 0.583979 | 0.807835 | 5 | 25 |

The 600 m point is a genuine operating boundary: it needs 25 looks, compared
with nine looks at 400 and 500 m.

## Target-count stress at 600 m with six UAVs

| Targets | Views per target | Delivered PD | Worst PD | Reports | Minimum looks for PD >= 0.8 |
|---:|---:|---|---:|---:|---:|
| 3 | 2 | [1.000000, 0.966820, 0.807835] | 0.807835 | 5 | 25 |
| 4 | 1 | [1.000000, 0.824323, 0.827349, 0.796536] | 0.796536 | 6 | 26 |

The Q=4 point narrowly misses the requirement at 25 looks and crosses it at 26
looks (55.467 ms). A Q=6 exact-protection run was stopped after excessive
runtime; the current exact protection search is therefore not scalable enough
to claim a Q=6 performance number. Candidate pruning or target-wise
decomposition is required before that experiment is valid.

## Interpretation boundary

These are single-seed structural experiments, not Monte Carlo confidence
results. Cross-UAV LLR independence and independent packet erasures remain
model assumptions. The next promotion gate is a multi-seed run with correlated
observation/error sensitivity.
