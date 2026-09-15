# V1.1 staged optimization log

> Physical-model hold (radar-budget audit): RCS values in this log are in
> square metres. The historical model omitted radar Tx/Rx gain and system loss,
> so the RCS 42.5 transition point implicitly included an unknown hardware
> budget. It remains an algorithmic diagnostic only and must not support a
> small-UAV range claim. See `RADAR_LINK_BUDGET_CALIBRATION.md`.

The replacement physical family is frozen as S1/S2/S3 geometry/RCS presets,
but no longer inherits the 27 dB regression bridge. The earlier S2 MC=100 run
at 27 dB saturated at mean P_D 0.984 for both methods and is retained only as a
conditional regression result. It is not physical validation. The next audit
uses a declared net-gain sensitivity axis without inventing a Tx/Rx split.

The first declared sweep bracketed the S2 transition at 10--20 dB. A refined
12.5/15/17.5 dB pilot with a local-only arm found the most interpretable point
at 15 dB: local-only mean/weak P_D was 0.790/0.467; PD-look-ahead with two
reports was 0.817/0.500; nearest fusion was 0.827/0.433. This establishes a
possible communication contribution but not a Pareto advantage of the proposed
placement. No hardware or headline claim is attached to 15 dB.

## Objective and boundary

The optimization target is not to make a selected difficult scenario pass.
It is to identify a physically interpretable transition regime in which one
local observation per target is insufficient, limited remote reporting restores
the declared weak-target requirement, and the proposed architecture improves
over capacity-constrained nearest fusion under the same hard budgets.

All results below are detector-consistent system-level simulations, not
end-to-end waveform results. They are exploratory because the transition RCS
was localized using the same seed before the candidate point was selected.

## Stage 1: coarse sensing-difficulty scan

With PD-lookahead fusion, `C_local=1`, MC=50, seed 2026, and true erasures:

| Target RCS | `P_D^weak`, `K_remote=0` | `P_D^weak`, `K_remote=8` |
|---:|---:|---:|
| 25.0 | 0.62 | 0.66 |
| 37.5 | 0.72 | 0.78 |
| 50.0 | 0.84 | 0.92 |

This localized the 0.80 transition between RCS 37.5 and 50 without changing
the algorithm or endpoint.

## Stage 2: transition refinement

At MC=100, RCS 42.5 produced `P_D^weak=0.79` at zero reports and 0.86 at an
eight-report cap. RCS 45.0 produced 0.80 and 0.88, respectively. RCS 42.5 was
therefore retained as the exploratory transition point; it was not claimed as
a confirmatory environment.

## Stage 3: initial reporting frontier

The unanchored selector allowed remote observations to consume processing
capacity before every target retained local evidence. This produced a
non-monotone response and failed the mechanism check:

| `K_remote` | Lookahead mean `P_D` | Lookahead `P_D^weak` | Nearest mean `P_D` | Nearest `P_D^weak` |
|---:|---:|---:|---:|---:|
| 0 | 0.948 | 0.79 | 0.944 | 0.77 |
| 2 | 0.934 | 0.77 | 0.937 | 0.76 |
| 4 | 0.926 | 0.76 | 0.938 | 0.78 |
| 6 | 0.943 | 0.78 | 0.946 | 0.78 |
| 8 | 0.956 | 0.86 | 0.949 | 0.81 |

This failure motivated a structural change rather than another parameter
search.

## Stage 4: local-anchor-first selector

`selector.require_local_anchor=true` reserves one detector-best local
observation for each serviceable target before remote allocation. It encodes
the hypothesis that remote evidence complements rather than replaces local
acquisition and introduces no scalar tuning weight.

At RCS 42.5 and MC=100:

| `K_remote` | Mean `P_D` | `P_D^weak` | Mean reports |
|---:|---:|---:|---:|
| 0 | 0.953 | 0.79 | 0.00 |
| 2 | 0.958 | 0.84 | 0.84 |
| 4 | 0.966 | 0.85 | 1.05 |
| 6 | 0.967 | 0.85 | 1.09 |
| 8 | 0.967 | 0.85 | 1.10 |

The response is now monotone in mean detection and saturates after a small
number of reports. The smallest tested cap meeting `P_D^weak>=0.80` is 2.

At `K_remote=2`, capacity-constrained nearest fusion plus the same detector
selector obtained mean `P_D=0.937`, `P_D^weak=0.76`, and used 2.00 reports.
The anchored PD-lookahead candidate obtained 0.958, 0.84, and 0.84 reports.
Paired candidate-minus-nearest differences were:

| Endpoint | Difference | Trial-cluster bootstrap 95% CI |
|---|---:|---:|
| Mean `P_D` | +0.021 | [0.008, 0.035] |
| `P_D^weak` | +0.08 | [-0.01, 0.17] |
| Remote reports | -1.16 | [-1.31, -1.01] |

The mean-detection and reporting gains pass the exploratory gate. The
weak-target point estimate passes the 0.80 requirement, but its paired
superiority interval still crosses zero.

## Candidate protocol for the next independent run

- target RCS: 42.5;
- `C_local=1`, `K_remote=2`;
- receiver capacity 2, fusion-observation capacity 2;
- target-assignment capacity 1;
- `fusion.rule=capacitated_pd_lookahead`;
- `selector.require_local_anchor=true`;
- true report erasure;
- next seed must differ from 2026;
- no algorithm or scenario changes after the next seed is chosen.

Exploratory configuration SHA-256:
`E0515DC5886D0D0689CED1F1AB9E719A8A3DA2CC3F8A3B025DC7ADEA3A6D432C`.
The hash identifies the MC=100 exploratory file; the next confirmation config
will necessarily have a new seed and MC count and must receive its own hash.

Before MC=1000, the next bounded step is an independent-seed MC=200 replication
plus a waveform-level check of the RCS 42.5 operating point. Failure of either
step stops promotion.
