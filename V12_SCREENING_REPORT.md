# V1.2 MC=20 screening report

## Decision

The V1.2 implementation passed its correctness and calibration gates, but the
MC=20 performance evidence is not yet sufficient for promotion. The joint
bundle method produced favorable point estimates against the three principal
non-local baselines, while every paired 95% interval still crossed zero.

## Frozen protocol

- Physical profile: `small-uav-compact-800m`
- UAVs/targets: `M=15`, `Q=10`
- Remote-report budget: `K_remote=2`
- Receiver/fusion processing capacity: 2 observations per UAV
- Fusion assignment capacity: 1 target per UAV
- Local-observation cap: 1 per target
- Mandatory local anchor: disabled
- Communication error: true erasure
- Monte Carlo trials: 20, seed 2026
- No RCS, radar-gain, or geometry adjustment was made after observing results.

## Main screening results

| Method | Mean PD | Weak-target PD | Active PFA | Reports | Observations |
|---|---:|---:|---:|---:|---:|
| Joint bundle CG | 0.495 | 0.500 | 0.0537 | 2.00 | 11.70 |
| Fixed fusion + joint bundle | 0.435 | 0.450 | 0.0535 | 2.00 | 11.65 |
| V1.1 sequential detector-aware C2F | 0.445 | 0.450 | 0.0509 | 2.00 | 10.10 |
| Budgeted sensing-SINR | 0.455 | 0.400 | 0.0507 | 2.00 | 11.40 |
| Local-only joint bundle | 0.490 | 0.450 | 0.0550 | 0.00 | 10.00 |

The paired reference deltas are defined as `Joint bundle CG - comparator`:

| Comparator | Delta PD | Paired 95% CI | Screening interpretation |
|---|---:|---:|---|
| Fixed fusion + joint bundle | +0.060 | [-0.007, +0.127] | favorable, inconclusive |
| V1.1 sequential detector-aware C2F | +0.050 | [-0.022, +0.122] | favorable, inconclusive |
| Budgeted sensing-SINR | +0.040 | [-0.016, +0.096] | favorable, inconclusive |
| Local-only joint bundle | +0.005 | [-0.047, +0.057] | no demonstrated reporting gain |

## Correctness evidence

- All five methods' active-target `P_FA` errors were below 0.01 and every
  trial-cluster interval covered the configured value 0.05.
- Exact-pricing column generation matched the exhaustive joint oracle on all
  four lexicographic components in 20/20 independent small-system scenarios.
- The small oracle used 24.1 columns on average (range 24--26) and 2.05 pricing
  iterations on average (range 2--3).
- The full joint method used 641.15 generated columns and 3.4 pricing
  iterations per trial on average.

## Claim boundary and next gate

Supported now: the implemented RMP is lexicographically correct over its
generated columns; exact small pricing reproduces the exhaustive oracle in the
tested regime; detector calibration and shared random-number semantics are
consistent across methods.

Not supported yet: a statistically resolved detection improvement at
`M=15,Q=10`, or a demonstrated benefit from two remote reports over local-only
bundles. The next authorized experiment is independent MC=200 using this exact
configuration. MC=1000 remains conditional on MC=200 calibration and paired
comparison results; physical parameters must remain frozen.

The executable promotion rule is
`tools/check_joint_bundle_v12_promotion.py`. It requires at least MC=200,
calibrated active-target false alarms for every arm, and a strictly positive
paired 95% lower bound against fixed fusion, V1.1 sequential selection,
budgeted sensing-SINR, and local-only bundles.

## 中文结论

V1.2 已经形成理论、模型、算法、代码、baseline、oracle 和 gate 的完整闭环。
当前 MC=20 中，joint 方法相对固定 fusion、V1.1 sequential 和 SINR baseline
的平均检测率分别高 0.060、0.050 和 0.040，但置信区间仍跨零；因此只能说
“方向有利”，不能说“显著优于”。相对 local-only 仅高 0.005，说明有限回传
是否真正有益仍是下一轮 MC=200 必须回答的问题，不能通过继续调 RCS 获得。
