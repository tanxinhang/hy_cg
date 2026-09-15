# V1.2 independent MC=200 gate report

## Decision

V1.2 passed detector calibration and three principal paired comparisons, but
it did not pass the preregistered comparison with the local-only bundle model.
The decision therefore remains **NO-GO for promotion** and **NO-GO for an
MC=1000 confirmatory run**.

## Frozen protocol

The run reused the MC=20 screening configuration without changing geometry,
RCS, radar gain, detection requirement, candidate limits, or hard capacities.
It used the `small-uav-compact-800m` profile, `M=15`, `Q=10`, two remote
reports, receiver/fusion capacity two, fusion target capacity one, true
erasure, seed 2026, and 200 independent trials. Four workers were used after a
serial-versus-parallel determinism regression passed.

## Results

| Method | Mean PD | Weak-target PD | Active PFA | Reports | Observations |
|---|---:|---:|---:|---:|---:|
| Joint bundle CG | 0.4705 | 0.465 | 0.0514 | 2.00 | 11.665 |
| Fixed fusion + joint bundle | 0.4410 | 0.425 | 0.0511 | 2.00 | 11.720 |
| V1.1 sequential detector-aware C2F | 0.4155 | 0.330 | 0.0512 | 2.00 | 10.075 |
| Budgeted sensing-SINR | 0.4250 | 0.300 | 0.0503 | 2.00 | 11.475 |
| Local-only joint bundle | 0.4585 | 0.400 | 0.0516 | 0.00 | 10.000 |

Paired differences are `Joint bundle CG - comparator`:

| Comparator | Delta PD | Paired 95% CI | Gate |
|---|---:|---:|---|
| Fixed fusion + joint bundle | +0.0295 | [+0.0082, +0.0508] | PASS |
| V1.1 sequential detector-aware C2F | +0.0550 | [+0.0327, +0.0773] | PASS |
| Budgeted sensing-SINR | +0.0455 | [+0.0232, +0.0678] | PASS |
| Local-only joint bundle | +0.0120 | [-0.0018, +0.0258] | FAIL |

Every method's active-target false-alarm estimate was within 0.0016 of 0.05,
and every trial-cluster interval covered 0.05. The joint method generated
643.575 columns and used 3.37 pricing iterations per trial on average, close to
the MC=20 screening values; no column explosion was observed.

## Interpretation

The positive fixed-fusion comparison isolates a fusion-assignment contribution
under the same joint bundle selector. The positive V1.1 comparison shows that
closing assignment, selection, and shared allocation in one master improved
the paired detector outcome under this profile. The sensing-SINR comparison
supports detector-aware bundle valuation. These statements are supported only
for the frozen MC=200 protocol.

The local-only comparison does not yet establish that two remote reports add
reliable mean detection value. Its point estimate is positive, and weak-target
PD increased from 0.400 to 0.465, but the preregistered paired mean-PD interval
crossed zero. This is a substantive model finding rather than a reason to tune
the scenario.

## Next optimization step

Before any larger Monte Carlo run, audit the scheduler-predicted incremental
PD of the two selected remote reports against their paired realized increment.
The audit should separate:

1. fusion reassignment gains already available to local-only bundles;
2. remote-report gains conditional on the same fusion assignment;
3. belief-to-truth delay--Doppler capture errors;
4. Cornish--Fisher scheduling predictions versus calibrated-detector outcomes;
5. trials in which both report and receiver/fusion capacities bind.

Only a model-derived correction to a demonstrated prediction mismatch should
change the algorithm. If the predictor is calibrated and the remote increment
remains negligible, the correct conclusion is that cooperation is not needed
at `K_remote=2` for this physical profile.

## 中文结论

MC=200 已经证明 joint bundle 相对固定 fusion、V1.1 sequential 和 SINR
baseline 的优势不是 MC=20 偶然波动，但尚未证明两个 remote report 相对
local-only 的平均检测收益。当前唯一失败项为 local-only paired CI 下界
-0.0018。因此不能进入 MC=1000，也不能修改 RCS、净增益或 gate；下一步应
先核对“调度预测的 remote 增益”与“真实 paired remote 增益”是否一致。
