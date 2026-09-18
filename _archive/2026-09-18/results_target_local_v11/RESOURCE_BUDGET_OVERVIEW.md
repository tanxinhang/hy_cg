# V1.1 resource-budget response: superseded exploratory MC=100

> Protocol warning: this grid was produced before the historical
> `comm_error_model="erasure"` implementation was disambiguated. Its numerical
> behavior was Gaussian error replacement, not a true dropped report. V1.1 now
> explicitly uses true erasure semantics, so this grid must not be used to
> freeze a formal operating point or support a current V1.1 claim.

This run evaluates every point in

```text
local observations per target = {0, 1, 2, unbounded}
remote-report budget           = {0, 1, 2, 4, 8}
```

with seed 2026, 15 UAVs, 10 targets, and the
`capacitated-target-fusion-v1.1` preset. The scalar communication price is
disabled at every point. All 20 rows were produced, and no mean remote-report
count exceeded its declared hard budget.

## Main diagnostic

| Local cap | Remote budget | Mean `P_D` | 95% Wilson CI | Worst-target `P_D` | Mean observations | Mean remote reports |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 0.787 | [0.761, 0.811] | 0.13 | 8.00 | 8.00 |
| 1 | 0 | 0.957 | [0.943, 0.968] | 0.92 | 10.00 | 0.00 |
| 1 | 8 | 0.976 | [0.965, 0.984] | 0.96 | 11.02 | 7.02 |
| 2 | 0 | 0.969 | [0.956, 0.978] | 0.94 | 11.03 | 0.00 |
| unbounded | 0 | 0.975 | [0.963, 0.983] | 0.95 | 13.25 | 0.00 |
| unbounded | 8 | 0.980 | [0.969, 0.987] | 0.96 | 12.29 | 6.84 |

Under the superseded protocol, the result matched the anticipated **case A** mechanism: one local observation
per target retained most of the unbounded-local performance. At zero remote
budget, reducing the local cap from unbounded to one changed mean `P_D` from
0.975 to 0.957 and worst-target `P_D` from 0.95 to 0.92. Thus, the headline
detection level did not require unlimited local evidence.

The no-local row exposes the complementary communication bottleneck. With only
eight remote reports for ten targets, the mean active-target ratio was 0.80 and
mean `P_D` was 0.787. This is consistent with a hard service limit rather than
a tuned cost trade-off.

## Interpretation boundary

- The communication-failure semantics are obsolete for the current V1.1 preset.
- MC=100 is exploratory. The values are not a confirmatory comparison against
  nearest-target, max-in-rate, or other fusion rules.
- The decomposed greedy selector is not an exact optimizer. Small
  non-monotonic changes at intermediate report budgets can arise from greedy
  path changes and Monte Carlo uncertainty even though the feasible sets are
  nested.
- A budget point must be declared before a high-MC paired comparison. It must
  not be selected after inspecting this grid.
- The next decisive test is the budget-matched fusion comparison plus the
  small-system `joint_fusion_selection_oracle()` gap.

Source data: `resource-budget-surface/resource-budget-surface.csv`.
