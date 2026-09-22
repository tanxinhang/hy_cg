# Heterogeneous per-UAV power optimization

## Constraint and objective

Each UAV obeys the hard peak constraint

`0 < P_sense[j] + P_comm[j] <= 1 W`.

The comparison uses the same 4.8 W fleet budget. The optimizer maximizes a
weighted max-min surrogate: target sensing evidence is linked to summed
bistatic target gain, while reporting evidence is linked to each UAV's
reachable communication gain. Fleet sensing and communication budgets are
3.84 W and 0.96 W. This is one linear-program solve, not a power grid scan.

## Trial-32, 600 m result

| Policy | Per-UAV total powers (W) | Worst delivered PD | Looks needed for PD >= 0.8 |
|---|---|---:|---:|
| Uniform | [0.8, 0.8, 0.8, 0.8, 0.8, 0.8] | 0.688733 | 36 |
| Geometry/link max-min | [0.830, 0.261, 1.000, 1.000, 0.708, 1.000] | 0.728809 | 32 |

The heterogeneous allocation improves the bottleneck without adding fleet
energy or violating the 1 W per-UAV cap. It remains a geometry/link surrogate;
the next stage should update powers using marginal changes in the actual
delivered-PD objective after the selected report graph is known.

## One-step TP-UIC/coordination closed-loop check

The TP-UIC active set from the heterogeneous pass was `[3,5,6]` in one-based
UAV numbering. A coordination-aware update therefore moved sensing power away
from inactive sensing transmitters UAV 1, 2 and 4 and assigned those nodes
communication power for LLR reporting. The same 4.8 W fleet budget and strict
1 W per-UAV cap were retained.

| Policy | Per-UAV total powers (W) | Worst delivered PD | Looks needed for PD >= 0.8 |
|---|---|---:|---:|
| Uniform | [0.8, 0.8, 0.8, 0.8, 0.8, 0.8] | 0.688733 | 36 |
| Geometry/link max-min | [0.830, 0.261, 1.000, 1.000, 0.708, 1.000] | 0.728809 | 32 |
| TP-UIC/coordination update | [0.6, 0.6, 1.0, 0.6, 1.0, 1.0] | 0.844116 | 22 |

The closed-loop candidate is accepted because the full TP-UIC, FBL reporting
and delivered-PD recomputation improves the true bottleneck objective. This is
one accepted alternating-optimization step, not yet a convergence or global
optimality claim.

## Trust-region continuation

Starting from the accepted closed-loop point, a second update moved another
0.099 W from communication to sensing on active TP-UIC transmitters. Full
recomputation increased the worst delivered PD from 0.844116 to 0.878642 and
reduced the required looks from 22 to 19, so the update was accepted. The
coordinator also reduced the selected report count from five to four.

A third, smaller 0.05 W update on UAV 5 crossed an FBL communication cliff:
target 3 fell to 0.787679 and required looks increased to 27. This update was
rejected by the monotonic acceptance rule. The retained operating point is
therefore round 2, not round 3.

| Iteration | Worst delivered PD | Required looks | Decision |
|---:|---:|---:|---|
| Uniform baseline | 0.688733 | 36 | baseline |
| Geometry/link initialization | 0.728809 | 32 | accepted |
| Joint round 1 | 0.844116 | 22 | accepted |
| Joint round 2 | 0.878642 | 19 | accepted |
| Joint round 3 | 0.787679 | 27 | rejected |
