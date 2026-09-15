# V1.2 RCS sensitivity scan

## Purpose and boundary

This scan tests whether the absolute detection regime and the incremental
value of two remote reports change with target RCS. It is not used to select a
new headline operating point. Geometry, radar gain, report budget, processing
capacities, seed, and all algorithm settings were frozen; only mean RCS varied
over the preregistered logarithmic grid `0.025, 0.05, 0.1, 0.2 m^2`.

Doubling RCS from `0.05` to `0.1 m^2` increases desired echo power by
`10 log10(2) = 3.01 dB`. The corresponding levels are approximately -16.0,
-13.0, -10.0, and -7.0 dBsm.

In the model, RCS is not a tuned objective weight or a master-problem decision
variable. It enters the bistatic gain as
`g^s_ijq = lambda^2 sigma_q / [(4 pi)^3 d_iq^2 d_jq^2]`, which changes the
sensing SINR, the finite-look detector moments, and finally every conditional
bundle coefficient `p_qfb(sigma_q)`. The scan therefore compares four frozen
physical scenarios rather than allowing the optimizer to choose RCS.

## MC=20 screening results

| RCS (m²) | Joint PD | Local-only PD | Fixed-fusion PD | Joint − local | Joint − fixed |
|---:|---:|---:|---:|---:|---:|
| 0.025 | 0.365 | 0.350 | 0.330 | +0.015 | +0.035 |
| 0.050 | 0.495 | 0.490 | 0.435 | +0.005 | +0.060 |
| 0.100 | 0.615 | 0.635 | 0.605 | -0.020 | +0.010 |
| 0.200 | 0.765 | 0.720 | 0.710 | +0.045 | +0.055 |

The paired joint-minus-local 95% intervals were:

- `0.025 m^2`: [-0.014, +0.044]
- `0.050 m^2`: [-0.047, +0.057]
- `0.100 m^2`: [-0.070, +0.030]
- `0.200 m^2`: [-0.024, +0.114]

Thus absolute detection increased with RCS, as expected, but the incremental
remote-report gain was non-monotonic and unresolved at every point. In
particular, `0.1 m^2` improved the operating detection level but did not improve
the joint-versus-local comparison.

## Decision

Keep `0.05 m^2` as the frozen compact-800m headline profile. Use `0.025--0.2
m^2` only as a declared target-class sensitivity axis. A future paper may
report `0.1 m^2` if that value is independently justified by the target class,
but it must not be selected because it produces a preferred algorithm ranking.

The next algorithmic diagnostic remains prediction alignment: compare the
scheduler-predicted marginal value of selected remote reports with their paired
realized detector increment. Increasing RCS is not a substitute for resolving
that question.
