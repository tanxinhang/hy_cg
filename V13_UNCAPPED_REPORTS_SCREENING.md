# V1.3 Uncapped-Report Screening

## Question and controlled change

This experiment asks whether the V1.2 global remote-report cap, rather than
the detector or RCS model, suppresses cooperative gain. Only
`selector.max_remote_reports` changed from `2` to `-1`. RCS remained
`0.05 m^2`; receiver capacity, fusion-processing capacity, target-assignment
capacity, geometry, detector, seed, and all algorithm settings were frozen.

Removing the hard ceiling does not make reporting free. The bundle master
still minimizes report count in its third lexicographic tier, after locking
worst and total detection deficit.

## Nominal bundle result, MC=20

| Setting | P_D | Remote reports | Observations | Median D |
|---|---:|---:|---:|---:|
| V1.2, `K_report=2` | 0.495 | 2.0 | 11.7 | 1.84 |
| Report cap removed | 0.550 | 9.7 | 17.0 | 3.18 |
| Local-only control | 0.490 | 0.0 | 10.0 | 1.78 |

The uncapped nominal method improved absolute detection by `+0.055` relative
to the same-seed `K_report=2` screening and used 9.7 reports on average. Its
paired advantage over local-only was `+0.060`, with 95% CI
`[-0.0042,+0.1242]`; this screening interval still crosses zero.

## RCS-robust result with report cap removed, MC=20

| Method | P_D | Weak-target P_D | Reports | Median D |
|---|---:|---:|---:|---:|
| RCS-robust bundle CG | 0.575 | 0.450 | 11.4 | 4.34 |
| Nominal bundle CG | 0.550 | 0.500 | 9.7 | 3.18 |
| Fixed-fusion bundle | 0.575 | 0.500 | 10.75 | 4.15 |
| Local-only bundle | 0.490 | 0.450 | 0 | 1.78 |

Paired RCS-robust differences were:

- versus nominal bundle CG: `+0.025`, 95% CI `[-0.0219,+0.0719]`;
- versus fixed-fusion bundle: approximately `0`, 95% CI
  `[-0.0620,+0.0620]`;
- versus local-only bundle: `+0.085`, 95% CI
  `[+0.0179,+0.1521]`.

Thus removing the report cap exposed a cooperative gain relative to local-only
sensing. It did not establish an advantage of robust over nominal bundle
pricing or of joint over fixed fusion. Weak-target detection also remained at
0.45 for the robust method, so average-PD improvement must not be described as
resolved fairness.

## Bottleneck transfer

After the report cap was removed:

- RCS-robust receiver-capacity binding rate: `1.00`;
- fusion-processing binding rate: `1.00`;
- target-assignment binding rate: `1.00`.

The system therefore moved from a report-limited regime to a
processing-limited regime. The next controlled experiment should relax
receiver and fusion capacity together only if they represent scalable compute
resources; RCS, geometry, detector settings, and the now-uncapped report policy
must remain frozen.

## Decision

- Remove the hard report-count cap from the V1.3 candidate: **YES**.
- Retain reports as a lexicographic overhead objective: **YES**.
- Promote the RCS-robust method over nominal/fixed fusion: **NO-GO at MC=20**.
- Treat receiver/fusion processing as the next diagnosed bottleneck: **YES**.
