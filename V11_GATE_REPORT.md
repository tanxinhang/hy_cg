# V1.1 diagnostic gate report

## Decision

**No-go for V1.1 promotion.** The corrected operating point is
`C_local=1, K_remote=8`, obtained using the weak-target requirement 0.80 rather
than the selector/oracle design point 0.95. The diagnostic factorial at that
point did not establish a fusion-placement benefit. MC=1000 was therefore not
launched, the LaTeX mainline remains unchanged, and `target-local-v1` remains
the unique headline release.

The decision follows the preregistered logic: first identify the smallest
remote-report budget that attains the weak-target requirement; then require
capacitated fusion to improve over a capacity-constrained nearest-fusion
control under identical budgets. The weak-target requirement is
$P_D^{weak}\geq0.80$; $P_D^{req}=0.95$ remains only the target-wise design
point inside selection and the joint oracle.

## Corrected weak-target budget sweep

Protocol: MC=100, seed 2026, true report erasure, `C_local=1`, receiver
capacity 2, fusion-observation capacity 2, and target-assignment capacity 1.

| `K_remote` | Mean `P_D` | `P_D^weak` | Weak-target cluster 95% CI | Mean reports |
|---:|---:|---:|---:|---:|
| 0 | 0.947 | 0.76 | [0.68, 0.84] | 0.00 |
| 1 | 0.943 | 0.77 | [0.69, 0.85] | 1.00 |
| 2 | 0.937 | 0.76 | [0.67, 0.84] | 2.00 |
| 4 | 0.938 | 0.75 | [0.66, 0.83] | 3.99 |
| 8 | 0.964 | 0.85 | [0.77, 0.92] | 7.18 |

Using the preregistered point-estimate rule, the smallest tested budget with
$P_D^{weak}\geq0.80$ is therefore $K_r^\star=8$. The intervals are reported
to show pilot uncertainty; they were not substituted post hoc for the stated
budget-selection rule.

## Corrected diagnostic factorial

Protocol: MC=100, seed 2026, common deterministic geometry/channel streams,
15 UAVs, 10 targets, true report erasure, `C_local=1`, `K_remote=8`, receiver
capacity 2, fusion-observation capacity 2, and target-assignment capacity 1.
All cells face the same external budgets.

| Cell | Fusion | Selection | Mean `P_D` | `P_D^weak` | Reports | Observations |
|---|---|---|---:|---:|---:|---:|
| NS | capacity-constrained nearest | budgeted sensing SINR | 0.918 | 0.65 | 8.00 | 16.49 |
| ND | capacity-constrained nearest | detector-aligned | 0.965 | 0.88 | 5.45 | 10.83 |
| CS | capacitated relative-value | budgeted sensing SINR | 0.893 | 0.58 | 8.00 | 15.24 |
| CD | capacitated relative-value | detector-aligned | 0.964 | 0.85 | 7.18 | 10.58 |

Primary paired contrasts, reported as CD minus comparator with trial-cluster
bootstrap 95% confidence intervals:

| Contrast | Mean `P_D` difference | `P_D^weak` difference | Interpretation |
|---|---:|---:|---|
| CD - ND | -0.001 [-0.010, 0.008] | -0.03 [-0.09, 0.03] | no fusion-placement gain |
| CD - CS | +0.071 [0.054, 0.089] | +0.27 [0.17, 0.37] | detector selection gain |
| CD - NS | +0.046 [0.030, 0.061] | +0.20 [0.12, 0.29] | combined-cell gain, not attributable to fusion |

The factorial interaction was +0.024 for mean detection and +0.04 for the weak
target. These are diagnostic point estimates, not confirmatory claims.

## Weak-target and capacity audit

The weak target is fixed before method-specific selection as the target with
the lowest best feasible sensing SINR under the scheduler's belief. No true
position, realized RCS, packet outcome, or detector outcome is used. For CD,
the weak-target rescue rate relative to the local-only control was 0.458.

The hard capacities were operational rather than decorative. In CD, receiver
and target-assignment capacities bound in 100% of trials, fusion-observation
capacity in 46%, and report capacity in 55%. ND showed binding rates of 100%,
100%, 61%, and 10%, respectively. Target assignment counts target hypotheses;
fusion processing counts admitted observations, so the two capacities encode
different workloads.

## Audit closure

- A budget-matched sensing-SINR baseline is implemented and does not copy the
  proposed method's selected count.
- The joint oracle uses the same-information-set lexicographic vector
  `(d_max, d_sum, remote reports, processing load)` and exports component-wise
  gaps without scalar weights.
- True erasure and Gaussian error replacement are separate communication
  models; V1.1 uses true erasure.
- The detector look count is denoted `K_look=16` in the system model.
- Trial-cluster confidence intervals are exported for the primary endpoints.

The small-system oracle machinery is ready, but a larger oracle study is not
used to rescue a failed architecture gate. The supported conclusion is that
the detector-aligned observation selector is useful under hard budgets; the
tested capacitated fusion-placement heuristic is not established as an
improvement over capacity-constrained nearest-target fusion.

## Experimental PD-lookahead redesign

To test whether the failed placement result came from the fixed singleton
utility, an additional rule replayed detector-PD selection for every candidate
target--fusion pair and then maximized minimum followed by total predicted PD
under the target-assignment capacity. This introduced no fitted scalar weight.

At `C_local=1, K_remote=8`, MC=100 and seed 2026, the redesign achieved mean
`P_D=0.972`, `P_D^weak=0.92`, 6.37 reports, and 10.67 observations. Relative
to capacity-constrained nearest fusion plus the same detector selector, its
paired differences were:

| Endpoint | PD-lookahead minus nearest | Trial-cluster bootstrap 95% CI |
|---|---:|---:|
| Mean `P_D` | +0.007 | [-0.002, 0.016] |
| `P_D^weak` | +0.04 | [0.01, 0.08] |
| Remote reports | +0.92 | [0.58, 1.26] |
| Selected observations | -0.16 | [-0.26, -0.07] |

Thus the weak-target improvement is promising, but it is partly purchased by
greater report use and the mean-detection interval includes zero. Moreover,
under the redesign's own budget rule, `K_remote=0` already yielded
`P_D^weak=0.82` versus 0.80 for nearest fusion. Their paired differences at
zero reports were only +0.001 [-0.005, 0.008] in mean detection and +0.02
[-0.03, 0.07] for the weak target. Consequently the minimum qualifying budget
for the redesigned method is zero, and the original remote-rescue narrative
is not supported at the declared 0.80 requirement.

The redesign is retained as an experimental rule but does not pass the full
promotion standard: it shows a weak-target gain at the eight-report cap, not a
clear Pareto improvement or a necessary remote-cooperation regime.

### Local-anchor follow-up

The next structural revision retained one local observation per target before
remote allocation. At the exploratory RCS 42.5 transition point this removed
the non-monotone frontier and produced a candidate point at `K_remote=2`:
mean `P_D=0.958`, `P_D^weak=0.84`, and 0.84 reports. The matched nearest-fusion
control obtained 0.937, 0.76, and 2.00. Mean-detection and report-count paired
intervals favored the candidate, while the weak-target superiority interval
still crossed zero. This reopens the architecture as a candidate but does not
overturn the no-go release decision without independent replication and
waveform-level validation. See `V11_OPTIMIZATION_LOG.md`.

Diagnostic source data and figure are under
`results_target_local_v11/v11-factorial/`.
