# Capacitated target-specific fusion V1.1

> **归档提示（2026-09-18）**：本文引用的 `results_*` 产物已移入
> `_archive/2026-09-18/`；命令行示例里的 `--out` 目录仍写作历史原名
> （重跑时依旧输出到该名）。

> Radar-budget qualification: all exploratory RCS values are in \(m^2\). The
> original sensing equation omitted explicit radar antenna gain and system
> loss. Results at RCS 42.5 are therefore transition-point diagnostics, not a
> validated small-UAV detection-range claim. The repaired model and calibration
> hold are documented in `RADAR_LINK_BUDGET_CALIBRATION.md`.

## One-sentence argument

In prior-assisted multi-UAV OTFS-ISAC sensing, V1.1 replaces a hand-priced
reporting trade-off and nearest-target fusion heuristic with a capacity-limited
target--fusion assignment and detector-aligned observation selection under
explicit receiver-processing, fusion-processing, and remote-report budgets;
its benefit must be established by budget-response experiments and a
small-system joint oracle, while waveform, power, blocklength, and trajectory
remain fixed.

## Terminology ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| target--fusion assignment | binary decision `y[q,f]` assigning target `q` to fusion UAV `f` | replaces “nearest-target proposed method” |
| observation selection | binary decision `x[i,j,q]` admitting the bistatic observation `i -> q -> j` | not called link selection when the reporting leg is meant |
| observation routing | binary decision `z[i,j,q,f]` indicating that a selected observation is fused at `f` | stored sparsely as `selected[q]` plus `ReportingPlan.f_q` |
| remote report | selected observation with `j != f_q` | one 640-bit packet in the fixed-blocklength model |
| receiver-processing budget | maximum selected observations processed by receiver UAV `j` | hard system constraint |
| fusion-processing budget | maximum selected observations fused by UAV `f` | hard system constraint |
| remote-report budget | maximum inter-UAV reports | hard system constraint, not a tuned weight |
| adaptive C2F | adaptive coarse-to-fine DD refinement | computational accelerator of the inner selector |
| PD-lookahead assignment | target--fusion utility obtained by replaying detector-PD selection under pair-level capacity allowances | experimental redesign; not the V1.1 preset |
| local anchor | one detector-best local observation retained before remote allocation | experimental structural constraint; no fitted weight |

## Optimization contract

The primary formulation is lexicographic rather than a weighted sum:

\[
\operatorname{lexmin}_{\mathbf x,\mathbf y,\mathbf z}
\left(
\max_q [P_D^{\rm req}-\widehat P_{D,q}]_+,
\sum_q [P_D^{\rm req}-\widehat P_{D,q}]_+,
R_{\rm remote},
C_{\rm process}
\right).
\]

Here, the selector/oracle design point remains
$P_D^{\rm req}=0.95$. The separate empirical operating-point rule uses
$P_D^{\rm weak,req}=0.80$; the two thresholds must not be conflated.

The equivalent reliability-first view maximizes the worst-target detection
probability under fixed budgets. The implementation uses the following hard
constraints:

\[
\sum_f y_{qf}=1,\qquad
\sum_q y_{qf}\le C_f^{\rm target},
\]

\[
\sum_{i,q}x_{ijq}\le C_j^{\rm rx},\qquad
\sum_{i,j,q}z_{ijqf}\le C_f^{\rm fusion},
\]

\[
R_{\rm remote}=\sum_{i,j,q}\sum_{f\ne j}z_{ijqf}
\le K_{\rm report}.
\]

The sparse runtime representation enforces
`z[i,j,q,f] = 1` exactly when `(i,j)` appears in `selected[q]` and
`ReportingPlan.f_q[q] == f`; this avoids constructing a dense four-dimensional
binary tensor without changing the decision semantics.

With a fixed 640-bit packet and fixed blocklength, payload and serial latency
are deterministic multiples of `R_remote`. They are therefore reported as two
metrics but are not treated as independent optimization dimensions.

## Implemented decomposition

1. **Slow-timescale assignment.** `fusion.rule=capacitated_value` evaluates
   fixed target--fusion utilities from feasible singleton detector deflections,
   normalizes them within each target, maximizes the minimum retained utility,
   and then maximizes their sum within that bottleneck tier. The fixed-utility
   capacitated assignment subproblem is exact; this does **not** make the
   decomposed joint sensing problem exact. This layer has no scalar
   communication price.
2. **Fast-timescale selection.** Adaptive C2F refines promising DD candidates
   and greedily commits the largest detector-predicted marginal gain that
   remains feasible under all hard budgets.
3. **Small-system oracle.** `joint_fusion_selection_oracle()` enumerates both
   fusion assignments and observation subsets and applies the same
   reliability-first lexicographic objective. Its companion evaluators report
   the four objective-component gaps without a weighted scalarization. Both
   method and oracle must receive the same belief-side information.

An experimental rule, `fusion.rule=capacitated_pd_lookahead`, replaces the
singleton-deflection proxy with a detector-PD replay for every fixed
target--fusion pair. It uses an equal-share remote allowance only inside the
assignment proxy; the downstream selector still enforces the exact global
budgets. This candidate is intentionally not promoted into the named preset.

The named preset is `capacitated-target-fusion-v1.1`. It disables
`selector.use_delay_price` and sets `lambda_c=0`. Numerical budgets are not
embedded in the preset because they must be declared as system conditions in
each experiment.

## Evidence plan

The first diagnostic experiment is the complete response surface

\[
C_{\rm local}\in\{0,1,2,\infty\},\qquad
K_{\rm report}\in\{0,1,2,4,8\}.
\]

Run it with:

```powershell
python -m isac_sim `
  --preset capacitated-target-fusion-v1.1 `
  --mode resource-budget-surface `
  --mc 100 --workers 8 `
  --out results_target_local_v11
```

Every grid point must be reported. The grid is a resource-response surface,
not a search from which one favourable setting is selected. The formal remote
budget rule is the smallest grid value attaining the predeclared weak-target
requirement $P_D^{\rm weak}\geq0.80$. If no value qualifies, no operating point is frozen and a
confirmatory run is not launched.

The five main evidence blocks should be:

1. remote-report budget versus mean and worst-target detection;
2. local processing versus remote reporting;
3. nearest-target, max-in-rate, capacitated-value, and joint-oracle fusion;
4. prediction-mismatch boundary;
5. fine DD evaluations or runtime versus problem size.

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| V1.1 removes the artificial scalar communication price | preset and selector configuration | supported by code and tests |
| receiver, fusion, and reporting resources are hard constraints | selection feasibility checks | supported by code and tests |
| target assignment respects per-UAV capacity | exact fixed-utility capacitated subproblem | supported by code and tests |
| weak-target budget rule freezes `C_local=1, K_remote=8` | corrected true-erasure MC=100 sweep | supported as diagnostic protocol selection |
| V1.1 exposes the full local-processing/reporting response surface | 4x5 MC=100 grid | historical grid superseded; corrected full grid not yet rerun |
| detector-aligned selection improves over budget-matched SINR selection | diagnostic paired factorial | supported exploratorily |
| capacitated fusion improves over capacity-constrained nearest fusion | diagnostic paired factorial | not supported; gate failed |
| the decomposed method is close to the joint optimum | small-system joint-oracle gap | needs evidence |
| adaptive C2F reduces computation without harming reliability | matched-objective complexity study | partially supported by V1; must be rerun under V1.1 |

## Scope boundary

V1.1 keeps UAV positions fixed within a CPI and fixes transmit power, waveform,
blocklength, number of looks, and continuous sensing. Active illumination,
power allocation, adaptive blocklength, trajectory optimization, and robust
chance constraints are outside this revision. The corrected communication
model distinguishes a true erasure (zero contribution) from Gaussian error
replacement. The named V1.1 preset uses true erasures.

The corrected weak-target pilot froze the formal point at
`C_local=1, K_remote=8`: this was the smallest tested remote budget attaining
$P_D^{\rm weak}\geq0.80$. The subsequent diagnostic gate did not support
promotion of V1.1: at MC=100 the proposed
capacitated-fusion/detector cell was statistically tied with the
capacity-constrained-nearest/detector cell in mean detection and had a lower
weak-target point estimate. Consequently MC=1000 was not run, and the LaTeX mainline remains the frozen
`target-local-v1` release. See `V11_GATE_REPORT.md`.

The later PD-lookahead redesign improved the diagnostic weak-target endpoint
at `K_remote=8` relative to capacity-constrained nearest fusion, but consumed
more reports. Under its own budget sweep it already met the 0.80 weak-target
requirement at `K_remote=0`, where its paired advantage over nearest fusion was
not significant. It is therefore a promising mechanism probe rather than a
validated successor.

A subsequent local-anchor-first revision removed the non-monotone reporting
frontier at the exploratory RCS 42.5 transition point. At `K_remote=2`, it
achieved mean `P_D=0.958` and `P_D^weak=0.84` with 0.84 reports, versus 0.937,
0.76, and 2.00 for capacity-constrained nearest fusion. This is a promising
exploratory result selected after a staged scenario scan, not confirmation.
See `V11_OPTIMIZATION_LOG.md` for the full audit trail and stop rules.
