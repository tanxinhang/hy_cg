# V1.6 Mechanism-Stable Theory, Model, and Algorithm

## Release status

V1.6 is a **mechanism-stable release with generalization pending**.  It freezes
the sensing hypothesis, exact-LLR transport semantics, physical-headroom
oracles, candidate-loss decomposition, and falsification tests.  It does not
claim geometry-general low-RCS rescue: the three historical geometries remain
development cases, and G5 requires a pre-specified independent holdout bank.

## One-sentence argument

In a resource-constrained distributed UAV-ISAC system, we determine when
cooperation can and cannot create low-RCS detection headroom by separating
lossless local inference, lossy evidence transport, noncoherent aspect
complementarity, phase-coherent gain, candidate screening, and algorithmic
recovery, supported by executable positive and negative scientific gates and
bounded by a zero-mean Swerling-II observation model and unfinished
cross-geometry validation.

## Terminology ledger

| Canonical term | Definition | Excluded interpretation |
|---|---|---|
| exact local LLR aggregation | a disjoint receiver partition followed by summation of exact per-observation LLRs | arbitrary feature averaging |
| received KL information | discrimination information after an observed, hypothesis-independent erasure | fixed-`P_FA` detection probability |
| best Single-Tx oracle | all declared sensing power assigned to the best one transmitter under the same physical model | an unconstrained hardware baseline |
| noncoherent full oracle | robust information optimum over the complete feasible physical candidate pool | an oracle over a pruned shortlist |
| restricted oracle | exact optimum within the declared screened candidate pool | the full physical optimum |
| coherent oracle | fixed-total-power eigenvalue upper bound under an explicit phase-coherence matrix | a deployable capability of the canonical detector |
| candidate loss | full noncoherent oracle minus restricted oracle, using one metric | optimizer loss |
| algorithm loss | restricted oracle minus the proposed solution, using one metric | candidate-generation loss |
| development geometry | a geometry available during model and algorithm revision | independent generalization evidence |
| holdout geometry | a pre-specified geometry unavailable during model and endpoint tuning | additional within-geometry noise draws |
| lower-tail endpoint | geometry-level 5% quantile or lower CVaR of detection performance | sample minimum |

## 1. Fixed statistical model and boundary

For target `q`, let the global observation set be
\(\mathcal A_q=\biguplus_j\mathcal A_{jq}\), where the disjoint union requires
every physical observation to occur exactly once.  Under hypotheses
\(h\in\{0,1\}\), V1.6's exact aggregation theorem requires the unconditional
factorization

\[
p_h(\mathbf y)=\prod_{a\in\mathcal A_q}p_{h,a}(y_a).
\]

The canonical active detector remains the independent-look, zero-mean
variance-change model

\[
H_0:Y_{a,l}\sim\mathcal{CN}(0,N_a),\qquad
H_1:Y_{a,l}\sim\mathcal{CN}(0,N_a(1+\gamma_a)).
\]

It represents a Swerling-II-like fast-fluctuation abstraction with mean RCS in
the link budget.  It is neither a deterministic-amplitude coherent model nor a
common slow latent-scatterer model.  Conclusions derived from its convex KL
law are explicitly restricted to this hypothesis family.

## 2. Theorem chain and falsification boundary

### 2.1 Exact aggregation theorem

For \(\ell_a=\log[p_{1,a}(Y_a)/p_{0,a}(Y_a)]\), receiver `j` sends
\(\Lambda_j=\sum_{a\in\mathcal A_{jq}}\ell_a\).  Factorization and the disjoint
partition give the sample-wise identity

\[
\Lambda^{\mathrm{local}}=\sum_j\Lambda_j
=\sum_a\ell_a=\Lambda^{\mathrm{central}}.
\]

Thus local and centralized decisions have identical `P_D` and `P_FA` at the
same threshold when reporting is lossless.  The release gate checks the
identity to `1e-12`, not merely equality in expectation.

The theorem must fail after marginalizing a shared nuisance variable unless a
joint sufficient statistic is retained.  For
\(Y_i=\theta+N_i\), \(\theta\sim\mathcal N(0,\tau^2)\), the correct joint LLR
contains \((\sum_iY_i)^2\), whereas the sum of marginal LLRs contains
\(\sum_iY_i^2\).  V1.6 includes this as a mandatory negative test.

### 2.2 Erasure-information theorem

Let `E` be an observed packet-arrival indicator independent of the hypothesis,
with \(\Pr(E=1)=\chi\).  If erased data are represented by a distinct missing
symbol, then

\[
D_{\mathrm{KL}}(P_1^{rx}\Vert P_0^{rx})
=\chi D_{\mathrm{KL}}(P_1^{tx}\Vert P_0^{tx}).
\]

For receiver-local aggregation with local information values \(D_a\), the
aggregate packet preserves at least as much expected received KL as separate
packets if and only if

\[
\chi_{agg}\ge
\frac{\sum_a\chi_aD_a}{\sum_aD_a}.
\]

This is a KL statement, not ROC dominance.  Moreover, an aggregate packet has
delivery-information variance

\[
\chi(1-\chi)(\sum_aD_a)^2,
\]

which is no smaller than the independent-separate value
\(\chi(1-\chi)\sum_aD_a^2\).  Receiver aggregation therefore saves reports but
can increase catastrophic evidence loss; important aggregates require explicit
reliability protection rather than an average-rate argument.

V1.6 therefore includes an importance-aware protection primitive.  After any
mandatory per-packet reliability floors are funded, a linear protection budget
is assigned in descending `local KL / marginal cost` order.  This is the exact
bounded fractional-knapsack solution for maximizing expected received KL.  It
does not claim tail-risk optimality; outage-sensitive extensions must add an
explicit chance or CVaR constraint.

### 2.3 Noncoherent concentration theorem

For \(\gamma_i=\alpha_iP_i\), `L_i` independent looks produce

\[
D_i=L_i f(\alpha_iP_i),\qquad f(x)=x-\log(1+x).
\]

Because \(f(0)=0\), \(f'(x)>0\), and \(f''(x)>0\), under
\(\sum_iP_i=P\) and no scenario uncertainty,

\[
\max_{P_i\ge0,\sum_iP_i=P}\sum_iL_if(\alpha_iP_i)
=\max_iL_if(\alpha_iP).
\]

Hence noncoherent multi-transmitter illumination is not automatically useful.
For two identical views, equal power splitting is strictly worse than the best
single transmitter; at low SINR its information approaches one half of the
single-transmitter value.  This is a must-fail cooperation test, not a defect to
hide with tuning.

### 2.4 Symmetric complementarity theorem

For two scenarios with path coefficients `(a,b)` and `(b,a)`, `a>b>=0`, the
robust optimum is one of only two choices:

\[
W^*=L\max\{f(bP),f(aP/2)+f(bP/2)\}.
\]

Equal cooperative splitting improves the robust KL exactly when

\[
f(aP/2)+f(bP/2)>f(bP).
\]

At low SINR this reduces to \(a/b>\sqrt3\).  Cooperation in this model is
therefore justified by sufficiently strong aspect complementarity, not by the
number of UAVs alone.

### 2.5 Phase-coherent eigenvalue oracle

Let \(\mathbf h\) be the channel vector, \(\mathbf w\) the transmit weights
with \(\lVert\mathbf w\rVert^2\le P\), and
\(R=\mathbb E[\mathbf c(\boldsymbol\epsilon)
\mathbf c(\boldsymbol\epsilon)^H]\).  The expected coherent-power oracle is

\[
S_{coh}^*=P\lambda_{max}(R).
\]

Its gain over the best single transmitter is

\[
G_{coh}=\frac{\lambda_{max}(R)}{\max_i|h_i|^2}\ge1.
\]

Perfect phase alignment yields \(P\sum_i|h_i|^2\); independent uniform phase
makes `R` diagonal and reduces the gain exactly to one.  This oracle is kept
outside the canonical Swerling-II detector and cannot be reported as deployed
performance.

### 2.6 Detectable-RCS ordering

For method `m`, define
\(\sigma_{det}^{(m)}=\inf\{\sigma:g_m(\sigma)\ge\eta\}\).  If a method's
detection curve dominates a baseline for every RCS, its detectable-RCS
threshold cannot be higher.  Finite simulation does not prove curve-wide
dominance; the implementation therefore returns a tested bracket and flags
nonmonotone sweeps.  Final paired inference must require the upper 95% bound of
\(\Delta\sigma_{det}\) to be below zero.

## 3. Model upgrades

### 3.1 Resource-identifiable acquisition modes

Every active observation declares power `P`, independent looks `L`, DD
refinement, and normalized transmit energy `E=P L`.  V1.6 exposes four rescue
controls under the same resource constraints:

| Control | Rescue looks | Rescue power |
|---|---:|---:|
| baseline | 16 | 1.00 |
| looks-only | 144 | 1.00 |
| power-only | 16 | 1.25 |
| combined | 144 | 1.25 |

These controls distinguish architecture gain from added dwell and power.  A
method is not called more efficient unless it improves under equal total
energy; otherwise the result is described as a rescue-configuration gain.

### 3.2 Design and falsification aspects

Column generation retains the design angles `(0,45,90,135)` degrees.  Held-out
detector evaluation can now use a disjoint angle set, such as
`(22.5,67.5,112.5,157.5)`, without modifying design columns.  This prevents a
four-scenario robust optimizer from being evaluated only on the scenarios it
was given.

### 3.3 Geometry endpoints

The 800 m compact geometry remains the mechanism stress case.  A diverse-area
control can be generated through the explicit `area_xy` setting.  Geometry
aggregation now reports the lower 5% quantile and lower CVaR in addition to the
sample minimum and mean.  The minimum remains a fixed-bank stress statistic;
it is not used as a distributional generalization estimator.

## 4. Oracle-first algorithm

V1.6 evaluates the following hierarchy before assigning credit to the proposed
scheduler:

\[
\text{best Single-Tx}
\rightarrow\text{full noncoherent oracle}
\rightarrow\text{restricted oracle}
\rightarrow\text{proposed}
\rightarrow\text{coherent oracle}.
\]

For a common metric `M`, the dashboard reports

\[
H^{NC}=M_{full}-M_{single},\quad
H^C=M_{coh}-M_{full},
\]

\[
L^{cand}=M_{full}-M_{restricted},\quad
L^{alg}=M_{restricted}-M_{proposed}.
\]

The identity
\(M_{full}-M_{proposed}=L^{cand}+L^{alg}\) is an executable invariant.
KL and fixed-`P_FA` detection dashboards are never mixed.  A KL-optimized
oracle is not called a `P_D` upper bound.

Complete-pool branch-and-bound is required on tractable systems.  Large pools
must expose their shortlist scope, incumbent, upper bound, and certificate gap;
they cannot silently inherit the word “oracle”.  The active-system runner now
also exposes `candidate_strategy=full` for tractable falsification cases.

## 5. Five release gates

| Gate | Required evidence | V1.6 status |
|---|---|---|
| G1 Sufficient-statistic | theorem, partition validation, sample-wise equality, shared-latent negative test | pass |
| G2 Transport-information | erasure-KL equality, reliability threshold, aggregate tail-risk accounting | pass |
| G3 Physical-headroom | identical-view must-fail, complementarity two-sided test, coherent limiting cases | theory pass; 15/10 full noncoherent oracle pending |
| G4 Algorithm-recovery | full/restricted decomposition and fixed-`P_FA` large-system oracle | infrastructure pass; large-system evidence pending |
| G5 Generalization | frozen independent geometry bank, paired cluster inference, lower-tail endpoint | pending |

The deterministic gate artifact records sample-wise aggregation error
`1.78e-15`, a nonzero shared-latent counterexample gap, exact erasure linearity,
an identical-split/single information ratio `0.5065`, perfect/30-degree/uniform
phase coherent gains `1.3125/1.1985/1.0000`, and an exact complete-pool
small-system candidate decomposition.

## 6. Stable stress benchmark

The development benchmark remains 15 UAVs, 10 targets, target RCS `0.05 m^2`,
explicit radar net gain `0 dB`, at most five observations per target, and four
design aspects.  In the three development geometries, receiver-local LLR
aggregation reduced mean remote reports from `15.33` to `12.33`, while mean
worst-target `P_D` changed from `0.4568` to `0.4586`.  The perfect-phase
same-receiver oracle reached `0.6170`; even this optimistic result remained far
below `0.97`.  These observations suggest that weak-geometry physical evidence,
not omitted LLR summation, is the dominant unresolved limitation.

No 17.5 dB radar-net-gain result is part of this mechanism release.  Link-budget
sensitivity may be studied separately but cannot be mixed with algorithmic
rescue claims.

## 7. Reproducible commands

Run deterministic scientific gates:

```text
python tools/run_scientific_gates_v16.py --out results_scientific_gates_v16
```

The result of record for this source revision is
`results_scientific_gates_v16/run_2442fcf8a981/scientific_gates_v16.json`.

Run an unseen-aspect development evaluation:

```text
python tools/run_active_system_v15.py --uavs 15 --targets 10 --target-rcs 0.05 --radar-net-gain-db 0 --rescue-looks 144 --rescue-control combined --transport-mode receiver_local_llr --holdout-aspect-angles 22.5 67.5 112.5 157.5 --geometry-role development
```

The equal-resource factorial uses four otherwise identical runs with
`--rescue-control baseline`, `looks_only`, `power_only`, and `combined`.
The final generalization run must use a new seed bank, `--geometry-role
holdout`, at least 30 initial geometry clusters, and a pre-specified stopping
and sample-size rule.  It must not be inspected iteratively to retune V1.6.

## Assumptions and missing evidence

- The canonical observation likelihood assumes unconditional independence;
  slow common RCS, clutter, calibration, or interference latents require a
  joint receiver statistic or an explicit correlated likelihood.
- G4 lacks a large-system fixed-`P_FA` detection oracle over the full physical
  candidate pool; current full-pool certification is deliberately small-scale.
- G5 has not been run.  Thirty geometries are a starting floor, not a guarantee;
  the final count must follow the observed cluster-level variance and minimum
  relevant effect.
- The coherent oracle assumes an externally specified phase-coherence matrix
  and is not an implementation of distributed synchronization.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| exact local LLR aggregation is lossless under factorization and partition conditions | theorem plus sample-wise gate | supported |
| observed hypothesis-independent erasure scales KL by `chi` | theorem plus deterministic sweep | supported |
| aggregation is always no worse in `P_D` | KL does not imply ROC dominance | rejected |
| noncoherent multi-Tx is automatically beneficial | identical-view fixed-power counterexample | rejected |
| strong aspect complementarity can justify robust cooperation | exact symmetric condition and two-sided low-SINR gate | supported within model |
| coherent processing has phase-dependent physical headroom | eigenvalue oracle and limiting-case gates | supported as oracle only |
| V1.6 generalizes across UAV geometries | no frozen holdout bank yet | needs evidence |
