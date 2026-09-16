# Conference Paper Convergence Contract

## Review-revision clarification (2026-09-16)

See [ISAC_REVIEW_REVISION.md](ISAC_REVIEW_REVISION.md) for the 13 unique review
comments and completed changes. The five-page main paper has a separate
reproducibility supplement. Historical V1 erasure meant Gaussian replacement
(a=9) in source 69f3300, not current zero erasure. The manuscript describes the
executed model; the rerun tool explicitly selects gaussian_replacement.

Matched detection below means an unresolved paired difference, not established
equivalence. Approximate solution means a heuristic high-objective set, not a
proved approximation ratio. New proofs cover single-observation deflection
and a fixed-set report-count bound only.

## Frozen one-sentence argument

In prior-assisted multi-UAV OTFS-ISAC sensing, prediction-only target-local
fusion converts many selected observations into local evidence, allowing a
detector-aligned adaptive C2F selector to retain matched detection performance
with substantially less finite-blocklength reporting under the tested serial
MAC, while large belief errors and sparse deployments remain limiting cases.

## Single argument chain

1. The scheduler receives a predicted target belief.
2. Each target is assigned to one target-local fusion UAV.
3. Candidate observations become either local observations or remote reports.
4. One detector-aligned adaptive C2F selector chooses observations under caps.
5. The paper tests whether this architecture preserves detection while reducing
   payload, delay, and fine DD evaluations.

## Canonical terminology

| Canonical term | Meaning | Rejected variants or interpretations |
|---|---|---|
| target-local fusion | target-specific placement based on predicted position | globally optimal fusion |
| fusion UAV $f_q$ | the destination for all soft information about target $q$ | central receiver for all targets |
| local observation | an observation produced at $f_q$, with no report | zero-distance remote packet |
| remote report | a statistic sent from $j$ to $f_q$ under the FBL model | sensing link $i\to j$ |
| adaptive C2F selector | the detector-aligned coarse-frontier and fine-replay procedure | a new globally optimal greedy rule |
| V1 | target-local fusion plus adaptive C2F selection | the V1.6 low-RCS mechanism track |

## Three contribution claims

1. **Architecture:** prediction-only target-local fusion exposes the correct
   target-specific reporting graph and retains local evidence without a packet.
2. **Algorithm:** detector-aligned adaptive C2F solves the declared capped task
   objective approximately and reduces fine DD evaluations relative to full
   refinement.
3. **Evidence:** paired comparison and fusion-rule ablation support lower
   communication cost at matched detection performance.

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| V1 improves over exact-marginal greedy in mean $P_D$ | MC=1000 paired difference 0.0356, 95% CI [0.0311, 0.0401] | supported at the tested configuration |
| V1 matches sensing-SINR detection | paired difference -0.0020, 95% CI [-0.0050, 0.0010] | statistically unresolved; no significant difference detected, not established equivalence |
| V1 reduces payload and serial delay versus sensing-SINR | 0.751 versus 4.595 remote reports at equal observation count | supported; 83.7% reduction |
| target-local placement causes the reporting reduction | controlled fusion-rule ablation with the inner selector fixed | supported in MC=100 diagnostic evidence |
| adaptive C2F reduces waveform-level refinement | 61.3 versus 860.2 fine evaluations in the main experiment | supported; 92.9% reduction |
| nearest-target fusion is jointly optimal | no joint optimization proof or oracle comparison | not claimed |

## Material excluded from this paper

- V1.6 low-RCS rescue controls and physical-headroom oracle hierarchy.
- Coherent eigenvalue oracles, RCS generalization gates, and column generation.
- Universal submodularity, greedy approximation, or global optimality claims.
- Blind initial acquisition, posterior tracker implementation, correlated-LLR
  fusion, parallel MAC latency, and variable-blocklength optimization.

## Editing gate

Any new equation, module, experiment, or claim must directly support one of the
three contribution claims. Otherwise it belongs in an appendix, a separate
mechanism paper, or future work.

## ISAC consistency update (2026-09-16)

- Keep the fixed shared power budget, concurrent sensing leakage, bistatic DD
  geometry, FBL reporting, and independent centered-LLR detector. Add no V1.6 modules.
- V1 detector prediction uses fused H1 mean and variance plus the corrected H0
  threshold, matching `fusion.predicted_pd_for_links`; the deflection-only PD
  mapping belongs to the baseline.
- The implemented communication price multiplies delay in milliseconds. The
  manuscript must use `1000*n/B_c`, not `n/B_c`, with the released lambda.
- Local delivery has zero reporting cost, not zero sensing/processing cost.
- Local-delivery dominance is conditional on nondecreasing detection utility
  and unchanged other deliveries. It proves neither nearest-target optimality
  nor monotonicity of the moment approximation.
- Main fine-evaluation count: 61.278 / 860.244, a 92.88% reduction; separate
  MC=200 full-refinement control: 61.06 / 853.65, a 92.85% reduction.
  The earlier 76.9% value is not supported by these V1 records.
- DD evaluation cost is distinct from greedy scoring and wall-clock runtime.
- The soft-min and quadratic terms remain the tested configuration. Neither
  term is claimed individually necessary without a controlled ablation.

## Bounded theory/algorithm candidate (2026-09-16)

`V1_FUSION_THEORY_AND_ALGORITHM.md` derives a reporting-count lower bound and
an exact fixed-observation fusion subproblem. The experimental method
`proposed_c2f_adaptive_pd_fusion_polish` minimizes remote reports while preserving
each target's initial detector-predicted PD. It is excluded from default methods.
The guarantee is restricted to fixed selected observations and the independent,
orthogonal serial V1 model without coupled fusion capacities. It is not a joint
selection optimum, a true-detection guarantee, or a C2F approximation ratio.
Original headline results do not apply to this candidate.

Target-aware and prediction-based communication reduction have prior art in
DISAC target handover. The novelty assessment must distinguish fixed-CPI
bistatic statistic reporting from trajectory handover and existing backhaul
resource optimization; no priority or first-of-its-kind claim is authorized by
the limited literature screen.

## Probability and detection-theory refinement (2026-09-16)

See `V1_THEORY_PROBABILITY_NOVELTY_UPGRADE.md`. The added constructive
Blackwell comparison establishes optimal-ROC monotonicity only for fixed
observation laws and hypothesis-independent replacement. It does not establish
nearest-target optimality or monotonicity of the implemented CF detector.
The mixture-probability evaluator audits fixed selected sets; it does not replace
default C2F scoring. Its exact-arithmetic brackets are not machine-certified
interval arithmetic. The targeted literature comparison is not exhaustive.
The physical model and three-contribution contract remain unchanged.

## Exhaustive and hard-budget audit (2026-09-16)

`V1_EXACT_BUDGET_REINFORCEMENT.md` documents complete M=3/4,Q=2 enumeration
and nominal-size hard-budget comparisons. Small-instance objective optimality
must not be described as an original-scale detection guarantee. The cap-60
zero-report SINR counterexample must remain visible beside the supportive
cap-12 balanced-SINR comparison. No universal Pareto, non-inferiority, or
lossless-C2F claim follows. Default model, selector and headline remain frozen.
