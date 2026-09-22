# Paired scientific validation protocol V1

## Implemented safeguards

1. A frozen `.npz` scenario stores truth geometry, belief mean/covariance, and
   every `BaseGains` channel field. Paired arms reload it without new draws.
2. Mainline formation uses belief means and requires one declared displacement
   cap `D_max`. Every UAV independently satisfies `||p_new - p_old|| <= D_max`.
   Truth-driven formation remains explicitly labelled `truth_oracle`.
3. Receiver vectors are fused using a covariance estimated from joint training
   statistics, rather than a scalar design-effect factor.
4. Linear weights and the false-alarm threshold are fitted on training samples.
   Test H0/H1 samples are used only for the reported PFA, PD, and Wilson CI.
5. Train/test splitting is deterministic, scene-level, and disjoint. Multiple
   input files allow independent base seeds to be pooled without confusing
   repeated trial indices.

The analytic matrix runner no longer emits the scalar design-effect correlation
sweep. Its PD output is always labelled `screening_only`. When AO optimizes
delivered probability, the result is stored as `optimized_objective`; the
`optimized_information` field is null rather than mislabelling probability as
information. A loaded frozen snapshot cannot be silently combined with another
formation transform, and its M/Q dimensions must match the requested run.

## Smoke check (not a performance claim)

Two seeds (`2026`, `303`), four scenes per seed, eight noise realisations per
scene, and receivers 0/1/2 were used only to check the pipeline. Individual
seed summaries changed direction: seed 2026 had conditional AUC about 0.45,
while seed 303 had about 0.58. In the held-out joint evaluation, target-level
test sample sizes were only 8 or 16; observed PD was 0 or 0.25 and intervals
were correspondingly broad. This establishes that the earlier near-one
analytic PD is not empirically supported by this smoke sample.

No parameter is promoted from this check. A formal run must preregister at
least the scene count, noise realisations, seeds, PFA, movement budget, and
primary endpoint before execution.

## Formal-run recommendation

- At least 4 independent base seeds and 50 scenes per seed.
- At least 1,000 H0 and 1,000 H1 realisations in each held-out target/arm cell
  when estimating PD near a 0.05 false-alarm operating point.
- Fix a training/test split before inspecting arm results.
- Report per-seed and worst-seed results, empirical PFA, PD, Wilson interval,
  movement feasibility, and the learned joint correlation matrix.
- Treat truth-oracle formation and perfect-channel cancellation only as upper
  bounds, never as achieved methods.

The motion model is intentionally a bounded endpoint planner. Turn rate,
acceleration, collision avoidance, and no-fly regions are outside the current
system scope and are not optimization variables.
