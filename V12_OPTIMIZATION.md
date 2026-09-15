# V1.2 correctness and joint-optimization layer

This revision follows the `122d99d` audit in two deliberately separated
layers.

## Correctness patch

- The exact oracle now enforces
  `selector.max_local_observations_per_target`, so its feasible set matches the
  production selector before any optimality gap is reported.
- Every method uses `calibrated_fused_threshold()` for its selected detector.
  Under true erasure, threshold calibration is no longer a feature attached to
  one proposed-method label.
- Monte Carlo detection uses keyed streams indexed by
  `(seed, trial, target, transmitter, receiver, hypothesis, false-alarm index)`.
  A physical observation shared by two methods therefore sees the same packet
  and local-statistic primitives even if their selected-set sizes differ.
- `tools/check_v12_pfa.py` gates the realised active-target false-alarm rate for
  every requested method, including both absolute error and trial-cluster
  interval coverage.

## Restricted bundle master

`isac_sim.bundle_master` introduces a binary column for each
target--fusion--observation bundle. The master jointly enforces:

- one fusion/bundle choice per target;
- target capacity per fusion UAV;
- receiver processing capacity;
- fusion processing capacity;
- one shared remote-report pool;
- total and per-target observation caps.

It performs four sequential MILP solves, locking each completed tier before
the next: worst detection deficit, total detection deficit, remote reports,
then processing load. Thus the implementation solves the stated
lexicographic objective directly and does not reintroduce a scalar resource
price.

The scalable path now performs lexicographic LP column generation for the two
reliability tiers. It uses exact pricing for small shortlisted spaces and
multi-start detector-marginal greedy pricing otherwise, then solves the final
four-tier integer RMP exactly over all generated columns. Coarse tables form
the shortlist and refined delay--Doppler tables value the bundle columns.

For a full small-system pool, the regression suite verifies that the restricted
master exactly matches `joint_fusion_selection_oracle()` on every objective
tier.

The complete mathematical model, reduced cost, algorithm boundary, and
claim--evidence map are specified in `V12_THEORY_MODEL_ALGORITHM.md`.
