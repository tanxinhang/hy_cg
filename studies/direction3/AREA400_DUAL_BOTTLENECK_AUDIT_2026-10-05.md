# 400 m dual-bottleneck receiver audit (2026-10-05)

## Decision

The 400 m regime has two separate bottlenecks and they must not share one
headline metric:

1. **cancellation robustness** is the held-out gap from TP-UIC to the
   perfect-channel receiver at the same detector and CPI count;
2. **detection capability** is the absolute perfect-channel AUC/PD and its
   improvement as independent looks are accumulated.

The new factorial gate fixes both axes explicitly: direct-interference boosts
of +10/+30/+50 dB, TP-UIC versus perfect-channel cancellation, and 1/2/4 CPI
noncoherent accumulation.  Every arm/count cell receives its own calibration
threshold and scenes remain the exchangeability unit.

## High-interference cancellation screen

The first candidate attacked the nonlinear source of the growing gap rather
than changing a cancellation weight.  It searched a local delay--Doppler grid
after projecting the observation and candidate direct dictionaries outside the
believed target-protection subspace, then rebuilt the normal TP-UIC dictionary.
The search uses no simulator-truth fields; a metamorphic test changes
`x_direct` and `h_true` and verifies an identical refined dictionary.

At 400 m / +50 dB, on a new seed and 24 held-out scenes:

| Receiver | Test AUC | Gap to perfect channel |
|---|---:|---:|
| TP-UIC | 0.550 | 0.064 |
| Single-look nonlinear-refined TP-UIC | 0.543 | 0.071 |
| Perfect channel | 0.615 | 0.000 |

The candidate changes the statistic materially but makes the ordering slightly
worse (AUC delta -0.007).  It is therefore **rejected as a single-look
receiver** and must not be wired into production.  The likely failure mode is
data reuse: a single noisy look selects a nuisance location and then tests the
same look, so the location search can follow noise or imperfectly protected
target structure.  Its roughly five-by-five grid cost is also unsuitable for
production.

An independent low-interference check at 400 m / +10 dB used master seed
`20261006` and another 24 held-out scenes:

| Receiver | Test AUC | Gap to perfect channel |
|---|---:|---:|
| TP-UIC | 0.630 | 0.066 |
| Single-look nonlinear-refined TP-UIC | 0.653 | 0.043 |
| Perfect channel | 0.696 | 0.000 |

The refined receiver gains +0.023 AUC at this point and closes 0.023 of the
oracle gap.  A same-scene paired bootstrap gives a 95% interval of
[-0.024, +0.085] for the AUC change, so the sign is not established.  Together
with the -0.007 point change at +50 dB, this is evidence of a regime-dependent
effect, not a promotable general robustness improvement.  The single-look
variant remains rejected; the low-INR result is retained as motivation for
cross-fitted multi-look DD refinement rather than discarded.

## Detection branch

The prior independent result remains the positive detector direction.  At
400 m / +10 dB, two independent TP-UIC looks increased test AUC from 0.674 to
0.739 and PD from 0.225 to 0.525, at approximately twice the observation and
receiver cost.  This improves evidence supply but does not by itself close a
high-INR cancellation gap.

## Next gate

The next cancellation candidate should share one DD state across multiple
looks while fitting an independent complex gain per look.  The DD state must
be estimated on training/reference looks or by cross-fitting, and evaluated on
a held-out look; this removes the single-look data-reuse failure.  Replace the
grid with coarse-to-fine or Gauss--Newton only after that statistical gate
passes.

### Initial hierarchical-MAP cross-fit screen

That next candidate was implemented as a two-fold screen.  CPI 1 estimates the
shared DD state and CPI 2 supplies a held-out detection statistic; the roles
are then reversed and the two held-out statistics are summed.  Each reference
look profiles an independent complex direct-path gain, while DD displacement
has a Gaussian prior.  The equal-cost baseline sums two ordinary TP-UIC looks.

On a new seed (`20261007`), with eight held-out scenes per interference level:

| Boost | Ordinary two-CPI TP-UIC | Cross-fitted hierarchical MAP | Perfect-channel two-CPI |
|---|---:|---:|---:|
| +10 dB | 0.984 | 1.000 | 0.953 |
| +50 dB | 0.828 | 0.969 | 0.969 |

The same-scene AUC changes are +0.016 at +10 dB (bootstrap interval
[0.000, 0.063]) and +0.141 at +50 dB ([0.000, 0.281]).  The +50 dB point closes
the observed oracle gap completely and reverses the single-look failure, so
the hierarchical/cross-fit structure passes the **directional screen**.
Eight scenes are far too few for promotion, perfect-channel need not rank first
in a finite sample, and two calibration scenes cannot support a finite 5% tail
threshold.  No PD/PFA claim is made.  The current two-way 5x5 coordinate grid
is also a correctness prototype, not a viable production solver.

Decision: expand the +30/+50 dB test with a real calibration partition; only
then replace the grid with shared-dictionary Gauss--Newton and measure runtime.

### Formal +30/+50 dB confirmation

The fixed candidate was rerun with master seed `20261008`, separately at each
interference level, using exactly 1 train, 40 calibration and 40 held-out test
scenes.  Each receiver received its own 5% split-conformal H0 threshold.

| Boost | Receiver | Threshold | AUC | PFA | PD | Oracle AUC gap | Oracle PD gap |
|---|---|---:|---:|---:|---:|---:|---:|
| +30 dB | ordinary two-CPI | 26.327 | 0.727 | 0.000 | 0.125 | 0.027 | 0.125 |
| +30 dB | cross-fitted MAP | 23.023 | 0.748 | 0.000 | 0.175 | 0.006 | 0.075 |
| +30 dB | perfect channel | 22.838 | 0.754 | 0.025 | 0.250 | 0.000 | 0.000 |
| +50 dB | ordinary two-CPI | 119.600 | 0.693 | 0.025 | 0.025 | 0.062 | 0.225 |
| +50 dB | cross-fitted MAP | 25.453 | 0.754 | 0.000 | 0.200 | 0.000 | 0.050 |
| +50 dB | perfect channel | 22.838 | 0.754 | 0.025 | 0.250 | 0.000 | 0.000 |

At +30 dB the MAP changes AUC by +0.021 (paired bootstrap 95% interval
[-0.001, +0.054]) and PD by +0.050 ([0.000, +0.125]).  This is a positive but
not decisive trend.  At +50 dB it changes AUC by +0.062 ([-0.021, +0.146]) and
PD by **+0.175** ([+0.075, +0.300]).  The high-interference PD improvement is
the first formally nonzero receiver gain in this sequence.  MAP PFA is 0/40 at
both levels (Wilson upper bound 0.088); this does not indicate inflation.

The +50 dB point estimate closes the AUC oracle gap and reduces the PD oracle
gap from 0.225 to 0.050.  Its oracle-minus-MAP intervals are [-0.050, +0.043]
for AUC and [0.000, +0.125] for PD.  Thus oracle parity in AUC is plausible but
not proven as an identity, while a small residual working-point PD gap remains.

Decision: **the hierarchical/cross-fit MAP passes the formal +50 dB PD gate and
is retained as the high-interference receiver candidate.**  The +30 dB result
does not independently pass a strict superiority gate.  Neither result makes
the 5x5 grid production-ready; the next implementation task is an equivalent
Gauss--Newton/LM solver followed by a numerical-equivalence and runtime gate.

### Bounded Gauss--Newton replacement gate

A trust-region reflective Gauss--Newton solver now profiles the same
look-specific complex gains, minimises the same target-protected MAP residual,
and enforces the same +/-2 sigma DD box.  Four function evaluations were fixed
from an independent three-scene runtime probe before formal confirmation.

The matched +50 dB microbenchmark gives a median pure-solver time of 0.246 s
for GN4 versus 0.551 s for the 5x5 grid, a **2.24x solver speedup**.  Every
probed H0/H1 GN4 objective was lower than the coarse-grid objective.  The gain
is not an end-to-end 2.24x speedup: TP-UIC, covariance construction and GLRT
remain common fixed costs.

On the same formal seed and 1/40/40 split used above, GN4 produces:

| Solver | AUC | PFA | PD | Threshold | Median cross-fit stage / scene |
|---|---:|---:|---:|---:|---:|
| 5x5 grid | 0.754 | 0.000 | 0.200 | 25.453 | not instrumented in frozen run |
| GN4 | 0.758 | 0.000 | 0.200 | 23.448 | 11.204 s |

The paired GN4-minus-grid AUC change is +0.0038 with a 95% interval of
[-0.0244, +0.0294]; the PD change is exactly 0 with interval [0, 0].  Final
statistics have Pearson correlation 0.988 and median relative difference
1.63%.  GN4 therefore passes numerical/detection non-degradation and pure MAP
runtime gates.  It replaces the grid as the experimental solver default.

The broader receiver is **not yet runtime-closed**: the formal GN4 cross-fit
stage still costs about 11.2 s per scene in this Python harness.  The next
runtime target is shared dictionary/Jacobian caching and analytic manifold
Jacobians; claiming a production-ready end-to-end acceleration now would be
incorrect.

The next detector candidate remains multi-look evidence accumulation, with
1/2/4-look latency curves and a larger calibration partition.  Promotion
requires both:

- a smaller TP-UIC-to-perfect AUC gap at +30 and +50 dB, without lower target
  survival or inflated held-out PFA;
- materially higher perfect-channel and TP-UIC PD at the calibrated operating
  point, reported together with observation/compute cost.

Machine evidence:

- `data/area400_boost50_refinement_screen_20261005/`
- `data/area400_boost10_refinement_screen_20261006/`
- `data/area400_crossfit_map_screen_20261007/`
- `data/area400_crossfit_map_boost30_formal_20261008/`
- `data/area400_crossfit_map_boost50_formal_20261008/`
- `data/area400_crossfit_map_formal_analysis_20261008/summary.json`
- `data/area400_crossfit_gn_nfev4_boost50_formal_20261008/`
- `data/joint_dd_solver_benchmark_nfev4_20261009/`
- `tools/gate_area400_dual_axis.py`
