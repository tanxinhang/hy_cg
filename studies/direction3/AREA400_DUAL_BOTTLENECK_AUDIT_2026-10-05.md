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

The next detector candidate remains multi-look evidence accumulation, with
1/2/4-look latency curves and a larger calibration partition.  Promotion
requires both:

- a smaller TP-UIC-to-perfect AUC gap at +30 and +50 dB, without lower target
  survival or inflated held-out PFA;
- materially higher perfect-channel and TP-UIC PD at the calibrated operating
  point, reported together with observation/compute cost.

Machine evidence:

- `data/area400_boost50_refinement_screen_20261005/`
- `tools/gate_area400_dual_axis.py`
