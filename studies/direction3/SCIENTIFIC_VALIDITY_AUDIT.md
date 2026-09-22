# Scientific validity audit of the joint TP-UIC scheduler

## Decision

The current `0.9961` worst-target detection probability is **not admissible as
a mainline performance result**.  It is an oracle structural upper bound.  It
must not be presented as an achieved gain of TP-UIC, communication scheduling,
or power optimization.

## Unsupported or potentially inflated quantities

1. **Truth-assisted formation**

   `target_ring_formation` places UAVs using the true target positions.  The
   tested run used no movement limit and moved the fleet by 2589.05 m in total.
   This is not a feasible online coordination policy and dominates the reported
   gain.  Without this oracle formation, the otherwise similar diagnostic run
   had worst delivered PD near 0.099 rather than 0.996.

2. **Heuristic cross-UAV correlation factor**

   Dividing every signal eigenvalue by `1 + rho * (K - 1)` is a sensitivity
   heuristic, not a result derived from a joint receiver covariance.  Values
   produced for `rho = 0.1, 0.25, 0.5` have no calibrated physical meaning and
   may only be labelled qualitative stress tests.  They must not be claimed as
   predicted system performance.

3. **Moment-matched Gaussian PD**

   Detection probability is obtained from Gaussian moments of the covariance
   LLR rather than an exact finite-sample distribution or empirical ROC.  Its
   calibration at high PD and in the tail has not been demonstrated.  Values
   close to one therefore require Monte Carlo/CFAR validation with confidence
   intervals.

4. **Independence assumptions**

   The fusion routine concatenates receiver eigenmodes and treats receiver
   evidence as independent.  Packet erasures are also independently mixed.
   Shared transmit signals, common clutter/interference, geometry error, and
   correlated blockage can violate both assumptions.  The present calculation
   is optimistic until a joint covariance or paired empirical detector is used.

5. **Same-instance optimization and reporting**

   The explicit heterogeneous power vector, formation, activity mask,
   protection sets, fusion nodes, and reports are optimized/evaluated on one
   trial.  This is an in-sample result.  It does not establish expected or
   robust performance across geometry, target state, channel, and belief error.

6. **Reproducibility failure**

   Reusing trial 32 did not reproduce the earlier formation movement
   (2589.05 m versus 2238.67 m).  A trial identifier is insufficient provenance
   when code/configuration changes.  Geometry, belief, channel realization,
   configuration, and code revision must be saved for paired comparisons.

7. **Misleading output label**

   When `coordination_aware_ao` is enabled, `optimized_information` contains
   delivered probabilities, not information.  This field must be renamed or
   accompanied by an objective-type field before results are consumed.

## Components that are presently reasonable

- The per-UAV constraint `P_sense + P_comm <= 1 W` is explicitly checked.
- One-shot finite-blocklength success/erasure fusion is internally consistent
  with the declared absence of ACK/retransmission, subject to its independence
  assumption.
- Sender capacity, fusion-node capacity, link feasibility, and serial latency
  are enforced as hard constraints.
- Add/drop/swap with monotone acceptance is a reasonable local-search method,
  but only relative to the validity of the PD objective it is given.

## Minimum evidence required before a performance claim

1. Freeze and replay identical geometry/channel/belief snapshots for every arm.
2. Separate oracle formation from feasible belief-based motion with a stated
   per-UAV movement, speed, and time budget.
3. Compare no-IC, Stage-1, full TP-UIC, and oracle cancellation on paired seeds.
4. Validate analytic PD against empirical ROC/CFAR Monte Carlo and report
   confidence intervals.
5. Estimate or construct the cross-UAV joint covariance; remove the scalar
   design-effect values from headline tables.
6. Learn power/scheduling decisions on training scenarios and report held-out
   performance over multiple target counts and 400/600 m areas.

Until these gates pass, `0.9961` is retained only as a truth-assisted geometric
upper bound and the previously observed TP-UIC incremental gain remains
approximately negligible in the audited scenario.
