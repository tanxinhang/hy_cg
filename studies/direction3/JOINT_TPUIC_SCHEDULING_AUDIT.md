# Stepwise TP-UIC and coordination audit

## Gate 1: delivered-PD-aware AO and feasible-link enforcement

The AO objective was changed experimentally from ideal summed Jeffreys
information to the actual FBL-erasure delivered PD. Reporting links now have to
meet edge reachability, `R_min` and (when enabled) `chi_min` before admission.

| Method | Worst PD | Reports | Minimum selected chi |
|---|---:|---:|---:|
| Previous information-sum AO | 0.878642 | 4 | 0.0543 |
| Delivered-PD-aware AO | 0.878642 | 3 | approximately 1.0 |

Verdict: pass for constraint correctness and efficiency; neutral for the
bottleneck PD. The low-reliability, non-bottleneck report was removed.

## Gate 2: joint fusion-node search

Enumerating all 216 fusion assignments after AO convergence changed the target
2 fusion node and raised its PD from 0.969532 to 0.980315, but target 3 stayed
at 0.878642. Report count is retained only as the last tie-break after both
worst-target and total PD; placing it earlier can block useful intermediate
actions in the joint AO loop.

Embedding all 216 assignments inside every AO candidate was stopped after
excessive runtime. The retained architecture is two-level: fast constrained
scheduling inside AO, exact fusion search once after AO convergence.

## Gate 3: receiver-arm ablation at the accepted round-2 power

| Receiver arm | Worst delivered PD | Reports | Looks for PD >= 0.8 |
|---|---:|---:|---:|
| no IC | 0.878583 | 3 | 19 |
| TP-UIC Stage 1 | 0.878642 | 4 | 19 |
| TP-UIC full | 0.878642 | 3 | 19 |
| perfect channel | 0.878620 | 3 | 19 |

Verdict: TP-UIC is not the active bottleneck in this scenario. Full improves
the bottleneck over no-IC by only about 0.000060, and perfect cancellation does
not materially change the answer. The large joint-loop gain must be attributed
to power/activity/coordination, not to cancellation.

## Gate 4: no-ACK local scheduling, latency, and correlation sensitivity

> Scientific-validity correction: the numerical result in this section is an
> oracle, truth-assisted formation upper bound, not a mainline achieved result.
> The correlation sweep is an uncalibrated sensitivity heuristic.  See
> `SCIENTIFIC_VALIDITY_AUDIT.md` before citing any value below.

- Communication remains one-shot success/erasure fusion. ACK and
  retransmission are deliberately not modeled.
- Scheduling now refines constructive admission with add/drop/swap moves.
  Acceptance is ordered by worst-target delivered PD, total delivered PD, and
  finally report count. Putting report count before total PD was tested and
  rejected because it trapped AO at a poor intermediate solution.
- Serial-MAC block latency is enforced against the 10 ms fusion window.
- Cross-UAV correlation is included only as a design-effect sensitivity test;
  it is not yet a production joint-covariance model.

For the 600 m, 6-UAV/3-target, two-views-per-target, 512-channel-use,
25-look case with each UAV capped at 1 W, the rerun obtained delivered PD
`[1.0000, 0.9961, 0.9983]`. Four reports consumed 1.067 ms, below the 10 ms
budget. Increasing the assumed correlation from 0 to 0.1, 0.25 and 0.5 reduced
the worst PD from 0.9961 to approximately 0.985, 0.97 and 0.9296.

This run's formation movement was 2589.05 m, whereas the earlier result file
records 2238.67 m. Absolute PD values are therefore not a strict paired
comparison. Production experiments should save and reload an explicit
geometry snapshot rather than rely only on a trial number and derived
formation routine.

## Remaining gates

- Replace the correlation sensitivity approximation with a joint covariance
  or empirical-ROC model.
- Save/reuse geometry snapshots for paired algorithm comparisons.
- Automate per-UAV power iteration only after those reproducibility gates.
