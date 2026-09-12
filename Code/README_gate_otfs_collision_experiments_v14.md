# Gate OTFS Collision Experiments V14

V14 is a maintenance/stabilization update on top of V13.  It keeps the V12/V13 tracking-filter and pollution-ablation framework, but fixes several logic issues that affected interpretation of measurement-pollution and temporal-hysteresis experiments.

## Main fixes

1. **No-pollution mode reports zero pollution probability.**
   When `--meas-pollution none` is used, the pollution probability and realized polluted-measurement rate are both zero.  The earlier hypothetical nonzero probability is no longer reported as an active pollution statistic.

2. **Structured-pollution directions are normalized inside the pollution function.**
   This makes CRN and non-CRN modes consistent.  The configured bias magnitudes are now the actual intended magnitudes in the xy plane.

3. **Assignment hysteresis is now a separate method.**
   Enabling `--assignment-hysteresis` adds a `collision_aware_hys` curve instead of silently changing the original `collision_aware` curve.  This keeps plots and ablations interpretable.

4. **Hysteresis is coverage-safe and feasibility-aware.**
   The previous assignment is kept if the new assignment is infeasible or has worse coverage under `hys_force_coverage`.  A new assignment that improves coverage is accepted.

5. **Lost-track reinitialization no longer bypasses innovation gating.**
   With `--kf-innovation-gate`, reinitialization measurements are checked before a hard reset.  If the measurement is gated and the action is `inflate`, a weak robust update is used; if the action is `skip`, the prediction is kept.  Use `--no-gate-reinit-measurement` only for ablation.

6. **Pollution diagnostics are clearer.**
   New/renamed fields include:
   - `pollution_prob_detected_mean`
   - `pollution_prob_detected_over_all_targets`
   - `pollution_prob_potential_all_mean`
   - `pollution_expected_count_detected`

7. **Innovation-gate default threshold adjusted.**
   The default is now `--kf-gate-chi2 13.3`, approximately a 99% threshold for the effective 4-D xy/vx-vy measurement used in this simplified tracker.

## Recommended tracking command

```bash
python gate_otfs_collision_experiments_v14.py   --gate tracking  --tracking-mc 20   --num-frames 30  --inner-mc 50   --tracking-M 15   --tracking-Q 10  --eval-kernel-mode dirichlet   --target-pd-fusion or  --tracking-reinit-lost   --tracker-mode dd_rwif  --meas-pollution biased_peak  --pollution-pmax 0.35  --assignment-hysteresis  --hys-abs-margin 0.05   --kf-innovation-gate  --kf-gate-chi2 13.3  --kf-gate-action inflate  --kf-outlier-r-inflate 50 --save-measurement-trace --num-workers 10  --fresh-out-dir  --out-dir tracking_v14_rwif_polluted_hys_gate
```

## Suggested ablations

- Compare `collision_aware` vs `collision_aware_hys` to isolate hysteresis.
- Compare `--meas-pollution none` vs `biased_peak` to isolate structured measurement pollution.
- Sweep `--hys-abs-margin 0,0.02,0.05,0.1` and `--kf-gate-chi2 9.5,13.3,16.8`.
- Use `--save-measurement-trace` to diagnose catastrophic outliers.

