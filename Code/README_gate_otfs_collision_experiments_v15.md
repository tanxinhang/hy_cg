# gate_otfs_collision_experiments_v15

V15 is a stability patch on top of V14.  It keeps the V14 collision-aware assignment, DD-RWIF/KF tracking backend, biased-peak/clutter-peak measurement pollution, temporal hysteresis, and measurement trace diagnostics, but fixes the main failure mode found from the V14 trace analysis:

> A biased or high-collision measurement can report a very large `pos_std`.  The usual Mahalanobis gate then uses this large covariance in `S = HPH^T + R`, so kilometer-level residuals can still have a small normalized innovation and pass the gate.

## New robust gate controls

All new thresholds are disabled when set to `0` or a non-positive value.

- `--kf-abs-pos-gate-m`: direct xy-position residual gate in meters.
- `--kf-abs-vel-gate-mps`: direct xy-velocity residual gate in m/s.
- `--kf-gate-pos-std-cap-m`: caps `pos_std` only inside the Mahalanobis gate.  The actual update still uses the raw covariance unless the gate action modifies it.
- `--kf-gate-vel-std-cap-mps`: caps `vel_std` only inside the Mahalanobis gate.
- `--kf-reject-pos-std-m`: gates detected measurements whose self-reported `pos_std` is too large.
- `--kf-reject-cross-norm`: gates detected measurements whose `full_cross_norm` exceeds the threshold.

The per-target trace now includes:

- `abs_pos_resid_m`
- `abs_vel_resid_mps`
- `gate_reason`
- `gate_reason_abs_pos`
- `gate_reason_abs_vel`
- `gate_reason_pos_std`
- `gate_reason_cross`
- `gate_reason_maha`

The frame summary now includes gate reason counts and absolute residual summaries:

- `abs_pos_gate_count`
- `pos_std_gate_count`
- `cross_gate_count`
- `maha_gate_count`
- `abs_pos_resid_mean_m`
- `abs_pos_resid_p95_m`
- `abs_pos_resid_max_m`

## Recommended first run

```bash
python gate_otfs_collision_experiments_v15.py  --gate tracking  --tracking-mc 20  --num-frames 30  --inner-mc 50   --tracking-M 15  --tracking-Q 10  --eval-kernel-mode dirichlet  --target-pd-fusion or  --tracking-reinit-lost  --tracker-mode dd_rwif   --meas-pollution biased_peak  --pollution-pmax 0.35  --assignment-hysteresis  --hys-abs-margin 0.05  --kf-innovation-gate  --kf-gate-action skip   --kf-gate-chi2 13.3  --kf-abs-pos-gate-m 500   --kf-gate-pos-std-cap-m 500  --kf-reject-pos-std-m 1200  --save-measurement-trace  --num-workers 10  --fresh-out-dir   --out-dir tracking_v15_rwif_polluted_hys_absgate
```

## Suggested ablations

Start from the command above and vary only one group at a time:

- Absolute position gate: `--kf-abs-pos-gate-m 300,500,800,0`
- Gate covariance cap: `--kf-gate-pos-std-cap-m 300,500,1000,0`
- Self-reported std rejection: `--kf-reject-pos-std-m 800,1200,2000,0`
- Gate action: `--kf-gate-action skip` vs `--kf-gate-action inflate`
- Hysteresis: `--hys-abs-margin 0,0.02,0.05,0.1`

The target is not only lower mean RMSE.  Check whether the catastrophic tail is suppressed:

- `max_track_pos_error_m`
- count of trial-frame errors above 500 m / 1000 m / 2000 m
- `gated_measurement_rate`
- gate reason counts
- `p05_pd_target`
- `track_loss_rate`
- `switch_rate`
