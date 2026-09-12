# gate_otfs_collision_experiments_v16

V16 builds on V15.1.  It does not change the robust V15 gate logic; instead it adds two items needed for cleaner tracking evaluation and stronger baselines.

## Main V16 additions

1. **Identity-aware GOSPA-like tracking metric**

   The closed-loop tracking output now includes:

   - `gospa_pos_m`
   - `gospa_loc_m`
   - `gospa_miss_m`
   - `gospa_miss_count`
   - `clipped_pos_rmse_m`

   This metric combines localization error and lost-track penalties so that RMSE and loss rate cannot hide each other.  Since target IDs are fixed in this simulator, this is an identity-aware GOSPA-like score, not a full unlabeled-set GOSPA implementation.

   Default parameters:

   ```bash
   --gospa-cutoff-m 300
   --gospa-p 2
   --gospa-alpha 2
   ```

2. **Collision-unaware global assignment baselines**

   Add:

   ```bash
   --include-global-assignment-baselines
   ```

   This enables:

   - `global_utility`: capacity-constrained global utility assignment, without DD-collision cost.
   - `global_utility_hys`: the same baseline with temporal hysteresis, when `--assignment-hysteresis` is enabled.

   These baselines help distinguish the gain from global assignment itself from the gain of DD-collision awareness.

3. **Updated post-processing script**

   `analyze_tracking_stats_v16.py` supports the V16 GOSPA-like fields and produces paired CI/Wilcoxon tables.

## Suggested main command

```bash
python gate_otfs_collision_experiments_v16.py --gate tracking --tracking-mc 100 --num-frames 30 --inner-mc 50 --tracking-M 15 --tracking-Q 10 --eval-kernel-mode dirichlet --target-pd-fusion or --tracking-reinit-lost --tracker-mode dd_rwif --meas-pollution biased_peak --pollution-pmax 0.35 --assignment-hysteresis --hys-abs-margin 0.05 --kf-innovation-gate --kf-gate-action skip --kf-gate-chi2 13.3 --kf-abs-pos-gate-m 500 --kf-gate-pos-std-cap-m 500 --kf-reject-pos-std-m 1200 --include-global-assignment-baselines --save-measurement-trace --num-workers 10 --fresh-out-dir --out-dir tracking_v16_mc100_main
```

## Post-processing

```bash
python analyze_tracking_stats_v16.py 
  --out-dir tracking_v16_mc100_main 
  --start-frame 10 
  --baseline strongest 
  --methods nearest strongest global_utility collision_aware global_utility_hys collision_aware_hys
```

Outputs are written to `tracking_v16_mc100_main/analysis_stats/`.

