# gate_otfs_collision_experiments_v18

V18 extends V17 with a **shared-measurement JPDA-like backend** for the closed-loop tracking gate.

## What changed from V17

V17 provided `pda/gpda` and `dd_pda/dd_gpda` as **single-track probabilistic DD-peak association** backends. They are useful ablations, but they are not full JPDA because each track receives its own candidate set and no cross-track measurement exclusivity is enforced.

V18 adds:

- `jpda_shared` and `dd_jpda_shared` tracker modes.
- `jpda` and `dd_jpda` are aliases for the shared-measurement JPDA-like backend.
- A scene-level shared measurement pool per frame.
- Multi-track validation against all shared measurements.
- Exact joint-association enumeration per connected validation component.
- One-track-at-most-one-measurement and one-measurement-at-most-one-track constraints.
- Marginal association probabilities and moment-matched KF/DD-RWIF updates.
- JPDA diagnostics:
  - `jpda_joint_event_count`
  - `jpda_exact_component_frac`
  - `jpda_num_measurements`
  - existing GPDA-style fields: `gpda_candidate_count_mean`, `gpda_best_assoc_prob_mean`, `gpda_miss_assoc_prob_mean`, `gpda_assoc_entropy_mean`

If a connected component has too many joint events, V18 falls back to row-wise PDA-like marginals for that component and records this via `jpda_exact_component_frac`.

## Tracker mode meanings

| mode | meaning |
|---|---|
| `kf` | deterministic single-measurement CV-KF |
| `dd_rwif` | DD reliability-weighted information-filter update |
| `pda`, `gpda` | V17 single-track PDA/GPDA-like peak association |
| `dd_pda`, `dd_gpda` | DD-weighted single-track PDA/GPDA-like peak association |
| `jpda`, `jpda_shared` | V18 shared-measurement JPDA-like joint association |
| `dd_jpda`, `dd_jpda_shared` | V18 DD-weighted shared-measurement JPDA-like joint association |

## Recommended smoke test

```bash
python gate_otfs_collision_experiments_v18.py   --gate tracking   --tracking-mc 1   --num-frames 2   --inner-mc 5   --tracking-M 6   --tracking-Q 3   --k-tgt-min 1   --k-uav-max 2   --k-cand 3   --eval-kernel-mode dirichlet   --target-pd-fusion or   --tracking-reinit-lost   --tracker-mode jpda_shared   --meas-pollution biased_peak   --pollution-pmax 0.35   --assignment-hysteresis   --kf-innovation-gate   --kf-gate-action skip   --kf-abs-pos-gate-m 500   --kf-gate-pos-std-cap-m 500   --kf-reject-pos-std-m 1200   --include-global-assignment-baselines   --save-measurement-trace   --fresh-out-dir   --out-dir v18_smoke_jpda
```

## Recommended main JPDA-like comparison

```bash
python gate_otfs_collision_experiments_v18.py   --gate tracking   --tracking-mc 100   --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10   --eval-kernel-mode dirichlet   --target-pd-fusion or   --tracking-reinit-lost   --tracker-mode dd_jpda_shared   --meas-pollution biased_peak   --pollution-pmax 0.35   --assignment-hysteresis   --hys-abs-margin 0.05   --kf-innovation-gate   --kf-gate-action skip   --kf-gate-chi2 13.3   --kf-abs-pos-gate-m 500   --kf-gate-pos-std-cap-m 500   --kf-reject-pos-std-m 1200   --include-global-assignment-baselines   --save-measurement-trace   --num-workers 10   --fresh-out-dir   --out-dir tracking_v18_ddjpda_shared_main
```

Analyze with:

```bash
python analyze_tracking_stats_v18.py   --out-dir tracking_v18_ddjpda_shared_main   --start-frame 10   --baseline global_utility   --methods nearest strongest global_utility collision_aware global_utility_hys collision_aware_hys
```

## Important wording

V18 is still a compact research simulator, not a production-grade radar tracker. The new backend is best described as a **shared-measurement JPDA-like baseline**. It now includes the defining JPDA mechanism that V17 lacked: joint association over a shared measurement pool with measurement exclusivity.

## Gate-1：碰撞到检测退化
```bash
python gate_otfs_collision_experiments_v18.py   --gate gate1   --gate1-mc 100   --inner-mc 200   --eval-kernel-mode actual_otfs   --proxy-kernel-mode dirichlet   --target-pd-fusion or   --fresh-out-dir   --out-dir gate1_actual_otfs_mc100
```