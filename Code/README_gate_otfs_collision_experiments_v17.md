# gate_otfs_collision_experiments_v17

V17 extends V16 with a PDA/GPDA-like probabilistic DD-peak association tracking backend for comparison with deterministic KF/DD-RWIF tracking updates.

## What is new in V17

1. **PDA/GPDA-like tracking backends**
   - `--tracker-mode gpda`, `pda`, or `jpda`: probabilistic candidate association using KF candidate updates.
   - `--tracker-mode dd_gpda`, `dd_pda`, or `dd_jpda`: the same candidate association, but with DD-RWIF candidate updates.

   In the current simulator, target identities are fixed and the measurement model is not a full multi-target clutter/association problem. Therefore this is a **PDA/JPDA-like DD peak association baseline**, not a full JPDA implementation over all target-track association hypotheses. It is intended as a fair standard-style association opponent for the biased/cluttered DD peak pollution model.

2. **Candidate set for structured DD pollution**
   - Under `biased_peak` or `clutter_peak` pollution, if a polluted peak is realized and `--no-gpda-clean-candidate` is not set, the GPDA-like backend is given a small candidate set containing:
     - the clean target peak;
     - the polluted biased/clutter peak.
   - Candidate priors are weighted by the realized pollution probability.
   - A missed/no-valid-measurement hypothesis is included through `--gpda-missed-prior`.

3. **GPDA diagnostics**
   Added to `tracking_summary.csv` and `tracking_measurement_trace.csv`:
   - `gpda_candidate_count_mean`
   - `gpda_best_assoc_prob_mean`
   - `gpda_miss_assoc_prob_mean`
   - `gpda_assoc_entropy_mean`

## Recommended comparison

Run the deterministic backend:

```bash
python gate_otfs_collision_experiments_v17.py ^
  --gate tracking ^
  --tracking-mc 100 ^
  --num-frames 30 ^
  --inner-mc 50 ^
  --tracking-M 15 ^
  --tracking-Q 10 ^
  --eval-kernel-mode dirichlet ^
  --target-pd-fusion or ^
  --tracking-reinit-lost ^
  --tracker-mode dd_rwif ^
  --meas-pollution biased_peak ^
  --pollution-pmax 0.35 ^
  --assignment-hysteresis ^
  --hys-abs-margin 0.05 ^
  --kf-innovation-gate ^
  --kf-gate-action skip ^
  --kf-gate-chi2 13.3 ^
  --kf-abs-pos-gate-m 500 ^
  --kf-gate-pos-std-cap-m 500 ^
  --kf-reject-pos-std-m 1200 ^
  --include-global-assignment-baselines ^
  --save-measurement-trace ^
  --num-workers 10 ^
  --fresh-out-dir ^
  --out-dir tracking_v17_ddrwif_main
```

Run the probabilistic association backend:

```bash
python gate_otfs_collision_experiments_v17.py ^
  --gate tracking ^
  --tracking-mc 100 ^
  --num-frames 30 ^
  --inner-mc 50 ^
  --tracking-M 15 ^
  --tracking-Q 10 ^
  --eval-kernel-mode dirichlet ^
  --target-pd-fusion or ^
  --tracking-reinit-lost ^
  --tracker-mode dd_gpda ^
  --meas-pollution biased_peak ^
  --pollution-pmax 0.35 ^
  --assignment-hysteresis ^
  --hys-abs-margin 0.05 ^
  --kf-innovation-gate ^
  --kf-gate-action skip ^
  --kf-gate-chi2 13.3 ^
  --kf-abs-pos-gate-m 500 ^
  --kf-gate-pos-std-cap-m 500 ^
  --kf-reject-pos-std-m 1200 ^
  --include-global-assignment-baselines ^
  --save-measurement-trace ^
  --num-workers 10 ^
  --fresh-out-dir ^
  --out-dir tracking_v17_ddgpda_main
```

## Notes

- `gpda`/`pda`/`jpda` are aliases for the KF-based probabilistic candidate association backend.
- `dd_gpda`/`dd_pda`/`dd_jpda` are aliases for the DD-RWIF-based probabilistic candidate association backend.
- Use this backend as a standard association baseline/ablation, not as the primary claimed contribution unless a full clutter candidate model is later added.
